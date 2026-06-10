# GSEA bubble plot for E. coli PLS results
# Loads GO term names from a .obo file, takes positive and negative GSEA
# output tables, and plots a bubble chart of the top enriched terms.

import pandas as pd
import bokeh.io
import bokeh.plotting
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, ColorBar, HoverTool
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256


def load_go_mapping(obo_file):
    """
    Parse a GO .obo file and return a dict mapping GO IDs to term names.

    Parameters
    ----------
    obo_file : str
        Path to go-basic.obo (or full go.obo).

    Returns
    -------
    go_dict : dict
        {GO:XXXXXXX: 'term name', ...}
    """
    go_dict = {}

    with open(obo_file, "r") as f:
        current_id = None
        current_name = None

        for line in f:
            line = line.strip()

            if line == "[Term]":
                current_id = None
                current_name = None

            elif line.startswith("id: GO:"):
                current_id = line.split("id: ")[1]

            elif line.startswith("name:"):
                current_name = line.split("name: ")[1]

            elif line == "" and current_id and current_name:
                go_dict[current_id] = current_name

    return go_dict


def plot_gsea(pos, neg, obo_file, n_top=8, plot_width = 1000):
    """
    Plot a bubble chart of the top enriched GO terms from GSEA output.

    Takes the top n_top terms by FDR q-val from each of the positive and
    negative enrichment tables, maps GO IDs to human-readable names, and
    plots NES on the x-axis with bubble size proportional to gene set size
    and color proportional to FDR q-val.

    Parameters
    ----------
    pos : pd.DataFrame
        GSEA output table for positively enriched gene sets.
    neg : pd.DataFrame
        GSEA output table for negatively enriched gene sets.
    obo_file : str
        Path to go-basic.obo for GO ID → name mapping.
    n_top : int, optional
        Number of top terms to show from each direction (default 8).
    plot_width: int, optional
        Width of bubble plot

    Returns
    -------
    p : bokeh figure
        Main bubble chart.
    size_legend_fig : bokeh figure
        Companion figure showing the gene set size legend.
    """
    # --- Standardise column names ---
    for df in [pos, neg]:
        if 'GS<br> follow link to MSigDB' in df.columns:
            df.rename(columns={'GS<br> follow link to MSigDB': 'GO term'}, inplace=True)

    pos = pos[["NAME", "NES", "FDR q-val", "SIZE"]].copy()
    neg = neg[["NAME", "NES", "FDR q-val", "SIZE"]].copy()
    pos["direction"] = "Positive"
    neg["direction"] = "Negative"

    # --- Select top terms by FDR q-val ---
    pos_top = pos.nsmallest(n_top, "FDR q-val")
    neg_top = neg.nsmallest(n_top, "FDR q-val")
    df_plot = pd.concat([pos_top, neg_top], ignore_index=True)

    # --- Map GO IDs to readable names ---
    go_map = load_go_mapping(obo_file)
    df_plot["label"] = df_plot["NAME"].map(go_map).fillna(df_plot["NAME"])
    df_plot["label"] = df_plot["label"].astype(str)

    # --- Scale bubble sizes ---
    df_plot["size_scaled"] = 5 + 15 * (df_plot["SIZE"] / df_plot["SIZE"].max())

    source = ColumnDataSource(df_plot)

    # --- Color mapper: FDR q-val → Viridis ---
    color_mapper = linear_cmap(
        field_name="FDR q-val",
        palette=Viridis256,
        low=df_plot["FDR q-val"].min(),
        high=df_plot["FDR q-val"].max()
    )

    # --- Main bubble chart ---
    p = bokeh.plotting.figure(
        y_range=list(df_plot.sort_values("NES")["label"]),
        height=600,
        width=plot_width,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )

    p.scatter(
        x="NES",
        y="label",
        size="size_scaled",
        source=source,
        fill_color=color_mapper,
        line_color=None,
        fill_alpha=0.8
    )

    # NES = 0 reference line
    p.line([0, 0], [-100, 100], line_dash="dashed")

    # Axes formatting
    p.xaxis.major_label_text_font_size = "14pt"
    p.yaxis.major_label_text_font_size = "15pt"

    # Hover tool
    hover = HoverTool(tooltips=[
        ("GO Term", "@label"),
        ("NES", "@NES"),
        ("FDR q-val", "@{FDR q-val}"),
        ("Size", "@SIZE"),
    ])
    p.add_tools(hover)

    # Color bar for FDR q-val
    color_bar = ColorBar(
        color_mapper=color_mapper['transform'],
        width=15,
        location=(0, 0),
        title="FDR q-val",
        title_text_font_size="10pt",
        major_label_text_font_size="12pt"
    )
    p.add_layout(color_bar, 'right')

    # --- Size legend ---
    size_vals = [df_plot["SIZE"].min(), df_plot["SIZE"].median(), df_plot["SIZE"].max()]
    size_labels = [str(int(s)) for s in size_vals]
    scaled = [5 + 15 * (s / df_plot["SIZE"].max()) for s in size_vals]

    size_legend_fig = bokeh.plotting.figure(
        width=150,
        height=200,
        toolbar_location=None,
        title="Gene Set Size",
        x_range=(-1, 2),
        y_range=size_labels,
    )
    size_legend_fig.scatter(
        x='x', y='y', size='sizes',
        source=ColumnDataSource(dict(x=[0] * 3, y=size_labels, sizes=scaled)),
        fill_color='gray', line_color=None, fill_alpha=0.8
    )
    size_legend_fig.xaxis.visible = False
    size_legend_fig.xgrid.grid_line_color = None
    size_legend_fig.ygrid.grid_line_color = None
    size_legend_fig.outline_line_color = None

    bokeh.io.show(row(p, size_legend_fig))

    return p, size_legend_fig
