# PLS regression analysis for E. coli mRNA data
# Computes PLS coefficients, G+/G- projection scores,
# and returns diagnostic plots.

import numpy as np
import pandas as pd

import bokeh.io
import bokeh.plotting
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, ColorBar
from bokeh.transform import linear_cmap
from bokeh.palettes import Magma256

from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import pearsonr


def run_pls(df, t, condition):
    """
    Fit PLS regression on mRNA data, compute G+/G- projection scores,
    and return diagnostic plots.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame: rows = samples, columns = genes + 'growth_rate'.
    t : float
        Threshold for defining G+ (beta > t) and G- (beta < -t) gene sets.
    condition: str
        E. coli condition label for legend titles

    Returns
    -------
    beta_df : pd.DataFrame
        Gene-level PLS coefficients.
    p_scatter : bokeh figure
        G- vs G+ scatter plot, colored by growth rate.
    p_hist : bokeh figure
        Overall beta coefficient histogram.
    """
    y = df['growth_rate'].values
    X = df.drop(columns=['growth_rate']).values
    gene_names = df.drop(columns=['growth_rate']).columns.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pls_model = PLSRegression(n_components=2)
    pls_model.fit(X_scaled, y)

    beta = pls_model.coef_.flatten()

    # --- Beta coefficient dataframe ---
    beta_df = pd.DataFrame({'gene': gene_names, 'beta': beta})

    # --- Beta histogram ---
    p_hist = bokeh.plotting.figure(
        width=400, height=350,
        x_axis_label='beta values',
        title='PLS coefficient distribution: ' + condition
    )
    hist, edges = np.histogram(beta, bins=50)
    p_hist.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                fill_color=None, line_color='black')
    p_hist.xaxis.major_label_text_font_size = '12pt'
    p_hist.yaxis.major_label_text_font_size = '12pt'
    p_hist.xaxis.ticker.desired_num_ticks = 4
    p_hist.yaxis.ticker.desired_num_ticks = 5
    p_hist.xgrid.visible = False
    p_hist.ygrid.visible = False
    p_hist.outline_line_color = None
    p_hist.background_fill_color = None


    # --- G+ / G- projection scores ---
    pos_mask = beta > t
    neg_mask = beta < -t

    beta_pos = beta.copy()
    beta_pos[~pos_mask] = 0

    beta_neg = np.abs(beta.copy())
    beta_neg[~neg_mask] = 0

    pos_axis = (X_scaled @ beta_pos) / np.sum(pos_mask)
    neg_axis = (X_scaled @ beta_neg) / np.sum(neg_mask)

    print('G+/G- correlation:', pearsonr(pos_axis, neg_axis)[0],
          '| G+ genes:', np.sum(pos_mask), '| G- genes:', np.sum(neg_mask))

    # --- Scatter plot: G- vs G+, colored by growth rate ---
    source = ColumnDataSource(data=dict(
        neg=neg_axis,
        pos=pos_axis,
        growth_rate=y
    ))

    mapper = linear_cmap(
        field_name='growth_rate',
        palette=Magma256,
        low=min(y),
        high=max(y)
    )

    p_scatter = bokeh.plotting.figure(width=600, height=450, title = condition)
    p_scatter.scatter('neg', 'pos',
                      source=source,
                      fill_color=mapper,
                      line_color='grey',
                      size=10,
                      fill_alpha=0.8)

    color_bar = ColorBar(
        color_mapper=mapper['transform'],
        width=15,
        location=(0, 0),
        title='Growth Rate'
    )
    p_scatter.add_layout(color_bar, 'right')

    p_scatter.xaxis.major_label_text_font_size = '12pt'
    p_scatter.yaxis.major_label_text_font_size = '12pt'
    color_bar.major_label_text_font_size = '12pt'
    p_scatter.xaxis.ticker.desired_num_ticks = 5
    p_scatter.yaxis.ticker.desired_num_ticks = 5

    bokeh.io.show(row(p_scatter, p_hist))

    return beta_df, p_scatter, p_hist
