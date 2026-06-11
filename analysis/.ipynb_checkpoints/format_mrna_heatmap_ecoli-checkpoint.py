# Format E. coli mRNA data for log2FC heatmap
# Follows the second heatmap in 20260202_mRNA_ESR.ipynb
#   - log2 fold change relative to the mean of control samples
#   - hierarchical clustering (Ward / Euclidean) to order genes
#   - genes with any NaN/inf across conditions are dropped before clustering
#
# Usage in notebook:
#   from format_mrna_heatmap_ecoli import ORDERED_COLS, format_mrna_heatmap
#
#   df_hwa_mrna = pd.read_excel('../data/e.coli/science.abk2066_table_s3.xlsx', sheet_name=1)
#   df_frac_clus, ordered_genes = format_mrna_heatmap(df_hwa_mrna)
#
#   # Plot with HoloViews:
#   hv.HeatMap(df_frac_clus, kdims=['sample', 'variable'], vdims='value').opts(...)
#      .redim.values(variable=ordered_genes)

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster

# ---------------------------------------------------------------------------
# Default sample ordering (by limitation type, matching original notebook)
# ---------------------------------------------------------------------------
ORDERED_COLS = [
    'c4', 'c4_1', 'c3_1', 'c3', 'c2', 'c2_1', 'c1', 'c1_1', 'c5', 'c0_1',   # carbon-limited
    'a4', 'a3', 'a4_1', 'a3_1', 'a2', 'a2_1', 'a1', 'a1_1',                   # amino-acid-limited
    'r4_1', 'r3_1', 'r5', 'r4', 'r2_1', 'r3', 'r2', 'r1_1', 'r1', 'r0', 'r0_1',  # ribosome-limited
]

# Columns to drop from the raw Table S3 sheet
DROP_COLS = ['locus', 'gene length (nt)', 'a4_1.1']


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def format_mrna_heatmap(
    df_hwa_mrna,
    control_samples=None,
    cluster_threshold=30,
    ordered_cols=None,
    drop_cols=None,
):
    """
    Format E. coli mRNA data for a log2FC HoloViews HeatMap.

    Parameters
    ----------
    df_hwa_mrna : pd.DataFrame
        Raw mRNA table — sheet 1 of science.abk2066_table_s3.xlsx.
        Must have a 'gene' column; all other columns are treated as samples.
    control_samples : list of str, optional
        Sample IDs to average as the fold-change reference.
        Default: ['c5', 'c0_1']  (lowest-growth carbon-limited conditions).
    cluster_threshold : float, optional
        Distance threshold for fcluster (criterion='distance'). Default: 30.
    ordered_cols : list of str, optional
        Sample order for x-axis. Default: ORDERED_COLS (by limitation type).
    drop_cols : list of str, optional
        Extra columns to drop before computing fractions. Default: DROP_COLS.

    Returns
    -------
    df_frac_clus : pd.DataFrame
        Long format with columns: 'variable' (gene), 'sample', 'value' (log2FC).
        Sorted by the provided ordered_cols and by gene cluster.
    ordered_genes : list of str
        Gene names sorted by cluster — pass to .redim.values(variable=ordered_genes).
    """
    if control_samples is None:
        control_samples = ['c5', 'c0_1']
    if ordered_cols is None:
        ordered_cols = ORDERED_COLS
    if drop_cols is None:
        drop_cols = DROP_COLS

    # ── 1. Drop unwanted columns ──────────────────────────────────────────────
    df = df_hwa_mrna.copy()
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # ── 2. Log2 fold change vs control mean ──────────────────────────────────
    # Vectorised: avoids the slow gene×sample double loop in the original
    genes = df['gene'].values
    data = df.drop(columns='gene')

    control_mean = data[control_samples].mean(axis=1).values  # shape (n_genes,)

    # Avoid log(0): set zeros/negatives to NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        log2fc = np.log2(
            data.values / control_mean[:, np.newaxis]
        )
    log2fc = pd.DataFrame(log2fc, columns=data.columns)
    log2fc.replace([np.inf, -np.inf], np.nan, inplace=True)
    log2fc.insert(0, 'gene', genes)

    # ── 3. Set gene as index (already wide: gene × sample) ───────────────────
    df_wide = log2fc.set_index('gene')

    df_filtered = df_wide.dropna()          # drop any gene with a NaN condition
    variable_values = df_filtered.index.values

    # ── 4. Hierarchical clustering ────────────────────────────────────────────
    rna_array = df_filtered.values.astype(float)
    distance_matrix = linkage(rna_array, method='ward', metric='euclidean',
                               optimal_ordering=False)
    clusters = fcluster(distance_matrix, t=cluster_threshold, criterion='distance')

    # ── 5. Sort genes by cluster ──────────────────────────────────────────────
    df_clustered = df_filtered.copy().reset_index()
    df_clustered['clusters'] = clusters
    df_clustered['variable'] = variable_values
    df_clustered = df_clustered.sort_values('clusters')
    ordered_genes = df_clustered['variable'].tolist()

    # ── 6. Melt to long format ────────────────────────────────────────────────
    df_long = pd.melt(df_clustered, id_vars='variable', var_name='sample')
    df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
    df_long = df_long.dropna(subset=['value'])

    # Remove non-sample columns that leaked in after melt (clusters, index, gene)
    df_long = df_long[~df_long['sample'].str.contains('clusters|index|gene',
                                                        case=False, na=False)]

    # ── 7. Sort samples by ordered_cols ──────────────────────────────────────
    order_map = {v: i for i, v in enumerate(ordered_cols)}
    df_long = df_long.sort_values(
        by='sample',
        key=lambda s: s.map(order_map)
    )

    return df_long.reset_index(drop=True), ordered_genes
