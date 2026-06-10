# Compute R/M/Q mRNA and protein fractions for S. pombe
# from Cistvan lab data and PomBase GO annotations.
# Annotation logic follows 20260527_model1_dataoverlay-Copy1_mitophagy.ipynb.
#
# Sectors:
#   R (ribosomal)  → structuralConstituentofRibosome_pombase.tsv
#   M (mitophagy)  → mitophagy_pombase.tsv
#   Q (other)      → everything else
#   Priority: R > M > Q
#
# Usage in notebook:
#   from load_spombe_fractions import load_annotations, compute_mrna_fractions,
#                                     compute_protein_fractions
#
#   ribo_ids, mito_ids = load_annotations(
#       '../data/s.pombe/annotations/structuralConstituentofRibosome_pombase.tsv',
#       '../data/s.pombe/annotations/mitophagy_pombase.tsv'
#   )
#
#   mrna_fracs = compute_mrna_fractions(
#       mrna_path='../data/s.pombe/normalized_counts_fromPaper.xlsx',
#       growth_path='../data/s.pombe/growth_info.xlsx',
#       ribo_ids=ribo_ids, mito_ids=mito_ids
#   )
#
#   prot_fracs = compute_protein_fractions(
#       prot_path='../data/s.pombe/proteomics_analysis_istvan.xlsx',
#       growth_path='../data/s.pombe/growth_info.xlsx',
#       ribo_ids=ribo_ids, mito_ids=mito_ids
#   )

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def load_annotations(ribo_tsv, mito_tsv):
    """
    Load PomBase ribosomal and mitophagy gene ID sets.

    Parameters
    ----------
    ribo_tsv : str
        Path to structuralConstituentofRibosome_pombase.tsv
    mito_tsv : str
        Path to mitophagy_pombase.tsv

    Returns
    -------
    ribo_ids : set
        PomBase Systematic IDs for ribosomal genes.
    mito_ids : set
        PomBase Systematic IDs for mitophagy genes.
    """
    ribo = pd.read_csv(ribo_tsv, sep='\t')
    mito = pd.read_csv(mito_tsv, sep='\t')
    return set(ribo['Systematic ID']), set(mito['Systematic ID'])


def _label_gene(gene_id, ribo_ids, mito_ids):
    """Assign R / M / Q label with priority R > M > Q."""
    if gene_id in ribo_ids:
        return 'R'
    elif gene_id in mito_ids:
        return 'M'
    return 'Q'


def _compute_fracs(df, ribo_ids, mito_ids, id_col, value_col):
    """
    Core fraction computation shared by mRNA and protein.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns: id_col, 'sample rep',
        'growth_rate', value_col.
    ribo_ids, mito_ids : set
    id_col : str
        Column with PomBase gene ID ('PomBaseID').
    value_col : str
        Column with expression values ('raw_counts' or 'psi_P').

    Returns
    -------
    fracs : pd.DataFrame
        Wide-format: one row per (sample rep, growth_rate),
        columns R, M, Q, Growth rate.
    """
    df = df.copy()
    df['label'] = df[id_col].apply(_label_gene, ribo_ids=ribo_ids, mito_ids=mito_ids)

    # Sum per label × sample
    df_frac = (
        df.groupby(['label', 'sample rep', 'growth_rate'], as_index=False)
        .agg(group_mass=(value_col, 'sum'))
    )
    totals = (
        df.groupby(['sample rep', 'growth_rate'])[value_col]
        .sum().reset_index(name='total_mass')
    )
    df_frac = df_frac.merge(totals, on=['sample rep', 'growth_rate'])
    df_frac['fraction'] = df_frac['group_mass'] / df_frac['total_mass']

    # Pivot to wide
    fracs = df_frac.pivot(
        index=['sample rep', 'growth_rate'],
        columns='label',
        values='fraction'
    ).reset_index()
    fracs.columns.name = None
    fracs = fracs.rename(columns={'growth_rate': 'Growth rate'})

    # Ensure all three columns exist
    for col in ['R', 'M', 'Q']:
        if col not in fracs.columns:
            fracs[col] = 0.0

    return fracs[['sample rep', 'R', 'M', 'Q', 'Growth rate']].sort_values('Growth rate').reset_index(drop=True)


def compute_mrna_fractions(mrna_path, growth_path, ribo_ids, mito_ids,
                           value_col='raw_counts'):
    """
    Compute R / M / Q mRNA fractions from sequencing data.

    Accepts either:
      - normalized_counts_fromPaper.xlsx  (columns: PomBaseID, medium,
            replicate, raw_counts / normalised_counts / psi_M)
      - df_seq_mrna_long.csv  (columns: PomBaseID, sample rep, raw_counts)

    Parameters
    ----------
    mrna_path : str
        Path to the mRNA data file (.xlsx or .csv).
    growth_path : str
        Path to growth_info.xlsx with columns 'medium', 'replicate',
        'growth_rate'.
    ribo_ids, mito_ids : set
        Output of load_annotations().
    value_col : str
        Which count column to use. Default 'raw_counts'.
        Other options: 'normalised_counts', 'psi_M'.

    Returns
    -------
    fracs : pd.DataFrame
        Columns: 'sample rep', 'R', 'M', 'Q', 'Growth rate'
    """
    if mrna_path.endswith('.xlsx'):
        df = pd.read_excel(mrna_path)
    else:
        df = pd.read_csv(mrna_path, index_col=0)

    # Build 'sample rep' if not already present
    if 'sample rep' not in df.columns:
        df['sample rep'] = df['medium'] + df['replicate'].fillna(0).astype(int).astype(str)

    df_growth = pd.read_excel(growth_path)
    df_growth['sample rep'] = df_growth['medium'] + df_growth['replicate'].astype(str)
    df_growth = df_growth[['sample rep', 'growth_rate']]

    df = df.merge(df_growth, how='left', on='sample rep')
    df = df.sort_values('growth_rate').reset_index(drop=True)

    return _compute_fracs(df, ribo_ids, mito_ids, id_col='PomBaseID', value_col=value_col)


def compute_protein_fractions(prot_path, growth_path, ribo_ids, mito_ids):
    """
    Compute R / M / Q protein fractions from proteomics data.

    Parameters
    ----------
    prot_path : str
        Path to proteomics_analysis_istvan.xlsx — has columns
        'medium', 'replicate', 'PomBaseIDs' (gene ID), 'psi_P' (intensity).
    growth_path : str
        Path to growth_info.xlsx.
    ribo_ids, mito_ids : set
        Output of load_annotations().

    Returns
    -------
    fracs : pd.DataFrame
        Columns: 'sample rep', 'R', 'M', 'Q', 'Growth rate'
    """
    df = pd.read_excel(prot_path, index_col=0)
    df = df.iloc[:-3]  # drop trailing summary rows
    df = df.reset_index().rename(columns={'PomBaseIDs': 'PomBaseID'})
    df['sample rep'] = df['medium'] + df['replicate'].fillna(0).astype(int).astype(str)
    df = df[['sample rep', 'PomBaseID', 'psi_P']]

    df_growth = pd.read_excel(growth_path)
    df_growth['sample rep'] = df_growth['medium'] + df_growth['replicate'].astype(str)
    df_growth = df_growth[['sample rep', 'growth_rate']]

    df = df.merge(df_growth, how='left', on='sample rep')
    df = df.sort_values('growth_rate').reset_index(drop=True)

    return _compute_fracs(df, ribo_ids, mito_ids, id_col='PomBaseID', value_col='psi_P')
