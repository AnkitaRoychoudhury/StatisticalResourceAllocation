# Compute R/C/O mRNA and protein fractions for E. coli
# from Balakrishnan et al. (2022) data and Schmidt et al. (2016) COG annotations.
# Annotation logic follows 20260113_annotateHwa_2.ipynb (the source of the
# fractions used in 20260113_model1_hwa.ipynb).
#
# COG letter mapping:
#   R (ribosomal)  → 'J'  (translation, ribosomal structure)
#   C (metabolic)  → 'C','E','F','G','H','P'  (energy, amino acid, carbohydrate,
#                    nucleotide, coenzyme, inorganic ion metabolism)
#   O (other)      → everything else
#
# Usage in notebook:
#   from load_ecoli_fractions import label_genes, compute_fractions
#
#   df_ann = pd.read_csv('../data/e.coli/Schmidt2016_proteomics.csv')
#   gene_to_label = label_genes(df_ann)
#
#   # mRNA fractions (table S3)
#   df_mrna_data   = pd.read_excel('../data/e.coli/science.abk2066_table_s3.xlsx', sheet_name=1)
#   df_mrna_growth = pd.read_excel('../data/e.coli/science.abk2066_table_s3.xlsx', sheet_name=0)
#   mrna_fracs = compute_fractions(df_mrna_data, df_mrna_growth, gene_to_label,
#                                  remove_samples=REMOVE_MRNA)
#
#   # Protein fractions (table S4)
#   df_prot_data   = pd.read_excel('../data/e.coli/science.abk2066_table_s4.xlsx', sheet_name=1)
#   df_prot_growth = pd.read_excel('../data/e.coli/science.abk2066_table_s4.xlsx', sheet_name=0)
#   prot_fracs = compute_fractions(df_prot_data, df_prot_growth, gene_to_label,
#                                  remove_samples=REMOVE_PROTEIN)

import pandas as pd

# COG letters for each sector
COG_R = ['J']
COG_C = ['C', 'E', 'F', 'G', 'H', 'P']

# Samples to exclude (R-limitation conditions)
REMOVE_MRNA    = ['r0','r1','r2','r3','r4','r5',
                  'r0_1','r1_1','r2_1','r3_1','r4_1']
REMOVE_PROTEIN = ['A2','H1','H5','E1','E2','E3','E4']


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def label_genes(df_ann):
    """
    Assign R / C / O labels to genes using COG letter categories.
    Follows 20260113_annotateHwa_2.ipynb.

    Parameters
    ----------
    df_ann : pd.DataFrame
        Schmidt2016_proteomics.csv with columns 'gene_name' and 'cog_letter'.

    Returns
    -------
    gene_to_label : dict
        {gene_name: 'R' | 'C' | 'O'}
    """
    df = df_ann.copy()
    df['cat'] = 'O'
    df.loc[df['cog_letter'].isin(COG_R), 'cat'] = 'R'
    df.loc[df['cog_letter'].isin(COG_C), 'cat'] = 'C'
    return dict(zip(df['gene_name'], df['cat']))


def compute_fractions(df_data, df_growth, gene_to_label, remove_samples=None):
    """
    Compute per-condition R / C / O fractions and attach growth rates.

    Parameters
    ----------
    df_data : pd.DataFrame
        Raw counts / intensities table. Must have a 'gene' column;
        all other non-metadata columns are treated as conditions.
        Metadata columns that are dropped automatically:
        'locus', 'gene length (nt)', 'GO_Label'.
    df_growth : pd.DataFrame
        Growth rate table with columns 'Sample ID' and 'Growth rate (1/h)'.
    gene_to_label : dict
        Output of label_genes().
    remove_samples : list of str, optional
        Sample IDs to exclude (e.g. R-limitation conditions).

    Returns
    -------
    fractions : pd.DataFrame
        Rows = conditions, columns = ['R', 'C', 'O', 'Growth rate'].
        NaN growth rate rows are dropped.
    """
    df = df_data.copy()

    # Annotate genes
    df['GO_Label'] = df['gene'].map(gene_to_label).fillna('O')

    # Identify condition columns
    meta_cols = {'gene', 'locus', 'gene length (nt)', 'GO_Label'}
    condition_cols = [c for c in df.columns if c not in meta_cols]

    # Compute fractions per condition
    fracs = pd.DataFrame(index=['R', 'C', 'O'], columns=condition_cols, dtype=float)
    for col in condition_cols:
        total = df[col].sum()
        if total == 0:
            fracs[col] = 0.0
        else:
            for label in ['R', 'C', 'O']:
                fracs.at[label, col] = df.loc[df['GO_Label'] == label, col].sum() / total

    # Attach growth rates
    growth_map = dict(zip(df_growth['Sample ID'], df_growth['Growth rate (1/h)']))
    fracs.loc['Growth rate'] = [growth_map.get(col, float('nan'))
                                for col in fracs.columns]

    # Transpose: rows = conditions
    fracs = fracs.T.reset_index().rename(columns={'index': 'sample'})
    fracs = fracs.dropna(subset=['Growth rate'])

    # Remove unwanted samples
    if remove_samples:
        fracs = fracs[~fracs['sample'].isin(remove_samples)].reset_index(drop=True)

    return fracs
