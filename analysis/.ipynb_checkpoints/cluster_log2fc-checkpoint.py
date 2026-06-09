# code for generating clusters for the fission yeast data
# pipeline inspired from 20230710_1_fissionYeast_BGanalysis_cluster.ipynb


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import math
import re
from tqdm import tqdm


from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import bokeh.io
import bokeh.plotting
import bokeh.palettes


import holoviews as hv
from holoviews import opts

bokeh.io.output_notebook()
hv.extension('bokeh')

import sys
sys.setrecursionlimit(10000)


def load_data():
    '''default data path is the istvan data sequencing outputs from kallisto, 
    normalized with deseq'''
    
    # Load data
    data_path = '../../data/istvan_data/sequencing/'#kallisto_outputs_copy/'

    df = pd.read_csv(data_path + 'normalized_counts.csv', index_col = 0)

    return df

def load_data2():
    '''default data path is the istvan data sequencing outputs from their paper'''
    
    # Load data
    data_path = '../../data/istvan_data/sequencing/'#kallisto_outputs_copy/'

    df = pd.read_excel(data_path + 'normalized_counts_fromPaper.xlsx')

    return df


def get_fracs(df, mean = True):
    '''Get repressed and induced fractions of each gene in each sample, 
    compared to the average expression of the N samples when mean = False.
    If induced, frac = counts sample i / counts sample N
    If repressed, frac = -counts sample N / counts sample i,
    
    Inputs:
    df: dataframe of the untidy counts matrix, genes vs counts (with exp as a column)
    mean: True if you want to take the average of the three replicates
    
    Outputs:
    df_frac: a melted dataframe with variable as the gene, value as the fraction, and sample name.'''
    
    if mean == True:
        
        unique_samples = df['exp'].unique()
    
        df_melt = pd.melt(df, id_vars = 'exp')
        unique_genes = df_melt['variable'].unique()


        df_frac = pd.DataFrame()
    
        for sample in tqdm(unique_samples):

            # get only sample df
            df_sample = df.loc[df['exp'].isin([sample, 'N'])]

            # melt df
            df_sample_melt = pd.melt(df_sample, id_vars = 'exp')

            # get means
            grouped_ser = df_sample_melt.groupby(['variable','exp'])['value'].mean()

            grouped_df = grouped_ser.reset_index()
            grouped_df = grouped_df.rename(columns = {'value':'mean'})

            # get fraction of means
            df_frac_curr = pd.DataFrame()

            for gene in unique_genes:
                curr_df = grouped_df.loc[grouped_df['variable']==gene]


                num_sample = curr_df.loc[curr_df['exp'] == sample]['mean'].values
                num_ctrl = curr_df.loc[curr_df['exp'] == 'N']['mean'].values

                if num_sample > num_ctrl:
                    frac = num_sample/num_ctrl

                else:
                    frac = -num_ctrl/num_sample
            
                # if num_sample == 0 or num_ctrl == 0:
                #     frac = np.nan
                # else:
                #     frac = np.log2(num_sample / num_ctrl)

                # df_frac_curr[gene] = frac

            df_frac_melt = pd.melt(df_frac_curr)
            df_frac_melt['sample'] = [sample] * 6734
            #print(df_frac_melt)

            df_frac = df_frac.append(df_frac_melt, ignore_index = True)
            
    else:
        
        samples = ['Gly1','Gly2','Gly3',
           'N1','N2','N3',
           'Phe1','Phe2','Phe3',
           'Pro1','Pro2','Pro3',
           'Ser1','Ser2', 'Ser3',
           'Trp1','Trp2','Trp3',
           'Glu1','Glu2','Glu3',
           'Ile1','Ile2','Ile3']
        
        df['exp nums'] = samples   
    
        df_melt = pd.melt(df, id_vars = 'exp')
    
        unique_genes = df_melt['variable'].unique()


         # get only nitrogen means
        df_n = df.loc[df['exp'].isin(['N'])]

        df_n_melt = pd.melt(df_n, id_vars = 'exp')
        df_n_melt = df_n_melt[df_n_melt['variable'] != 'exp nums']

        grouped_n = df_n_melt.groupby(['variable'])['value'].mean()
        df_n_mean = grouped_n.reset_index()
        df_n_mean = df_n_mean.rename(columns = {"value":'mean'})


        df_frac = pd.DataFrame()

        for sample in tqdm(samples):

            # now get fractions 
            df_sample = df.loc[df['exp nums'].isin([sample])]
            #df_sample = df_sample.drop('exp',axis=1)
            df_sample_melt = pd.melt(df_sample, id_vars = 'exp nums')

            df_frac_curr = pd.DataFrame()
            #fracs = []
            unique_genes = [item for item in unique_genes if item != 'exp nums']

            rows = []
            for gene in unique_genes:

                curr_df = df_sample_melt.loc[df_sample_melt['variable']==gene]
                num_sample = curr_df['value'].values
                num_ctrl = df_n_mean.loc[df_n_mean['variable']==gene]['mean'].values

                num_sample = float(num_sample[0])
                num_ctrl = float(num_ctrl[0])
                if num_sample == 0 or num_ctrl == 0:
                    frac = np.nan
                else:
                    frac = np.log2(num_sample / num_ctrl)
                    #print(frac)

                rows.append({
                "gene": gene,
                "log2_frac": frac,
                "sample": sample
            })

                #try:
                #df_frac_curr[gene]=frac
               # df_frac_curr.loc[0, gene]=frac
               # except:
               #     print(frac, num_ctrl, num_sample, gene)

            #after doing all genes
            #df_frac_melt = pd.melt(df_frac_curr)
            #df_frac_melt['sample'] = sample 
            df_frac_melt = pd.DataFrame(rows)

            #df_frac = df_frac.append(df_frac_melt, ignore_index = True)
            df_frac = pd.concat([df_frac, df_frac_melt], ignore_index = True)

        
    return df_frac


def get_clusters(df_frac, threshold = 2, ret_vv = False):
    '''This function performs hierarchical clustering on fraction data.
    
    Inputs:
    df_frac: a melted dataframe with variable as the gene, value as the fraction, and sample name
    threshold: default = 2, the distance at which to cut off the dendrogram to extract the clusters
    
    Outputs:
    clusters: A list of the clusters, in order of the genes
    reorder: The reordered dataframe based on the clusters for plotting. Samples vs genes
    reorder_melt: The tidy version of the reorder dataframe'''
    
    # unmelt df
    df_frac_untidy = df_frac.pivot(index = 'variable', columns = 'sample', values = 'value')
    df_frac_u2 = df_frac_untidy.reset_index()
    
    variable_values = df_frac_u2['variable'].values
    df_frac_u2 = df_frac_u2.drop('variable',axis=1)
    df_frac_u2 = df_frac_u2.replace([np.nan, np.inf, -np.inf],0)
    
    rna_array = df_frac_u2.values
    
    distance_matrix = linkage(rna_array, method='ward', metric='euclidean', optimal_ordering = False)
    
    
    clusters = fcluster(distance_matrix, threshold, criterion = 'distance')
    #clusters = fcluster(distance_matrix, t = 100, criterion = 'maxclust')
   
    # define reindexing
    reindex = []

    # iterate through all possible clusters
    for i in tqdm(range(max(clusters)+1)): #1277

        # find all genes that are in the cluster
        for j,clus in enumerate(clusters): #6347

            # if the cluster equals 0
            if clus == i:
                # append the location of the thing in reindex
                reindex.append(j)

    
    
    # reorder df for plotting
    df_frac_u2['variable'] = variable_values
    
    
    
    reorder = df_frac_u2.reindex(reindex)
    reorder = reorder.reset_index()
    reorder = reorder.drop('index', axis=1)
    #reorder = reorder.set_index('variable', drop = True)
    reorder_melt = pd.melt(reorder, id_vars = 'variable')

    if ret_vv == False:
        ret = clusters, reorder, reorder_melt
        
    else:
        ret = clusters, reorder, reorder_melt, variable_values
        
    return ret






