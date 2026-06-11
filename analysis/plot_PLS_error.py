import pandas as pd
import bokeh.plotting


def plot_errors_n(seed, label, df_null_merged, df_bs_merged):
    short_label = label[:2]
    
    # melt the dataframes
    df_null_merged_melt = pd.melt(df_null_merged, id_vars = ['seed', 'identity', 'num comps'])
    df_bs_merged_melt = pd.melt(df_bs_merged, id_vars = ['seed', 'identity', 'num comps'])

    #get summary statistics
    df_null_summary = df_null_merged_melt.groupby(['seed', 'num comps', 'variable'])['value'].agg(median = 'median', q1 = lambda x: x.quantile(0.25),
                                                                           q3 = lambda x: x.quantile(0.75)).reset_index()
    df_bs_summary = df_bs_merged_melt.groupby(['seed', 'num comps', 'variable'])['value'].agg(median = 'median', q1 = lambda x: x.quantile(0.25),
                                                                           q3 = lambda x: x.quantile(0.75)).reset_index()


    # get correct seed and label, order correctly so we can use lines
    df_curr_null = df_null_summary.loc[ (df_null_summary['seed']==seed) & (df_null_summary['variable'].str[:2] == short_label)].copy()
    df_curr_null['num comps'] = pd.to_numeric(df_curr_null['num comps'], errors='coerce') 
    df_curr_null = df_curr_null.sort_values(by = 'num comps', ascending = True)

    df_curr_bs = df_bs_summary.loc[ (df_bs_summary['seed']==seed) & (df_bs_summary['variable'].str[:2] == short_label)].copy()
    df_curr_bs['num comps'] = pd.to_numeric(df_curr_bs['num comps'], errors='coerce') 
    df_curr_bs = df_curr_bs.sort_values(by = 'num comps', ascending = True)


    # separate train and test for null
    df_null_train = df_curr_null.loc[df_curr_null['variable'].str.split().str[-1] == 'train']
    df_null_test = df_curr_null.loc[df_curr_null['variable'].str.split().str[-1] == 'test']

    # separate train and test for bs
    df_bs_train = df_curr_bs.loc[df_curr_bs['variable'].str.split().str[-1] == 'train']
    df_bs_test = df_curr_bs.loc[df_curr_bs['variable'].str.split().str[-1] == 'test']

    # plot
    p1 = bokeh.plotting.figure(width = 700, height = 350,
                              x_axis_label = 'number of components',
                              y_axis_label = label, 
                              title = label + ', seed = ' + str(seed))

    # plot null
    p1.line(df_null_train['num comps'], df_null_train['median'], legend_label = 'Null Shuffled Training Error', color = '#B96492', line_width = 5)
    p1.segment(df_null_train['num comps'], df_null_train['q3'], 
               df_null_train['num comps'], df_null_train['q1'], 
               line_width=2, color = '#B96492')  # IQR whiskers

    p1.line(df_null_test['num comps'], df_null_test['median'], legend_label = 'Null Shuffled Testing Error', color = '#843B62', line_width = 5)
    p1.segment(df_null_test['num comps'], df_null_test['q3'], 
               df_null_test['num comps'], df_null_test['q1'], 
               line_width=2, color = '#843B62')  # IQR whiskers

    # plot bs
    p1.line(df_bs_train['num comps'], df_bs_train['median'], legend_label = 'Bootstrapped Training Error',color = '#4B91AA', line_width = 5)
    p1.segment(df_bs_train['num comps'], df_bs_train['q3'], 
               df_bs_train['num comps'], df_bs_train['q1'], 
               line_width=2 ,color = '#4B91AA')  # IQR whiskers

    p1.line(df_bs_test['num comps'], df_bs_test['median'], legend_label = 'Bootstrapped Testing Error', color = '#326273', line_width = 5)
    p1.segment(df_bs_test['num comps'], df_bs_test['q3'], 
               df_bs_test['num comps'], df_bs_test['q1'], 
               line_width=2, color = '#326273')  # IQR whiskers

    p1.add_layout(p1.legend[0], 'right')

    p1.title.text_font_size = '15pt'
    p1.yaxis.major_label_text_font_size = '13pt'
    p1.xaxis.major_label_text_font_size = '13pt'
    p1.yaxis.axis_label_text_font_size = '13pt'
    p1.xaxis.axis_label_text_font_size = '13pt'

    return p1
