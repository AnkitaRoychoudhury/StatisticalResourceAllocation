# Resource allocation model (Model 1) - Bayesian posterior parameters
# Computes mRNA and protein fractions across growth rates,
# and plots a log2 fold-change heatmap vs the highest growth condition.

import numpy as np
import holoviews as hv
# Note: call hv.extension("bokeh") in your notebook, not here

# --- Best-fit parameters from Bayesian inference ---
PARAMS = dict(
    d_mrna   = 0.225,
    d_mrna_r = 0.098,
    d_mrna_c = 0.243,
    d_mrna_q = 0.569,
    gamma_1  = 3.42,
    gamma_2  = 0.005,
    alpha_q0 = 0.858,
    Km       = 0.0001,
)

CUSTOM_CMAP = ["#9C6EAF", "#B796C5", "#D2BDDB", "#E8E9ED", "#CFDDD6", "#A0BBAC", "#7CA28E"]


def run_model(growths=None, params=None):
    """
    Run Model 1 across a range of growth rates.

    Parameters
    ----------
    growths : array-like, optional
        Growth rate values to evaluate (default: 200 points from 0.05 to 0.30).
    params : dict, optional
        Model parameters. Defaults to PARAMS (Bayesian posterior values).

    Returns
    -------
    results : dict
        Arrays for each model output, keyed by name:
        'growths', 'S', 'R_P', 'C_P', 'Q_P',
        'mrna_r', 'mrna_c', 'mrna_q', 'mrna_tot',
        'mrna_frac_r', 'mrna_frac_c', 'mrna_frac_q',
        'alpha_r', 'alpha_c', 'tl_rate', 'growth_check'
    """
    if growths is None:
        growths = np.linspace(0.05, 0.30, 200)
    if params is None:
        params = PARAMS

    d_mrna   = params['d_mrna']
    d_mrna_r = params['d_mrna_r']
    d_mrna_c = params['d_mrna_c']
    d_mrna_q = params['d_mrna_q']
    gamma_1  = params['gamma_1']
    gamma_2  = params['gamma_2']
    alpha_q0 = params['alpha_q0']
    Km       = params['Km']

    S_list, R_P_list, C_P_list, Q_P_list = [], [], [], []
    mrna_r_list, mrna_c_list, mrna_q_list, mrna_tot_list = [], [], [], []
    alpha_r_list, alpha_c_list, tl_rate_list, growth_check = [], [], [], []

    for lam in growths:
        S       = lam * d_mrna / gamma_2
        R_P     = lam / gamma_1
        alpha_r = R_P * S
        alpha_q = alpha_q0 * S
        alpha_c = S - alpha_r - alpha_q

        mrna_r   = alpha_r / d_mrna_r
        mrna_c   = alpha_c / d_mrna_c
        mrna_q   = alpha_q / d_mrna_q
        mrna_tot = mrna_r + mrna_c + mrna_q

        C_P     = alpha_c / S
        Q_P     = alpha_q / S
        gamma_r = lam / (mrna_r / (mrna_tot + Km))
        tl_rate = gamma_r * (mrna_tot / (mrna_tot + Km))

        S_list.append(S);         R_P_list.append(R_P)
        C_P_list.append(C_P);     Q_P_list.append(Q_P)
        mrna_r_list.append(mrna_r);   mrna_c_list.append(mrna_c)
        mrna_q_list.append(mrna_q);   mrna_tot_list.append(mrna_tot)
        alpha_r_list.append(alpha_r); alpha_c_list.append(alpha_c)
        tl_rate_list.append(tl_rate)
        growth_check.append(gamma_r * mrna_r / (mrna_tot + Km))

    mrna_r   = np.array(mrna_r_list)
    mrna_c   = np.array(mrna_c_list)
    mrna_q   = np.array(mrna_q_list)
    mrna_tot = np.array(mrna_tot_list)

    return dict(
        growths      = growths,
        S            = np.array(S_list),
        R_P          = np.array(R_P_list),
        C_P          = np.array(C_P_list),
        Q_P          = np.array(Q_P_list),
        mrna_r       = mrna_r,
        mrna_c       = mrna_c,
        mrna_q       = mrna_q,
        mrna_tot     = mrna_tot,
        mrna_frac_r  = mrna_r / mrna_tot,
        mrna_frac_c  = mrna_c / mrna_tot,
        mrna_frac_q  = mrna_q / mrna_tot,
        alpha_r      = np.array(alpha_r_list),
        alpha_c      = np.array(alpha_c_list),
        tl_rate      = np.array(tl_rate_list),
        growth_check = np.array(growth_check),
    )


def plot_log2fc_heatmap(results):
    """
    Plot a log2 fold-change heatmap of mRNA fractions vs the highest
    growth rate condition.

    Parameters
    ----------
    results : dict
        Output of run_model().

    Returns
    -------
    heatmap : hv.HeatMap
    """
    growths     = results['growths']
    mrna_frac_r = results['mrna_frac_r']
    mrna_frac_c = results['mrna_frac_c']
    mrna_frac_q = results['mrna_frac_q']

    # Fold change relative to highest growth rate (last value)
    log2_fc_r = np.log2(mrna_frac_r / mrna_frac_r[-1])
    log2_fc_c = np.log2(mrna_frac_c / mrna_frac_c[-1])
    log2_fc_q = np.log2(mrna_frac_q / mrna_frac_q[-1])

    log2_fc = np.vstack([log2_fc_q, log2_fc_c, log2_fc_r])
    species  = ["other mRNA", "G⁻ mRNA", "ribosomal mRNA"]

    heatmap = hv.HeatMap(
        (growths, species, log2_fc),
        kdims=["Growth rate", "mRNA species"],
        vdims="log2FC"
    )

    heatmap = heatmap.opts(
        width=550,
        height=360,
        cmap=CUSTOM_CMAP,
        clim=(-2, 2),
        colorbar=True,
        colorbar_opts={'title': 'Log2 Fold Change'},
        tools=["hover"],
        ylabel="",
        xlabel="Growth rate",
        fontsize={'labels': 12, 'yticks': 12}
    )

    return heatmap
