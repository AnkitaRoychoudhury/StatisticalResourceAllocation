# Bayesian inference of Model 1 for S. pombe
# Fits protein and mRNA fraction data from Cistvan lab using PyMC.
# Follows 20260527_model1_bayesian-Copy1_mitophagy.ipynb.
#
# Key differences from E. coli:
#   - d_mrna_r is FIXED to 1 (not a free parameter)
#   - alpha_q0 has a tighter prior: Normal(0.9, 0.02)
#   - C_P > 0 feasibility constraint via pm.Potential
#   - Only R protein and all 3 mRNA fractions enter the likelihood
#     (Q protein is NOT observed)
#   - "C" sector is mitophagy (M), not catabolic
#
# Usage in notebook:
#   from bayesian_model1_spombe import fit_model_spombe, predict_from_trace_spombe,
#                                      plot_protein_fractions_spombe, plot_mrna_fractions_spombe
#
#   trace  = fit_model_spombe(protein_growth, R_P_frac, Q_P_frac, C_P_frac,
#                             mrna_growth, mrna_r_frac, mrna_m_frac, mrna_q_frac)
#   preds  = predict_from_trace_spombe(trace, protein_growth, mrna_growth)
#   p_prot = plot_protein_fractions_spombe(protein_growth, R_P_frac, M_P_frac, Q_P_frac, preds)
#   p_mrna = plot_mrna_fractions_spombe(mrna_growth, mrna_r_frac, mrna_m_frac, mrna_q_frac, preds)

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import bokeh.io
import bokeh.plotting

# Sector colors: ribosomal (purple), mitophagy (green), other (teal)
COLORS = dict(r="#9C6EAF", m="#7CA28E", q="#56899F")


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_model_spombe(protein_growth, R_P_frac, Q_P_frac, C_P_frac,
              mrna_growth, mrna_r_frac, mrna_m_frac, mrna_q_frac,
              draws=6000, tune=8000, target_accept=0.95, chains=4):
    """
    Fit Model 1 to S. pombe protein and mRNA fraction data using PyMC.

    Parameters
    ----------
    protein_growth : array-like
        Growth rates for protein fraction measurements.
    R_P_frac : array-like
        Observed ribosomal protein fractions (only R enters the likelihood).
    Q_P_frac, C_P_frac : array-like
        Accepted for API consistency with the E. coli version but not used
        in the S. pombe likelihood.
    mrna_growth : array-like
        Growth rates for mRNA fraction measurements.
    mrna_r_frac, mrna_m_frac, mrna_q_frac : array-like
        Observed ribosomal, mitophagy, and other mRNA fractions.
    draws, tune, target_accept, chains : int / float
        PyMC sampling parameters.

    Returns
    -------
    trace : arviz.InferenceData
    """
    protein_growth = np.asarray(protein_growth).ravel()
    mrna_growth    = np.asarray(mrna_growth).ravel()

    with pm.Model() as model:
        # --- Priors ---
        gamma_1  = pm.Normal('gamma_1', mu=3.5, sigma=0.4)
        alpha_q0 = pm.Normal('alpha_q0', mu=0.9, sigma=0.02)  # tight prior for S. pombe
        sigma_d  = pm.HalfNormal('sigma_d', 0.3)

        # d_mrna_r is FIXED to 1 for S. pombe
        d_mrna_r = 1
        d_mrna_c = pm.LogNormal('d_mrna_c', mu=0, sigma=sigma_d)
        d_mrna_q = pm.LogNormal('d_mrna_q', mu=0, sigma=sigma_d)

        sigma_protein = pm.HalfNormal('sigma_protein', 0.15)
        sigma_mrna    = pm.HalfNormal('sigma_mrna', 0.1)

        k = pm.LogNormal('dmrna/gamma_2', mu=np.log(0.2 / 0.05), sigma=0.5)

        # --- Protein model ---
        R_P = protein_growth / gamma_1
        Q_P = alpha_q0
        C_P = 1 - R_P - Q_P
        # Enforce C_P > 0
        pm.Potential('C_P_positive',
                     pm.math.switch(pt.all(pm.math.gt(C_P, 0)), 0.0, -np.inf))

        # --- mRNA model ---
        S_m     = mrna_growth * k
        R_P_m   = mrna_growth / gamma_1
        alpha_r = R_P_m * S_m
        alpha_q = alpha_q0 * S_m
        alpha_c = S_m - alpha_r - alpha_q

        mR = alpha_r / d_mrna_r
        mC = alpha_c / d_mrna_c
        mQ = alpha_q / d_mrna_q
        m_tot = mR + mC + mQ

        mR_frac = mR / m_tot
        mC_frac = mC / m_tot
        mQ_frac = mQ / m_tot

        # --- Likelihoods (R protein only; all 3 mRNA) ---
        pm.Normal('R_obs',  mu=R_P,    sigma=sigma_protein, observed=R_P_frac)
        pm.Normal('mR_obs', mu=mR_frac, sigma=sigma_mrna,   observed=mrna_r_frac)
        pm.Normal('mC_obs', mu=mC_frac, sigma=sigma_mrna,   observed=mrna_m_frac)
        pm.Normal('mQ_obs', mu=mQ_frac, sigma=sigma_mrna,   observed=mrna_q_frac)

        trace = pm.sample(draws=draws, tune=tune,
                          target_accept=target_accept, chains=chains)

    return trace


# ---------------------------------------------------------------------------
# Posterior predictions
# ---------------------------------------------------------------------------

def predict_from_trace_spombe(trace, protein_growth, mrna_growth):
    """
    Draw posterior predictive trajectories for protein and mRNA fractions.

    Parameters
    ----------
    trace : arviz.InferenceData
    protein_growth, mrna_growth : array-like

    Returns
    -------
    preds : dict
        Sorted growth arrays and median / 2.5% / 97.5% CIs for
        R, M (mitophagy), Q protein and mRNA fractions.
    """
    protein_growths = np.asarray(protein_growth).ravel()
    mrna_growths    = np.asarray(mrna_growth).ravel()

    stk = lambda v: trace.posterior[v].stack(sample=("chain", "draw")).values
    gamma_1_s       = stk('gamma_1')
    k_s             = stk('dmrna/gamma_2')
    alpha_q0_s      = stk('alpha_q0')
    d_mrna_c_s      = stk('d_mrna_c')
    d_mrna_q_s      = stk('d_mrna_q')

    n = len(gamma_1_s)
    R_P_pred  = np.zeros((n, len(protein_growths)))
    C_P_pred  = np.zeros_like(R_P_pred)
    Q_P_pred  = np.zeros_like(R_P_pred)
    mR_pred   = np.zeros((n, len(mrna_growths)))
    mC_pred   = np.zeros_like(mR_pred)
    mQ_pred   = np.zeros_like(mR_pred)

    for i in range(n):
        g1  = gamma_1_s[i]
        k   = k_s[i]
        aq0 = alpha_q0_s[i]
        dc  = d_mrna_c_s[i]
        dq  = d_mrna_q_s[i]

        R_P_pred[i] = protein_growths / g1
        Q_P_pred[i] = aq0
        C_P_pred[i] = 1 - R_P_pred[i] - aq0

        S_m     = mrna_growths * k
        alpha_r = (mrna_growths / g1) * S_m
        alpha_q = aq0 * S_m
        alpha_c = S_m - alpha_r - alpha_q
        mR = alpha_r / 1      # d_mrna_r = 1
        mC = alpha_c / dc
        mQ = alpha_q / dq
        m_tot = mR + mC + mQ
        mR_pred[i] = mR / m_tot
        mC_pred[i] = mC / m_tot
        mQ_pred[i] = mQ / m_tot

    def ci(arr):
        return np.mean(arr, 0), np.percentile(arr, 2.5, 0), np.percentile(arr, 97.5, 0)

    idx_p = np.argsort(protein_growths)
    idx_m = np.argsort(mrna_growths)

    R_med, R_low, R_high = ci(R_P_pred)
    C_med, C_low, C_high = ci(C_P_pred)
    Q_med, Q_low, Q_high = ci(Q_P_pred)
    mR_med, mR_low, mR_high = ci(mR_pred)
    mC_med, mC_low, mC_high = ci(mC_pred)
    mQ_med, mQ_low, mQ_high = ci(mQ_pred)

    return dict(
        protein_growths = protein_growths[idx_p],
        mrna_growths    = mrna_growths[idx_m],
        R_med=R_med[idx_p], R_low=R_low[idx_p], R_high=R_high[idx_p],
        C_med=C_med[idx_p], C_low=C_low[idx_p], C_high=C_high[idx_p],
        Q_med=Q_med[idx_p], Q_low=Q_low[idx_p], Q_high=Q_high[idx_p],
        mR_med=mR_med[idx_m], mR_low=mR_low[idx_m], mR_high=mR_high[idx_m],
        mC_med=mC_med[idx_m], mC_low=mC_low[idx_m], mC_high=mC_high[idx_m],
        mQ_med=mQ_med[idx_m], mQ_low=mQ_low[idx_m], mQ_high=mQ_high[idx_m],
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _ci_patch(p, x, low, high, color):
    x_patch = np.concatenate([x, x[::-1]])
    y_patch = np.concatenate([high, low[::-1]])
    p.patch(x_patch, y_patch, fill_color=color, fill_alpha=0.3, line_alpha=0)


def _best_fit_line(p, x, y, color, legend_label=None):
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(np.min(x), np.max(x), 100)
    p.line(x_fit, slope * x_fit + intercept,
           line_width=2, color=color, line_dash='dashed',
           legend_label=legend_label)


def plot_protein_fractions_spombe(protein_growth, R_P_frac, M_P_frac, Q_P_frac, preds):
    """
    Plot S. pombe protein fraction data with Bayesian CIs and best-fit lines.

    Parameters
    ----------
    protein_growth : array-like
    R_P_frac, M_P_frac, Q_P_frac : array-like
        Observed ribosomal, mitophagy, and other protein fractions.
    preds : dict  — output of predict_from_trace()

    Returns
    -------
    p : bokeh figure
    """
    pg = preds['protein_growths']
    p = bokeh.plotting.figure(
        width=700, height=400,
        x_axis_label='growth rate (/hr)',
        y_axis_label='fraction of proteome',
        title='Protein fractions'
    )

    for med, low, high, obs, label, col in [
        (preds['R_med'], preds['R_low'], preds['R_high'],
         R_P_frac, 'ribosomal proteins', COLORS['r']),
        (preds['C_med'], preds['C_low'], preds['C_high'],
         M_P_frac, 'mitophagy proteins', COLORS['m']),
        (preds['Q_med'], preds['Q_low'], preds['Q_high'],
         Q_P_frac, 'other proteins',     COLORS['q']),
    ]:
        _ci_patch(p, pg, low, high, col)
        p.line(pg, med, line_width=2, color=col, legend_label=label)
        p.scatter(protein_growth, obs, size=6, color=col, alpha=0.8)
        _best_fit_line(p, protein_growth, obs, col,
                       legend_label=label + ' best fit')

    p.add_layout(p.legend[0], 'right')
    p.legend.label_text_font_size = '11pt'
    p.xaxis.major_label_text_font_size = '12pt'
    p.yaxis.major_label_text_font_size = '12pt'

    bokeh.io.show(p)
    return p


def plot_mrna_fractions_spombe(mrna_growth, mrna_r_frac, mrna_m_frac, mrna_q_frac, preds):
    """
    Plot S. pombe mRNA fraction data with Bayesian CIs and best-fit lines.

    Parameters
    ----------
    mrna_growth : array-like
    mrna_r_frac, mrna_m_frac, mrna_q_frac : array-like
        Observed ribosomal, mitophagy, and other mRNA fractions.
    preds : dict  — output of predict_from_trace()

    Returns
    -------
    p : bokeh figure
    """
    mg = preds['mrna_growths']
    p = bokeh.plotting.figure(
        width=700, height=400,
        x_axis_label='growth rate (/hr)',
        y_axis_label='fraction of mRNA counts',
        title='mRNA fractions'
    )

    for med, low, high, obs, label, col in [
        (preds['mR_med'], preds['mR_low'], preds['mR_high'],
         mrna_r_frac, 'ribosomal mRNA', COLORS['r']),
        (preds['mC_med'], preds['mC_low'], preds['mC_high'],
         mrna_m_frac, 'mitophagy mRNA', COLORS['m']),
        (preds['mQ_med'], preds['mQ_low'], preds['mQ_high'],
         mrna_q_frac, 'other mRNA',     COLORS['q']),
    ]:
        _ci_patch(p, mg, low, high, col)
        p.line(mg, med, line_width=2, color=col, legend_label=label)
        p.scatter(mrna_growth, obs, size=6, color=col, alpha=0.8)
        _best_fit_line(p, mrna_growth, obs, col,
                       legend_label=label + ' best fit')

    p.add_layout(p.legend[0], 'right')
    p.legend.label_text_font_size = '11pt'
    p.xaxis.major_label_text_font_size = '12pt'
    p.yaxis.major_label_text_font_size = '12pt'

    bokeh.io.show(p)
    return p
