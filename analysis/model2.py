# Model 2: Nutrient-dependent resource allocation
# Solves steady-state equations for protein/mRNA fractions and elongation rate
# across growth rates, for two regimes:
#   - Co-limitation (E. coli): alpha=95, beta=2, eta=1, kappa=10, delta=130
#   - Ribosomal mRNA limitation (S. cerevisiae): alpha=2, beta=3000, eta=0.3, kappa=0.01, delta=130
#
# Follows 20260226_model2_ND_realistic_Figs.ipynb
#
# State vector: [mc, mr, mpol2, R, C, Pol2, Q, t]
#   mc    = metabolic mRNA
#   mr    = ribosomal mRNA
#   mpol2 = RNA Pol2 mRNA
#   R     = ribosomal protein fraction
#   C     = metabolic protein fraction
#   Pol2  = RNA Pol2 protein
#   Q     = other protein
#   t     = charged tRNA proxy
#
# Usage in notebook:
#   from model2 import (DEFAULT_ECOLI, DEFAULT_YEAST, DEFAULT_SHARED,
#                        solve_steady_state,
#                        plot_total_mrna, plot_ribosome_fraction, plot_elongation_rate)
#
#   ecoli_data = solve_steady_state(DEFAULT_ECOLI, DEFAULT_SHARED,
#                                   np.linspace(0.002, 0.9, 70))
#   yeast_data = solve_steady_state(DEFAULT_YEAST, DEFAULT_SHARED,
#                                   np.linspace(0.02, 0.7, 40), growth_min=0.1)
#
#   p1, p2 = plot_total_mrna(ecoli_data, yeast_data)
#   bokeh.io.show(p1); bokeh.io.show(p2)
#   p3, p4 = plot_ribosome_fraction(ecoli_data, yeast_data)
#   bokeh.io.show(p3); bokeh.io.show(p4)
#   p5, p6 = plot_elongation_rate(ecoli_data, yeast_data)
#   bokeh.io.show(p5); bokeh.io.show(p6)

import numpy as np
from scipy.optimize import root
import bokeh.plotting
from bokeh.layouts import gridplot

# ---------------------------------------------------------------------------
# Default parameters (override freely in the notebook)
# ---------------------------------------------------------------------------

DEFAULT_ECOLI = dict(
    alpha = 95,    # max translation rate
    beta  = 2,     # amino-acid production per metabolic protein
    eta   = 1,     # AA cost per ribosome cycle
    kappa = 10,    # metabolic capacity scalar
    delta = 130,   # tRNA decay rate
)

DEFAULT_YEAST = dict(
    alpha = 2,
    beta  = 3000,
    eta   = 0.3,
    kappa = 0.01,
    delta = 130,
)

DEFAULT_SHARED = dict(
    alpha_c   = 0.05,   # transcription allocation: metabolic mRNA
    alpha_r   = 0.9,    # transcription allocation: ribosomal mRNA
    alpha_pol2 = 0.05,  # transcription allocation: Pol2 mRNA
    mq        = 1.0,    # other mRNA (constant, non-transcribed here)
)

# Colors
COLOR_RRNA   = "#9C6EAF"   # purple  — ribosome fraction
COLOR_MRNA   = "#2B2B2B"   # near-black — total mRNA
COLOR_ELONG  = "#BA3B54"   # red     — elongation rate
COLOR_ECOLI  = "#4C72B0"   # blue
COLOR_YEAST  = "#DD8452"   # orange


# ---------------------------------------------------------------------------
# Steady-state equations
# ---------------------------------------------------------------------------

def _rhs(x, params):
    """
    Steady-state residuals for Model 2.

    Parameters
    ----------
    x : array, shape (8,)
        [mc, mr, mpol2, R, C, Pol2, Q, t]
    params : list
        [alpha_c, alpha_r, alpha_pol2, alpha, beta, eta, kappa, delta, mq, lam]
    """
    x = np.clip(x, 1e-12, 1e15)
    mc, mr, mpol2, R, C, Pol2, Q, t = x
    alpha_c, alpha_r, alpha_pol2, alpha, beta, eta, kappa, delta, mq, lam = params

    m_total = mc + mr + mpol2 + mq
    Phi = (t / (t + 1.0)) * (1.0 / (m_total + 1.0))

    return np.array([
        alpha_c * Pol2 - mc,
        alpha_r * Pol2 - mr,
        alpha_pol2 * Pol2 - mpol2,
        alpha * Phi * mr * R - lam * R,
        alpha * Phi * mc * R - lam * C,
        alpha * Phi * mpol2 * R - lam * Pol2,
        alpha * Phi * mq * R - lam * Q,
        beta * kappa * C - delta * t - eta * kappa * Phi * mc * R,
    ])


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_steady_state(species_params, shared_params, growths,
                       growth_min=None, x0=None):
    """
    Sweep growth rates and find the physical steady state at each point.

    Parameters
    ----------
    species_params : dict
        Keys: alpha, beta, eta, kappa, delta.
        Use DEFAULT_ECOLI or DEFAULT_YEAST, or supply your own.
    shared_params : dict
        Keys: alpha_c, alpha_r, alpha_pol2, mq.
        Use DEFAULT_SHARED or supply your own.
    growths : array-like
        Growth rate values (non-dimensional) to evaluate.
    growth_min : float, optional
        Discard solutions below this growth rate (applied after solving).
        Useful for yeast where low-growth solutions are unphysical.
    x0 : array, shape (8,), optional
        Initial guess for the state vector. Defaults to ones.

    Returns
    -------
    data : dict or None
        Keys:
          'growth'      – valid growth rates (1D array)
          'mtot'        – total mRNA concentration
          'mtot_fold'   – fold change relative to lowest valid growth
          'R_frac'      – ribosomal protein fraction (R / total protein)
          'elong_rate'  – elongation rate = alpha * t/(t+1)
          'tl_rate'     – translation rate = alpha * Phi * m_tot
        Returns None if no physical solutions were found.
    """
    alpha   = species_params['alpha']
    beta    = species_params['beta']
    eta     = species_params['eta']
    kappa   = species_params['kappa']
    delta   = species_params['delta']

    alpha_c    = shared_params['alpha_c']
    alpha_r    = shared_params['alpha_r']
    alpha_pol2 = shared_params['alpha_pol2']
    mq         = shared_params['mq']

    if x0 is None:
        x0 = np.ones(8)

    growths = np.asarray(growths)
    current_guess = x0.copy()

    valid_growth, mtot_list, R_frac_list, elong_list, tl_list = [], [], [], [], []

    for lam in growths:
        params = [alpha_c, alpha_r, alpha_pol2,
                  alpha, beta, eta, kappa, delta, mq, lam]
        sol = root(_rhs, current_guess, args=(params,), tol=1e-12, method='hybr')

        if sol.success and np.all(sol.x >= 0):
            current_guess = sol.x
            mc, mr, mpol2, R, C, Pol2, Q, t = sol.x
            P_tot  = R + C + Pol2 + Q
            m_tot  = mc + mr + mpol2 + mq
            Phi    = (t / (t + 1.0)) * (1.0 / (m_tot + 1.0))

            valid_growth.append(lam)
            mtot_list.append(m_tot)
            R_frac_list.append(R / P_tot)
            elong_list.append(alpha * (t / (t + 1.0)))
            tl_list.append(alpha * Phi * m_tot)

    if len(valid_growth) == 0:
        print(f"No physical steady states found "
              f"(alpha={alpha}, beta={beta}, eta={eta}, kappa={kappa}, delta={delta})")
        return None

    valid_growth = np.array(valid_growth)
    mtot_arr     = np.array(mtot_list)
    R_frac_arr   = np.array(R_frac_list)
    elong_arr    = np.array(elong_list)
    tl_arr       = np.array(tl_list)

    # Optional: trim low-growth solutions
    if growth_min is not None:
        mask       = valid_growth >= growth_min
        valid_growth = valid_growth[mask]
        mtot_arr   = mtot_arr[mask]
        R_frac_arr = R_frac_arr[mask]
        elong_arr  = elong_arr[mask]
        tl_arr     = tl_arr[mask]

    return dict(
        growth     = valid_growth,
        mtot       = mtot_arr,
        mtot_fold  = mtot_arr / mtot_arr[0],
        R_frac     = R_frac_arr,
        elong_rate = elong_arr,
        tl_rate    = tl_arr,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _base_fig(title, x_label, y_label, y_range=None):
    opts = dict(
        width=380, height=300,
        output_backend='svg',
        title=title,
        x_axis_label=x_label,
        y_axis_label=y_label,
    )
    if y_range is not None:
        opts['y_range'] = y_range
    p = bokeh.plotting.figure(**opts)
    p.grid.visible = False
    p.xaxis.major_label_text_font_size = "13px"
    p.yaxis.major_label_text_font_size = "13px"
    p.axis.axis_label_text_font_size = "13px"
    return p


def plot_total_mrna(ecoli_data, yeast_data,
                    ecoli_label="E. coli (co-lim)",
                    yeast_label="S. cerevisiae (ribo mRNA lim)"):
    """Total mRNA fold change vs growth rate — E. coli and yeast side by side."""
    ge = ecoli_data['growth']
    gy = yeast_data['growth']
    y_max = max(ecoli_data['mtot_fold'].max(), yeast_data['mtot_fold'].max()) * 1.1

    p1 = _base_fig(f"{ecoli_label} — total mRNA",
                   "growth rate (ND)", "mRNA fold change", y_range=[0, y_max])
    p1.line(ge, ecoli_data['mtot_fold'], line_width=2.5, color=COLOR_MRNA)

    p2 = _base_fig(f"{yeast_label} — total mRNA",
                   "growth rate (ND)", "mRNA fold change", y_range=[0, y_max])
    p2.line(gy, yeast_data['mtot_fold'], line_width=2.5, color=COLOR_MRNA)

    return p1, p2


def plot_ribosome_fraction(ecoli_data, yeast_data,
                           ecoli_label="E. coli (co-lim)",
                           yeast_label="S. cerevisiae (ribo mRNA lim)"):
    """Ribosomal protein fraction vs growth rate — E. coli and yeast side by side."""
    ge = ecoli_data['growth']
    gy = yeast_data['growth']
    y_max = max(ecoli_data['R_frac'].max(), yeast_data['R_frac'].max()) * 1.1

    p3 = _base_fig(f"{ecoli_label} — ribosome fraction",
                   "growth rate (ND)", "ribosomal protein fraction", y_range=[0, y_max])
    p3.line(ge, ecoli_data['R_frac'], line_width=2.5, color=COLOR_RRNA)

    p4 = _base_fig(f"{yeast_label} — ribosome fraction",
                   "growth rate (ND)", "ribosomal protein fraction", y_range=[0, y_max])
    p4.line(gy, yeast_data['R_frac'], line_width=2.5, color=COLOR_RRNA)

    return p3, p4


def plot_elongation_rate(ecoli_data, yeast_data,
                         ecoli_label="E. coli (co-lim)",
                         yeast_label="S. cerevisiae (ribo mRNA lim)"):
    """Elongation rate vs growth rate — E. coli and yeast side by side."""
    ge = ecoli_data['growth']
    gy = yeast_data['growth']
    y_max = max(ecoli_data['elong_rate'].max(), yeast_data['elong_rate'].max()) * 1.1

    p5 = _base_fig(f"{ecoli_label} — elongation rate",
                   "growth rate (ND)", "elongation rate", y_range=[0, y_max])
    p5.line(ge, ecoli_data['elong_rate'], line_width=2.5, color=COLOR_ELONG)

    p6 = _base_fig(f"{yeast_label} — elongation rate",
                   "growth rate (ND)", "elongation rate", y_range=[0, y_max])
    p6.line(gy, yeast_data['elong_rate'], line_width=2.5, color=COLOR_ELONG)

    return p5, p6
