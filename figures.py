"""This module produces the figures for the manuscript. Run
``python figures.py`` to regenerate everything.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.collections import LineCollection
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import networkx as nx

from model import (
    f,
    make_schedule,
    compute_priority_heatmap,
)
from plot_utils import (
    COLORS,
    LINESTYLES,
    figures_dir,
    supp_figures_dir,
    add_panel_label,
    _build_network,
    draw_priority_heatmap,
    _style_time_panel,
    plot_two_issue_CEP,
    run_scenario,
    run_N_baseline,
    _N_BASELINE,
    figureR3_scenarios,
    figureR4_scenarios,
    figureD1_scenarios,
)


def figureI1_survey_firstdiff_scatter(save=True):
    """
    Figure I1 (Introduction section; manuscript Fig. 1): majority concern over
    an issue does not translate into majority priority.
    Panel A: percentage of adults expressing concern about climate change
             (Gallup) and percentage stating climate change or the economy
             should be a top national priority (Pew), among U.S. adults,
             2007-2024 (restricted to overlapping years).
    Panel B: year-over-year changes in climate and economy priority (annual
             first differences of the Pew series), with their Pearson
             correlation and the OLS fit of Delta economy priority on
             Delta climate priority (statistics reported in the SI).
    """
    import pandas as pd

    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
    csv_path = os.path.join(current_dir, "survey_data", "climate_concern_vs_priority_confirmed.csv")
    df = pd.read_csv(csv_path).set_index('year')

    cols = ['gallup_climate_worry_pct', 'pew_climate_priority_pct', 'pew_economy_priority_pct']
    df_overlap = df.dropna(subset=cols)

    d_climate = df_overlap['pew_climate_priority_pct'].diff().dropna()
    d_economy = df_overlap['pew_economy_priority_pct'].diff().dropna()

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Raw time series (overlapping years only)
    ax = axs[0]
    ax.plot(df_overlap.index, df_overlap['pew_economy_priority_pct'],
            '^-', color=COLORS['primary2'], label='economy priority', markersize=8, lw=2)
    ax.plot(df_overlap.index, df_overlap['gallup_climate_worry_pct'],
            'o-', color=COLORS['primary4'], label='climate concern', markersize=8, lw=2)
    ax.plot(df_overlap.index, df_overlap['pew_climate_priority_pct'],
            's-', color=COLORS['primary1'], label='climate priority', markersize=8, lw=2)
    ax.axhline(y=50, color='black', linestyle='-', linewidth=1.8, alpha=0.7, zorder=0)
    ax.set_ylabel('Percentage (%)', fontsize=20)
    ax.set_xlabel('Year', fontsize=20)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.7)

    # Panel B: Scatter of first differences
    ax = axs[1]
    ax.scatter(d_climate.values, d_economy.values, color='#37474f', s=30, zorder=3)
    for yr, dx, dy in zip(d_climate.index, d_climate.values, d_economy.values):
        ax.annotate(str(yr), (dx, dy), textcoords='offset points',
                    xytext=(5, 5), fontsize=12, color='#555555')

    from scipy import stats
    import statsmodels.api as sm

    x_arr = d_climate.values
    y_arr = d_economy.values
    ols_res = sm.OLS(y_arr, sm.add_constant(x_arr)).fit()
    intercept, slope = ols_res.params
    se_intercept, se_slope = ols_res.bse
    t_intercept, t_slope = ols_res.tvalues
    p_intercept, p_slope = ols_res.pvalues

    x_fit = np.linspace(d_climate.min(), d_climate.max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, color='#c62828', lw=1.5, ls='--')

    r, p_pearson = stats.pearsonr(x_arr, y_arr)
    ax.text(0.05, 0.95, f'r = {r:.2f}', transform=ax.transAxes,
            fontsize=15, va='top', color='#c62828')

    print(f"[figureI1 Panel B] n = {int(ols_res.nobs)}  "
          f"Pearson r = {r:+.3f}  p = {p_pearson:.3e}  "
          f"R^2 = {ols_res.rsquared:.3f}  "
          f"F({int(ols_res.df_model)},{int(ols_res.df_resid)}) = {ols_res.fvalue:.2f}, "
          f"p = {ols_res.f_pvalue:.3e}  "
          f"| OLS slope = {slope:+.3f}  (SE = {se_slope:.3f}, "
          f"t({int(ols_res.df_resid)}) = {t_slope:+.2f}, p = {p_slope:.3e})  "
          f"intercept = {intercept:+.3f}  (SE = {se_intercept:.3f}, "
          f"t({int(ols_res.df_resid)}) = {t_intercept:+.2f}, p = {p_intercept:.3f})")

    lim = max(abs(d_climate.min()), abs(d_climate.max()),
              abs(d_economy.min()), abs(d_economy.max())) * 1.3
    ax.axhline(0, color='gray', ls='-', lw=0.5)
    ax.axvline(0, color='gray', ls='-', lw=0.5)
    ax.fill_between([-lim, 0], 0, lim, alpha=0.06, color='#2e7d32')
    ax.fill_between([0, lim], -lim, 0, alpha=0.06, color='#2e7d32')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r'$\Delta$ Climate priority (pp)', fontsize=20)
    ax.set_ylabel(r'$\Delta$ Economy priority (pp)', fontsize=20)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(True, alpha=0.5)

    for i, ax in enumerate(axs):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(-0.15, 1.05, chr(65 + i), transform=ax.transAxes,
                fontsize=25, fontweight='bold', va='top')

    if save:
        fname = os.path.join(figures_dir, "figureI1_survey_firstdiff_scatter.pdf")
        plt.savefig(fname)
        print(f"Saved: {fname}")

    plt.close()
    return fig


def figureM1_network_and_conformity(save=True):
    """
    Figure M1 (Model section; manuscript Fig. 2): model components.
    Panel A (left): two-group network with full within-group connectivity
    (learning), as two stacked sub-panels — no intergroup learning (rho = 0)
    on top, moderate intergroup learning (rho = 0.5) below.
    Panel B (right): conformity response function f(X; alpha) for varying
    levels of normative conformity, spanning the full figure height.
    """
    # --- Panel A (left): two stacked network sub-panels ---
    group1 = [f'G1_{i}' for i in range(1, 6)]
    group2 = [f'G2_{i}' for i in range(1, 6)]
    angle = np.linspace(0, 2 * np.pi, 6)[:-1]
    pos = {}
    for i, node in enumerate(group1):
        pos[node] = (-1.5 + np.sin(angle[i]), np.cos(angle[i]))
    for i, node in enumerate(group2):
        pos[node] = (1.5 + np.sin(angle[i]), np.cos(angle[i]))
    node_colors = ['blue' if n.startswith('G1') else 'red' for n in group1 + group2]
    rho_values = [0, 0.5]
    subtitles = [r'No intergroup learning ($\rho=0$)',
                 r'Moderate intergroup learning ($\rho=0.5$)']

    # Layout: 2x2 mosaic with conformity panel spanning both rows on the right.
    fig = plt.figure(figsize=(14, 6))
    axs = fig.subplot_mosaic(
        [['net_top', 'conformity'],
         ['net_bot', 'conformity']],
    )
    ax_net_top = axs['net_top']
    ax_net_bot = axs['net_bot']
    ax_conf = axs['conformity']

    for rho_val, subtitle, ax in zip(rho_values, subtitles, [ax_net_top, ax_net_bot]):
        rng = np.random.default_rng(42)
        G = _build_network(group1, group2, rho_val, rng)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, ax=ax)
        ax.set_title(subtitle, fontsize=20, style='italic')
        ax.axis('off')

    add_panel_label(ax_net_top, 'A', x=0.05, y=1.05, fontsize=25)

    # --- Panel B (right): conformity response S-curves ---
    x_values = np.linspace(0.001, 0.999, 1000)
    alphas = [1, 2, 4]
    colors = [COLORS['black'], COLORS['primary4'], COLORS['primary2']]
    labels = [
        r'Linear ($\alpha=1$)',
        r'Normative conformity ($\alpha=2$)',
        r'Normative conformity ($\alpha=4$)',
    ]
    for alpha, color, label in zip(alphas, colors, labels):
        y_values = f(x_values, alpha)
        ax_conf.plot(x_values, y_values, color=color, linewidth=2, label=label)

    ax_conf.set_xlabel(r'proportion of individuals concerned ($X$)', fontsize=20)
    ax_conf.set_ylabel(r'conformity response ($f(X;\alpha)$)', fontsize=20)
    ax_conf.legend(fontsize=12, loc='upper left')
    ax_conf.tick_params(axis='both', labelsize=15)
    ax_conf.set_xlim(0, 1)
    ax_conf.set_ylim(0, 1)
    ax_conf.set_aspect('equal', 'box')
    add_panel_label(ax_conf, 'B', x=-0.17, y=1.05, fontsize=25)

    if save:
        fname = os.path.join(figures_dir, "figureM1_network_and_conformity.pdf")
        plt.savefig(fname)
        print(f"Saved: {fname}")

    plt.close()
    return fig


def figureR1_two_issue_dynamics(save=True):
    """
    Figure R1 (manuscript Fig. 3): high social learning produces concern
    polarization and underprioritization. Three-panel time series under the
    baseline parameters (Table S1).
    A: population priority for H and L (P(t) and 1 - P(t))
    B: population-level concern for each issue (C_H(t) and C_L(t))
    C: group-level concern for each issue (one curve per group), showing
       between-group divergence for H
    """
    data = run_N_baseline()
    t = data['t']
    t_final = _N_BASELINE['t_final']

    linewidth = 2.5
    legend_kwargs = dict(
        labelspacing=0.3, handlelength=3.0, handleheight=1.5, handletextpad=0.5,
        borderpad=0.3, borderaxespad=0.3, columnspacing=0.6, frameon=True,
    )

    fig, axs = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
    ax_A, ax_B, ax_C = axs

    # Panel A: priorities (H = blue, L = orange)
    ax_A.plot(t, data['P'], color=COLORS['primary1'], linewidth=linewidth, linestyle='-',
              alpha=0.9, label='issue H')
    ax_A.plot(t, 1 - data['P'], color=COLORS['primary2'], linewidth=linewidth, linestyle='--',
              alpha=0.9, label='issue L')
    ax_A.set_ylabel(r"Priority ($P$)")
    ax_A.legend(loc='center right', **legend_kwargs)

    # Panel B: population-level concern C_H (blue) and C_L (orange)
    ax_B.plot(t, data['C_H'], color=COLORS['primary1'], linewidth=linewidth, linestyle='-',
              alpha=0.85, label='issue H')
    ax_B.plot(t, data['C_L'], color=COLORS['primary2'], linewidth=linewidth, linestyle='--',
              alpha=0.85, label='issue L')
    ax_B.set_ylabel("Population concern\n" + r"for issue $i$ ($\bar{C}_i$)")
    ax_B.legend(loc='lower right', **legend_kwargs)

    # Panel C: group-level trajectories for issue H (green) and issue L (purple),
    # using colors distinct from the priority panels (blue/orange)
    col_H, col_L = COLORS['primary3'], '#6a1b9a'
    ax_C.plot(t, data['C_1_H'], color=col_H, linewidth=linewidth, linestyle='-',
              alpha=0.85, label='issue H group 1')
    ax_C.plot(t, data['C_2_H'], color=col_H, linewidth=linewidth, linestyle=':',
              alpha=0.85, label='issue H group 2')
    ax_C.plot(t, data['C_1_L'], color=col_L, linewidth=linewidth, linestyle='-',
              alpha=0.85, label='issue L group 1')
    ax_C.plot(t, data['C_2_L'], color=col_L, linewidth=linewidth, linestyle=':',
              alpha=0.85, label='issue L group 2')
    ax_C.set_ylabel("Group concern\n" + r"for issue $i$ ($C_{gi}$)")
    ax_C.set_xlabel("Time")
    ax_C.legend(loc='center right', **legend_kwargs)

    for letter, ax in zip('ABC', axs):
        ax.set_xlim(0, t_final)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1.2, alpha=0.5, zorder=0)
        ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        add_panel_label(ax, letter)

    if save:
        fname = os.path.join(figures_dir, 'figureR1_two_issue_dynamics.pdf')
        plt.savefig(fname)
        print(f"Saved: {fname}")

    plt.close()
    return fig


# Markers highlighting the operating points, shared between each row's slice
# panel and its companion heatmap: (s_H, color, shape). Distinct shapes let the
# reader track a point across panels even when two points share a color.
_R2_MARKERS_RHO0 = [(0.9, COLORS['primary1'], 'o'),   # baseline
                    (0.6, COLORS['primary2'], 's')]   # reduced social learning
_R2_MARKERS_RHO02 = [(0.9, COLORS['primary2'], '^')]  # intergroup learning


def _draw_priority_slice(ax, s_H_values, P_slice, markers, fontscale=1.0,
                         ylabel_fontsize=None, xticks=None):
    """Draw a horizontal slice P(s_H) of a phase heatmap taken at fixed severity.

    The line is colored by P with the same coolwarm map as the heatmaps, but
    with boosted saturation so it contrasts against the white background, and is
    broken at the discontinuous priority drop so no spurious vertical segment is
    drawn. `markers` is a list of (s_H, color, shape) matching the dots on the
    companion heatmap. Axis styling matches the time-series panels (A, D).
    `fontscale` multiplies the label/tick font sizes; `ylabel_fontsize` overrides
    the y-axis label size (defaulting to 20*fontscale); `xticks`, if given, sets
    the x-axis tick positions.
    """
    P_slice = np.asarray(P_slice)
    pts = np.array([s_H_values, P_slice]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    seg_P = 0.5 * (P_slice[:-1] + P_slice[1:])
    keep = np.abs(np.diff(P_slice)) <= 0.15          # drop the segment spanning the jump
    colors = rgb_to_hsv(plt.cm.coolwarm(plt.Normalize(0, 1)(seg_P))[:, :3])
    colors[:, 1] = np.clip(colors[:, 1] * 1.8, 0, 1)  # boost saturation
    colors[:, 2] = np.clip(colors[:, 2] * 0.82, 0, 1)  # darken for contrast on white
    lc = LineCollection(segs[keep], colors=hsv_to_rgb(colors)[keep], linewidth=3.5, zorder=3)
    ax.add_collection(lc)

    for s_H_pt, col, shape in markers:
        P_pt = float(P_slice[np.argmin(np.abs(s_H_values - s_H_pt))])
        ax.scatter(s_H_pt, P_pt, s=90, c=col, marker=shape, zorder=5)

    ax.set_xlim(s_H_values[0], s_H_values[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.tick_params(axis='both', labelsize=15 * fontscale)
    ax.set_xlabel(r"Social Learning for H ($s_H$)", fontsize=20 * fontscale)
    ax.set_ylabel(r"Equilibrium Priority of H ($P^*$)",
                  fontsize=ylabel_fontsize if ylabel_fontsize is not None else 20 * fontscale)
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1.2, alpha=0.5, zorder=0)
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def figureR2_perturbations_and_phases(save=True):
    """
    Figure R2 (manuscript Fig. 4): both reduced social learning and intergroup
    learning can restore prioritization of H. 2x3 grid. Left column:
    perturbation time series. Middle column: horizontal slices P*(s_H) of the
    equilibrium heatmaps at fixed severity I = 0.75. Right column: equilibrium
    priority heatmaps over s_H and shared objective severity (I_H = I_L).

        A (s_H perturbation, rho=0) | B (slice, rho=0)   | C (heatmap, rho=0)
        D (rho perturbation)        | E (slice, rho=0.2) | F (heatmap, rho=0.2)

    Markers highlight the same operating points across each row's slice and
    heatmap (circle: s_H=0.9, rho=0; square: s_H=0.6, rho=0; triangle: s_H=0.9,
    rho=0.2). The slices show priority dropping discontinuously as s_H increases
    past a threshold.
    """
    b = _N_BASELINE
    t_final = b['t_final']
    linewidth = 2.5
    I_slice = 0.75  # severity level at which the middle-column slices are taken
    fs = 1.3              # font-size scale for the whole figure (labels/ticks)
    eq_label_fs = 23      # y-axis + colorbar label size for the equilibrium panels (B,C,E,F)
    eq_xticks = [0.4, 0.6, 0.8, 1.0]  # s_H ticks for the equilibrium panels
    eq_cbar_labelpad = 16  # gap between colorbar and its label
    eq_cbar_pad = 0.04     # gap between heatmap and colorbar

    data_base = run_N_baseline()
    data_sH = run_N_baseline(s_H=0.6)
    data_rho = run_N_baseline(rho=0.2)
    t = data_base['t']

    ts_legend_kwargs = dict(
        labelspacing=0.3, handlelength=3.0, handleheight=1.5, handletextpad=0.5,
        borderpad=0.3, borderaxespad=0.3, columnspacing=0.6, frameon=True, fontsize=15,
    )

    y_min = 0.5

    # --- Figure layout: 2 rows x 3 cols (A B C / D E F) ---
    fig, axs = plt.subplots(2, 3, figsize=(21, 12))
    ax_A, ax_B, ax_C = axs[0, 0], axs[0, 1], axs[0, 2]
    ax_D, ax_E, ax_F = axs[1, 0], axs[1, 1], axs[1, 2]

    # === Panel A: s_H perturbation time series ===
    ax_A.plot(t, data_base['P'], color=COLORS['primary1'], linewidth=linewidth, linestyle='-',
              alpha=0.9, label='Baseline')
    ax_A.plot(data_sH['t'], data_sH['P'], color=COLORS['primary4'], linewidth=linewidth, linestyle='-.',
              alpha=0.9, label=r'Reduced Social Learning ($s_H = 0.6$)')
    ax_A.set_ylabel(r"Priority of H ($P$)", fontsize=20 * fs)
    ax_A.set_xlabel("Time", fontsize=20 * fs)
    ax_A.legend(loc='upper right', **ts_legend_kwargs)

    # === Panel D: rho perturbation time series ===
    ax_D.plot(t, data_base['P'], color=COLORS['primary1'], linewidth=linewidth, linestyle='-',
              alpha=0.9, label='Baseline')
    ax_D.plot(data_rho['t'], data_rho['P'], color=COLORS['primary4'], linewidth=linewidth, linestyle='--',
              alpha=0.9, label=r'with Intergroup Learning ($\rho = 0.2$)')
    ax_D.set_ylabel(r"Priority of H ($P$)", fontsize=20 * fs)
    ax_D.set_xlabel("Time", fontsize=20 * fs)
    ax_D.legend(loc='upper right', **ts_legend_kwargs)

    # Time-series axis formatting
    for ax in [ax_A, ax_D]:
        ax.set_xlim(0, t_final)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.tick_params(axis='both', labelsize=15 * fs)
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1.2, alpha=0.5, zorder=0)
        ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # === Right column: phase heatmaps (zoomed y-range, with marker dots) ===
    s_H_C, I_C, heat_C = compute_priority_heatmap(b['s_L'], b['sigma'], b['alpha'], 0.0, ic=b['ic'])
    draw_priority_heatmap(ax_C, fig, s_H_C, I_C, heat_C, y_min=y_min, fontscale=fs,
                          ylabel_fontsize=eq_label_fs, cbar_label_fontsize=eq_label_fs, xticks=eq_xticks,
                          cbar_labelpad=eq_cbar_labelpad, cbar_pad=eq_cbar_pad)
    for s_H_pt, col, shape in _R2_MARKERS_RHO0:
        ax_C.scatter(s_H_pt, I_slice, s=90, c=col, marker=shape, zorder=5)

    s_H_D, I_D, heat_D = compute_priority_heatmap(b['s_L'], b['sigma'], b['alpha'], 0.2, ic=b['ic'])
    draw_priority_heatmap(ax_F, fig, s_H_D, I_D, heat_D, y_min=y_min, fontscale=fs,
                          ylabel_fontsize=eq_label_fs, cbar_label_fontsize=eq_label_fs, xticks=eq_xticks,
                          cbar_labelpad=eq_cbar_labelpad, cbar_pad=eq_cbar_pad)
    for s_H_pt, col, shape in _R2_MARKERS_RHO02:
        ax_F.scatter(s_H_pt, I_slice, s=90, c=col, marker=shape, zorder=5)

    # === Middle column: horizontal slices of the heatmaps at I = I_slice ===
    # Extract the slice directly from the heatmap rows so B/E align exactly with C/F.
    row_C = int(np.argmin(np.abs(I_C - I_slice)))
    row_D = int(np.argmin(np.abs(I_D - I_slice)))
    _draw_priority_slice(ax_B, s_H_C, heat_C[row_C, :], _R2_MARKERS_RHO0, fontscale=fs,
                         ylabel_fontsize=eq_label_fs, xticks=eq_xticks)
    _draw_priority_slice(ax_E, s_H_D, heat_D[row_D, :], _R2_MARKERS_RHO02, fontscale=fs,
                         ylabel_fontsize=eq_label_fs, xticks=eq_xticks)

    # Panel letters in reading order (A B C / D E F)
    for letter, ax in zip('ABCDEF', [ax_A, ax_B, ax_C, ax_D, ax_E, ax_F]):
        add_panel_label(ax, letter, fontsize=34)

    if save:
        fname = os.path.join(figures_dir, 'figureR2_perturbations_and_phases.pdf')
        plt.savefig(fname)
        print(f"Saved: {fname}")

    plt.close()
    return fig


def _draw_delayed_prioritization(ax, constant_IH, t_final=100):
    """Overlay objective severity I_H(t) (gray) and the priority of H over L
    at three levels of social learning (s_H = 0.6, 0.7, 0.9) on `ax`.

    Shared by figureR3 (rising severity) and figureR3S (severity held constant
    at its maximum). Social learning alone sets the equilibrium priority; the
    gradual increase in severity affects only the timing, delaying when
    priority reaches that equilibrium.
    """
    t_grid = np.linspace(0, t_final, 501)
    scenarios = figureR3_scenarios(constant_IH=constant_IH)

    # Plot I_H(t) — the objective severity (gray). All scenarios share the same
    # I_H(t); evaluate it directly via make_schedule rather than integrating the
    # system just for this curve.
    I_H_func = make_schedule(scenarios[0]['I_H'])
    I_H_curve = np.array([I_H_func(ti) for ti in t_grid])
    ax.plot(t_grid, I_H_curve, color='gray', linestyle=LINESTYLES['solid'],
            linewidth=2.5, alpha=0.85, label=r'Severity of $H$')

    for scn in scenarios:
        data = run_scenario(scn, t_final)
        ax.plot(data['t'], data['P'], color=scn['color'], linestyle=scn['linestyle'],
                linewidth=scn['linewidth'], alpha=0.85, label=scn['label'])

    _style_time_panel(ax, t_final)
    ax.set_ylabel(r"Severity of H ($I_H$) /" + "\n" + r"Priority of H ($P$)")
    ax.set_xlabel("Time")
    ax.legend(frameon=True)


def figureR3_delayed_prioritization(save=True):
    """
    Figure R3 (manuscript Fig. 5): higher social learning delays prioritization
    of issue H. Single panel; the gray line shows the objective severity of H,
    rising linearly until reaching 1, while L is held at 0.5. Colored lines
    show the priority of H over L at three levels of social learning. Stronger
    social learning lengthens the delay before priority reaches equilibrium.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    _draw_delayed_prioritization(ax, constant_IH=False)

    if save:
        fname = os.path.join(figures_dir, 'figureR3_delayed_prioritization.pdf')
        plt.savefig(fname)
        print(f"Saved: {fname}")

    plt.close()
    return fig


def figureR3S_delayed_prioritization(save=True):
    """
    Figure R3S (SI companion to figureR3; manuscript Fig. S2): with severity
    held constant at its maximum (I_H fixed at 1 from the outset), priority
    reaches equilibrium within 10 time units. Comparing with Fig. 5 shows that
    whether severity starts at its maximum or rises gradually to it affects
    only the lag in reaching the equilibrium priority, not the equilibrium
    level itself, which is determined by social learning.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    _draw_delayed_prioritization(ax, constant_IH=True)

    if save:
        fname = os.path.join(supp_figures_dir, 'figureR3S_delayed_prioritization.pdf')
        plt.savefig(fname)
        print(f"Saved: {fname}")

    plt.close()
    return fig


def figureR4_dynamic_connectivity(save=True):
    """
    Figure R4 (manuscript Fig. 6): restoring intergroup learning corrects
    underprioritization only above a threshold. Single P(t) panel; the two
    scenarios differ in the size of the increase in intergroup learning at
    t = 50. Styled to match figureR3_delayed_prioritization.
    """
    scenarios = figureR4_scenarios()
    t_final = 100
    vline_at = 50

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    for scn in scenarios:
        data = run_scenario(scn, t_final)
        ax.plot(data['t'], data['P'], color=scn['color'], linestyle=scn['linestyle'],
                linewidth=scn['linewidth'], alpha=0.85, label=scn['label'])

    _style_time_panel(ax, t_final, vline_at=vline_at)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Priority of H ($P$)")
    ax.legend(frameon=True)

    if save:
        fname = os.path.join(figures_dir, 'figureR4_dynamic_connectivity.pdf')
        plt.savefig(fname)
        print(f"Saved: {fname}")

    plt.close()
    return fig


def figureD1_step_sH_drop(save=True):
    """
    Figure D1 (Discussion section; manuscript Fig. 7): reducing social learning
    can shift collective priority toward the high-social-learning issue.
    Starting from the high-social-learning equilibrium of Fig. 5 (s_H = 0.9,
    constant severity I_H = 1.0, I_L = 0.5), s_H is reduced at t = 50 to 0.8,
    0.7, or 0.6. Larger reductions produce faster and more complete shifts.
    """
    scenarios = figureD1_scenarios()
    t_final = 100
    t_step = 50

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    for scn in scenarios:
        data = run_scenario(scn, t_final)
        ax.plot(data['t'], data['P'], color=scn['color'], linestyle=scn['linestyle'],
                linewidth=scn['linewidth'], alpha=0.85, label=scn['label'])

    _style_time_panel(ax, t_final, vline_at=t_step)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Priority of H ($P$)")
    ax.legend(frameon=True)

    if save:
        fname = os.path.join(figures_dir, 'figureD1_step_sH_drop.pdf')
        plt.savefig(fname)
        print(f"Saved: {fname}")

    plt.close()
    return fig


def figureR2S_phase_diagrams_full(save=True):
    """
    Figure R2S (SI for R2; manuscript Fig. S1): equilibrium priority of H
    across the full range of objective severity I_H = I_L in [0, 1], extending
    Fig. 4C and F. Panel A: without intergroup learning (rho = 0). Panel B:
    with intergroup learning (rho = 0.2). No marker dots or text annotations.
    """
    b = _N_BASELINE

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    ax_A, ax_B = axs

    s_H_A, I_A, heat_A = compute_priority_heatmap(b['s_L'], b['sigma'], b['alpha'], 0.0, ic=b['ic'])
    draw_priority_heatmap(ax_A, fig, s_H_A, I_A, heat_A)
    s_H_B, I_B, heat_B = compute_priority_heatmap(b['s_L'], b['sigma'], b['alpha'], 0.2, ic=b['ic'])
    draw_priority_heatmap(ax_B, fig, s_H_B, I_B, heat_B)

    add_panel_label(ax_A, 'A', fontsize=25)
    add_panel_label(ax_B, 'B', fontsize=25)

    if save:
        fname = os.path.join(supp_figures_dir, 'figureR2S_phase_diagrams_full.pdf')
        plt.savefig(fname)
        print(f"Saved: {fname}")

    plt.close()
    return fig


def figureR4S_dynamic_connectivity(save=True):
    """
    Figure R4S (SI for figureR4; manuscript Fig. S3): group-level dynamics
    underlying Fig. 6. 3-panel version with population priority P(t) on top,
    and group-level concerns for issues H and L (one curve per group) below,
    showing that only the large jump collapses between-group divergence.
    """
    return plot_two_issue_CEP(figureR4_scenarios(), t_final=100,
                              filename='figureR4S_dynamic_connectivity.pdf',
                              out_dir=supp_figures_dir, vline_at=50, save=save)


# =============================================================================
# Main Execution
# =============================================================================

def generate_all_figures():
    """Generate every figure in the paper (main and supplementary), in section
    order: Introduction -> Model -> Results -> Discussion -> Supplementary."""

    print("=" * 60)
    print("Introduction")
    print("=" * 60)
    figureI1_survey_firstdiff_scatter(save=True)

    print("\n" + "=" * 60)
    print("Model")
    print("=" * 60)
    figureM1_network_and_conformity(save=True)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    figureR1_two_issue_dynamics(save=True)
    figureR2_perturbations_and_phases(save=True)
    figureR3_delayed_prioritization(save=True)
    figureR4_dynamic_connectivity(save=True)

    print("\n" + "=" * 60)
    print("Discussion")
    print("=" * 60)
    figureD1_step_sH_drop(save=True)

    print("\n" + "=" * 60)
    print("Supplementary")
    print("=" * 60)
    figureR2S_phase_diagrams_full(save=True)
    figureR3S_delayed_prioritization(save=True)
    figureR4S_dynamic_connectivity(save=True)

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Main figures saved to: {figures_dir}")
    print(f"Supplementary figures saved to: {supp_figures_dir}")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_figures()
