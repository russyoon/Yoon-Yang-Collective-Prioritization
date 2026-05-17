"""This module produces the figures for the manuscript. Run
``python figures.py`` to regenerate everything.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import networkx as nx

from model import (
    f,
    run_two_issue_system,
    compute_priority_heatmap,
)
from plot_utils import (
    COLORS,
    LINESTYLES,
    SCENARIO_COLORS,
    SCENARIO_LINESTYLES,
    figures_dir,
    supp_figures_dir,
    add_panel_label,
    _build_network,
    draw_priority_heatmap,
    _style_time_panel,
    plot_two_issue_CEP,
    run_N_baseline,
    _N_BASELINE,
    figureR3_scenarios,
    figureR4_scenarios,
    figureD1_scenarios,
)


def figureI1_survey_firstdiff_scatter(save=True):
    """
    Figure I1 (Introduction section, first figure): two-panel survey data
    analysis.
    Panel A: Gallup climate worry, Pew climate priority, Pew economy priority
             (restricted to overlapping years).
    Panel B: Scatter of year-over-year first differences of Pew climate vs
             economy priority, highlighting the trade-off (quadrants II and IV).
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

    slope, intercept = np.polyfit(d_climate.values, d_economy.values, 1)
    x_fit = np.linspace(d_climate.min(), d_climate.max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, color='#c62828', lw=1.5, ls='--')

    r = np.corrcoef(d_climate.values, d_economy.values)[0, 1]
    ax.text(0.05, 0.95, f'r = {r:.2f}', transform=ax.transAxes,
            fontsize=15, va='top', color='#c62828')

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
    Figure M1 (Model section, first figure): side-by-side overview of the
    two ingredients of the model. The left panel shows the two-group network
    structure as two stacked sub-panels (rho = 0 on top, rho = 0.5 below).
    The right panel shows the conformity response S-curve f(X; alpha) for
    several alpha values, spanning the full figure height.
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
    subtitles = [r'No inter-group learning ($\rho=0$)',
                 r'Moderate inter-group learning ($\rho=0.5$)']

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
        ax.set_title(subtitle, fontsize=10, style='italic')
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


def figureR1_two_issue_dynamics(save=True, wide=False):
    """
    Figure R1: Three-panel time series under the shared baseline.
    A: P(t) and 1 - P(t)
    B: population concern C_H(t) and C_L(t)
    C: group trajectories C_{1H}(t) and C_{2H}(t)

    Parameters
    ----------
    wide : bool, default False
        Aspect-ratio variant. If False, uses the tall layout (figsize=(6, 9))
        and saves as figureR1_two_issue_dynamics.pdf. If True, uses the
        wider/shorter layout (figsize=(7, 7)) and saves as
        figureR1_two_issue_dynamics_wide.pdf. Plot content is identical
        between the two variants.
    """
    from matplotlib.ticker import MultipleLocator
    data = run_N_baseline()
    t = data['t']
    t_final = _N_BASELINE['t_final']

    linewidth = 2.5
    legend_kwargs = dict(
        labelspacing=0.3, handlelength=3.0, handleheight=1.5, handletextpad=0.5,
        borderpad=0.3, borderaxespad=0.3, columnspacing=0.6, frameon=True,
    )

    figsize = (7, 7) if wide else (6, 9)
    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)
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

    # Panel C: group-level trajectories for issue H (purple)
    ax_C.plot(t, data['C_1_H'], color='#6a1b9a', linewidth=linewidth, linestyle='-.',
              alpha=0.85, label='Group 1')
    ax_C.plot(t, data['C_2_H'], color='#6a1b9a', linewidth=linewidth, linestyle=':',
              alpha=0.85, label='Group 2')
    ax_C.set_ylabel("Group concern\n" + r"for issue H ($C_{gH}$)")
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
        suffix = '_wide' if wide else ''
        fname = os.path.join(figures_dir, f'figureR1_two_issue_dynamics{suffix}.pdf')
        plt.savefig(fname)
        print(f"Saved: {fname}")

    plt.close()
    return fig


def figureR2_perturbations_and_phases(save=True):
    """
    Figure R2: 2x2 grid pairing perturbation time series (left column) with
    priority phase diagrams (right column).

        A (s_H perturbation, rho=0)    |   C (phase heatmap, rho=0)
        B (rho perturbation, s_H=0.9)  |   D (phase heatmap, rho=0.2)
    """
    from matplotlib.ticker import MultipleLocator

    # --- Shared parameters & data ---
    b = _N_BASELINE
    t_final = b['t_final']
    linewidth = 2.5

    data_base = run_N_baseline()
    data_sH = run_N_baseline(s_H=0.6)
    data_rho = run_N_baseline(rho=0.2)
    t = data_base['t']

    ts_legend_kwargs = dict(
        labelspacing=0.3, handlelength=3.0, handleheight=1.5, handletextpad=0.5,
        borderpad=0.3, borderaxespad=0.3, columnspacing=0.6, frameon=True, fontsize=12,
    )

    # Phase-diagram model parameters
    C_1_H_0, C_2_H_0, C_1_L_0, C_2_L_0 = 0.55, 0.45, 0.45, 0.55
    s_L = 0.4
    sigma = 0.25
    alpha_p = 3
    y_min = 0.5

    # --- Figure layout: 2 rows x 2 cols ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    ax_A, ax_C = axs[0, 0], axs[0, 1]
    ax_B, ax_D = axs[1, 0], axs[1, 1]

    # === Panel A: s_H perturbation time series ===
    ax_A.plot(t, data_base['P'], color=COLORS['primary1'], linewidth=linewidth, linestyle='-',
              alpha=0.9, label='Baseline')
    ax_A.plot(data_sH['t'], data_sH['P'], color=COLORS['primary4'], linewidth=linewidth, linestyle='-.',
              alpha=0.9, label=r'Reduced Social Learning ($s_H = 0.6$)')
    ax_A.set_ylabel(r"Priority of H ($P$)", fontsize=20)
    ax_A.set_xlabel("Time", fontsize=20)
    ax_A.legend(loc='upper right', **ts_legend_kwargs)

    # === Panel B: rho perturbation time series ===
    ax_B.plot(t, data_base['P'], color=COLORS['primary1'], linewidth=linewidth, linestyle='-',
              alpha=0.9, label='Baseline')
    ax_B.plot(data_rho['t'], data_rho['P'], color=COLORS['primary4'], linewidth=linewidth, linestyle='--',
              alpha=0.9, label=r'with Inter-group Learning ($\rho = 0.2$)')
    ax_B.set_ylabel(r"Priority of H ($P$)", fontsize=20)
    ax_B.set_xlabel("Time", fontsize=20)
    ax_B.legend(loc='upper right', **ts_legend_kwargs)

    # Time-series axis formatting
    for ax in [ax_A, ax_B]:
        ax.set_xlim(0, t_final)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.tick_params(axis='both', labelsize=15)
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1.2, alpha=0.5, zorder=0)
        ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # === Panels C & D: phase heatmaps (zoomed y-range, with marker dots) ===
    ic = (C_1_H_0, C_2_H_0, C_1_L_0, C_2_L_0)

    s_H_C, I_C, heat_C = compute_priority_heatmap(s_L, sigma, alpha_p, 0.0, ic=ic)
    draw_priority_heatmap(ax_C, fig, s_H_C, I_C, heat_C, y_min=y_min)
    for (xp, yp), col in zip([(0.9, 0.75), (0.6, 0.75)],
                             [COLORS['primary1'], COLORS['primary2']]):
        ax_C.scatter(xp, yp, s=40, c=col, marker='o', zorder=5)

    s_H_D, I_D, heat_D = compute_priority_heatmap(s_L, sigma, alpha_p, 0.2, ic=ic)
    draw_priority_heatmap(ax_D, fig, s_H_D, I_D, heat_D, y_min=y_min)
    ax_D.scatter(0.9, 0.75, s=40, c=COLORS['primary2'], marker='o', zorder=5)

    # Panel letters in reading order (A C / B D)
    add_panel_label(ax_A, 'A', fontsize=25)
    add_panel_label(ax_C, 'C', fontsize=25)
    add_panel_label(ax_B, 'B', fontsize=25)
    add_panel_label(ax_D, 'D', fontsize=25)

    if save:
        fname = os.path.join(figures_dir, 'figureR2_perturbations_and_phases.pdf')
        plt.savefig(fname)
        print(f"Saved: {fname}")

    plt.close()
    return fig


def figureR3_path_dependence(save=True):
    """
    Figure R3: Single-panel figure overlaying rising objective severity
    I_H(t) and population priority P(t) for three social-learning strengths
    (s_H = 0.6, 0.7, 0.9). Shows how stronger social learning leads to
    delayed or failed prioritization as severity ramps up.
    """
    t_final = 100
    scenarios = figureR3_scenarios()

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    # Plot I_H(t) — the rising objective severity (gray)
    data0 = run_two_issue_system(
        scenarios[0]['initial_conditions'],
        scenarios[0]['s_H'], scenarios[0]['s_L'], scenarios[0]['sigma'],
        scenarios[0]['alpha'], scenarios[0]['rho'],
        scenarios[0]['I_H_func'], scenarios[0]['I_L_func'], t_final
    )
    ax.plot(data0['t'], data0['I_H'], color='gray', linestyle=LINESTYLES['solid'],
            linewidth=2.5, alpha=0.85, label=r'Severity of $H$')

    for scn in scenarios:
        data = run_two_issue_system(
            scn['initial_conditions'],
            scn['s_H'], scn['s_L'], scn['sigma'],
            scn['alpha'], scn['rho'],
            scn['I_H_func'], scn['I_L_func'], t_final
        )
        ax.plot(data['t'], data['P'], color=scn['color'], linestyle=scn['linestyle'],
                linewidth=scn['linewidth'], alpha=0.85, label=scn['label'])

    _style_time_panel(ax, t_final)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Severity of H ($I_H$) /" + "\n" + r"Priority of H ($P$)")
    ax.legend(frameon=True)

    if save:
        fname = os.path.join(figures_dir, 'figureR3_path_dependence.pdf')
        plt.savefig(fname)
        print(f"Saved: {fname}")

    plt.close()
    return fig


def figureR4_dynamic_connectivity(save=True):
    """
    Figure R4: single P(t) panel showing how inter-group connectivity alters priority.
    Styled to match figureR3_path_dependence.
    """
    scenarios = figureR4_scenarios()
    t_final = 100
    vline_at = 50

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    for scn in scenarios:
        data = run_two_issue_system(
            scn['initial_conditions'],
            scn['s_H'], scn['s_L'], scn['sigma'],
            scn['alpha'], scn['rho'],
            scn['I_H_func'], scn['I_L_func'], t_final
        )
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
    Figure D1 (Discussion section, first figure; manuscript Fig. 7):
    three-scenario comparison with a step drop in s_H at t = 50 under constant
    severity (I_H = 1.0, I_L = 0.5). The pre-step window t < 50 equilibrates
    the system at the high-social-learning, mis-prioritized state from
    figureR3's s_H = 0.9 scenario. At t = 50, s_H drops to 0.8, 0.7, or 0.6;
    larger drops produce faster and more complete recovery of priority for H.
    """
    scenarios = figureD1_scenarios()
    t_final = 100
    t_step = 50

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    for scn in scenarios:
        data = run_two_issue_system(
            scn['initial_conditions'], scn['s_H'], scn['s_L'], scn['sigma'],
            scn['alpha'], scn['rho'],
            scn['I_H_func'], scn['I_L_func'], t_final=t_final, n_points=501
        )
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
    Figure R2S (SI for R2): 1x2 side-by-side priority phase heatmaps over the
    full y-range I_H = I_L in [0, 1]. Panel A is rho = 0, panel B is rho = 0.2.
    No marker dots and no text annotations.
    """
    # Phase-diagram model parameters
    C_1_H_0, C_2_H_0, C_1_L_0, C_2_L_0 = 0.55, 0.45, 0.45, 0.55
    s_L = 0.4
    sigma = 0.25
    alpha_p = 3

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    ax_A, ax_B = axs

    ic = (C_1_H_0, C_2_H_0, C_1_L_0, C_2_L_0)
    s_H_A, I_A, heat_A = compute_priority_heatmap(s_L, sigma, alpha_p, 0.0, ic=ic)
    draw_priority_heatmap(ax_A, fig, s_H_A, I_A, heat_A)
    s_H_B, I_B, heat_B = compute_priority_heatmap(s_L, sigma, alpha_p, 0.2, ic=ic)
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
    Figure R4S (SI for figureR4; manuscript Fig. S2): 3-panel version with
    population priority P(t) on top, and group-level concerns for issues H
    and L (one curve per group) below.
    """
    return plot_two_issue_CEP(figureR4_scenarios(), t_final=100,
                              filename='figureR4S_dynamic_connectivity.pdf',
                              vertical=True, vline_at=50, save=save,
                              priority_first=True, out_dir=supp_figures_dir)


# =============================================================================
# Main Execution
# =============================================================================

def generate_all_figures():
    """Generate all main figures for the paper, in section order:
    Introduction -> Model -> Results -> Discussion."""

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
    figureR1_two_issue_dynamics(save=True, wide=True)
    figureR2_perturbations_and_phases(save=True)
    figureR3_path_dependence(save=True)
    figureR4_dynamic_connectivity(save=True)

    print("\n" + "=" * 60)
    print("Discussion")
    print("=" * 60)
    figureD1_step_sH_drop(save=True)

    print("\n" + "=" * 60)
    print("Supplementary")
    print("=" * 60)
    figureR2S_phase_diagrams_full(save=True)
    figureR4S_dynamic_connectivity(save=True)

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Figures saved to: {figures_dir}")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_figures()
