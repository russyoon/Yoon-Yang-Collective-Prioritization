"""Shared plotting infrastructure: style constants, helpers, and reusable
panels for the figure scripts."""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from model import run_two_issue_system, make_schedule


def run_scenario(scn, t_final, **kwargs):
    """Integrate a scenario dict through run_two_issue_system."""
    return run_two_issue_system(
        scn['initial_conditions'],
        scn['s_H'], scn['s_L'], scn['sigma'],
        scn['alpha'], scn['rho'],
        scn['I_H'], scn['I_L'], t_final, **kwargs,
    )

# Setup directories
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
figures_dir = os.path.join(current_dir, "main_figures")
supp_figures_dir = os.path.join(current_dir, "supp_figures")
for _d in (figures_dir, supp_figures_dir):
    if not os.path.exists(_d):
        os.makedirs(_d)

# =============================================================================
# Unified Color and Linestyle Palette
# =============================================================================
COLORS = {
    'primary1': '#0d47a1',    # Deep blue
    'primary2': '#e65100',    # Deep orange
    'primary3': '#1b5e20',    # Deep green
    'primary4': '#b71c1c',    # Deep red
    'neutral': '#424242',     # Dark gray
    'black': 'black',
}

# Linestyles with good visual distinction (tuples for precise control)
LINESTYLES = {
    'solid': '-',
    'dashed': (0, (5, 2)),           # Longer dashes, tighter gaps
    'dashdot': (0, (5, 2, 1, 2)),    # Long dash, short dash
    'dotted': (0, (1, 1.5)),         # Tight dots
}

# Combined cycles for scenarios (up to 4 scenarios)
SCENARIO_COLORS = [COLORS['primary1'], COLORS['primary2'], COLORS['primary3'], COLORS['primary4']]
SCENARIO_LINESTYLES = [LINESTYLES['solid'], LINESTYLES['dashed'], LINESTYLES['dotted'], LINESTYLES['dashdot']]

# Set matplotlib parameters for publication quality
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 10,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'savefig.transparent': True,
    'figure.constrained_layout.use': True,
})


def add_panel_label(ax, label, x=-0.2, y=1.1, fontsize=None):
    """Add a panel label (e.g., 'A', 'B') to an axes.

    If `fontsize` is None, falls back to `rcParams['font.size'] + 8`.
    Pass an explicit fontsize (e.g. 25) for larger publication-scale labels.
    """
    if fontsize is None:
        fontsize = plt.rcParams['font.size'] + 8
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold',
            va='top', ha='left')


def _build_network(group1, group2, rho_inter, rng):
    """Build a two-group network with full intra-group and stochastic inter-group edges."""
    G = nx.Graph()
    G.add_nodes_from(group1, group='Group 1')
    G.add_nodes_from(group2, group='Group 2')

    # Fully connected within each group
    for i in range(len(group1)):
        for j in range(i + 1, len(group1)):
            G.add_edge(group1[i], group1[j])
    for i in range(len(group2)):
        for j in range(i + 1, len(group2)):
            G.add_edge(group2[i], group2[j])

    # Inter-group connections
    if rho_inter > 0:
        for g1_node in group1:
            for g2_node in group2:
                if rng.random() < rho_inter:
                    G.add_edge(g1_node, g2_node)

    return G


def _style_time_panel(ax, t_final, vline_at=None):
    """Apply the canonical R3/D1 main-figure styling to a time-series axis.

    Sets xlim/ylim, the [0, 0.2tf, 0.4tf, 0.5tf, 0.6tf, 0.8tf, tf] major /
    [0.1tf, 0.3tf, 0.7tf, 0.9tf] minor xtick scheme, the 0/0.25/0.5/0.75/1
    yticks, major+minor grids, hidden top/right spines, the y=0.5 horizontal
    reference, and an optional vertical reference at `vline_at`.
    """
    def _tt(frac):
        v = frac * t_final
        return int(round(v)) if abs(v - round(v)) < 1e-9 else v

    ax.set_xlim(0, t_final)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticks([_tt(0), _tt(0.2), _tt(0.4), _tt(0.5), _tt(0.6), _tt(0.8), _tt(1)])
    ax.set_xticks([_tt(0.1), _tt(0.3), _tt(0.7), _tt(0.9)], minor=True)
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.grid(True, which='minor', linestyle='-', linewidth=0.3, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0.5, color='gray', linestyle='-', linewidth=1, alpha=0.5, zorder=0)
    if vline_at is not None:
        ax.axvline(x=vline_at, color='gray', linestyle='-', linewidth=1, alpha=0.5, zorder=0)


def draw_priority_heatmap(ax, fig, s_H_values, I_values, heatmap_P, y_min=None):
    """Render the priority heatmap on `ax` with consistent styling.

    If `y_min` is provided, zoom the y-axis to [y_min, 1] (matches figureR2);
    otherwise show the full I range with sparser ticks (matches figureR2S).
    """
    im = ax.imshow(
        heatmap_P,
        extent=[s_H_values[0], s_H_values[-1], I_values[0], I_values[-1]],
        aspect="auto", origin="lower", cmap="coolwarm", vmin=0, vmax=1
    )
    ax.set_xlabel(r"Social Learning for H ($s_H$)", fontsize=20)
    ax.set_ylabel(r"Objective Severity ($I_H = I_L$)", fontsize=20)
    ax.set_xticks(np.arange(round(s_H_values[0], 2),
                            round(s_H_values[-1] + 0.01, 2), 0.1))
    if y_min is not None:
        ax.set_ylim(y_min, 1)
        ax.set_yticks(np.arange(y_min, 1.01, 0.1))
    else:
        ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis='both', labelsize=15)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"Priority of H ($P$)", fontsize=15)
    cbar.ax.tick_params(labelsize=12)
    ax.contour(
        heatmap_P, levels=[0.5], colors="black",
        extent=[s_H_values[0], s_H_values[-1], I_values[0], I_values[-1]],
        linewidths=2
    )


def plot_two_issue_CEP(scenarios, t_final=300, linewidth=2, filename=None,
                      vertical=False, vline_at=None, width=8, height=9,
                      save=True, priority_first=False, out_dir=None):
    """
    Overlay C_H(t), C_L(t), and P(t) for multiple scenarios. Uses the canonical
    R3/D1 main-figure styling on every panel (see `_style_time_panel`).

    Parameters
    ----------
    scenarios : list of dict
        Each dict contains: 'label', 'initial_conditions', 's_H', 's_L', 'sigma',
        'alpha', 'rho', 'I_H', 'I_L'. Optional 'color', 'linestyle',
        'linewidth' override the default cycle.
    vertical : bool
        If True, 3x1 layout with shared x; else 1x3.
    vline_at : float, optional
        Vertical reference line time.
    """
    if vertical:
        fig, axs = plt.subplots(3, 1, figsize=(width, height), sharex=True)
        if priority_first:
            axP, axH, axL = axs
        else:
            axH, axL, axP = axs
    else:
        fig, axs = plt.subplots(1, 3, figsize=(3 * width, height / 3))
        axH, axL, axP = axs

    for i, scn in enumerate(scenarios):
        color = scn.get('color', SCENARIO_COLORS[i % len(SCENARIO_COLORS)])
        linestyle = scn.get('linestyle', SCENARIO_LINESTYLES[i % len(SCENARIO_LINESTYLES)])
        lw = scn.get('linewidth', linewidth)

        data = run_scenario(scn, t_final)
        t = data['t']
        label = scn.get('label', f"scenario {i+1}")

        # C_H(t): plot both groups with same color/linestyle; one legend entry per scenario
        axH.plot(t, data['C_1_H'], color=color, linestyle=linestyle, linewidth=lw,
                 alpha=0.85, label=label)
        axH.plot(t, data['C_2_H'], color=color, linestyle=linestyle, linewidth=lw,
                 alpha=0.85, label='_nolegend_')

        # C_L(t): issue L
        axL.plot(t, data['C_1_L'], color=color, linestyle=linestyle, linewidth=lw,
                 alpha=0.85, label=label)
        axL.plot(t, data['C_2_L'], color=color, linestyle=linestyle, linewidth=lw,
                 alpha=0.85, label='_nolegend_')

        # P(t): population priority of H
        axP.plot(t, data['P'], color=color, linestyle=linestyle, linewidth=lw,
                 alpha=0.9, label=label)

    # Canonical R3/D1 main-figure styling on every panel.
    for ax in [axH, axL, axP]:
        _style_time_panel(ax, t_final, vline_at=vline_at)

    # Labels (match R1/R3/D1 phrasing).
    if priority_first:
        bottom_ax = axL
    else:
        bottom_ax = axP
    if vertical:
        bottom_ax.set_xlabel("Time")
    else:
        for ax in [axH, axL, axP]:
            ax.set_xlabel("Time")
    axH.set_ylabel("Group concern\n" + r"for issue H ($C_{gH}$)")
    axL.set_ylabel("Group concern\n" + r"for issue L ($C_{gL}$)")
    axP.set_ylabel(r"Priority of H ($P$)")

    # Legend on the priority panel only (matches R3/D1 main figures).
    axP.legend(frameon=True)

    # Panel letters A, B, C
    if priority_first:
        add_panel_label(axP, 'A')
        add_panel_label(axH, 'B')
        add_panel_label(axL, 'C')
    else:
        for letter, ax in zip(['A', 'B', 'C'], [axH, axL, axP]):
            add_panel_label(ax, letter)

    if save:
        target_dir = out_dir if out_dir is not None else figures_dir
        fname = os.path.join(target_dir, filename if filename else "two_issue_CEP.pdf")
        plt.savefig(fname)
        print(f"Saved: {fname}")

    plt.close()
    return fig


# Shared baseline parameters for R1 / R2 / R2S (s_L = 0.4)
_N_BASELINE = dict(
    ic=(0.55, 0.45, 0.45, 0.55),
    s_H=0.9, s_L=0.4,
    sigma=0.25, alpha=3, rho=0,
    I_H=0.75, I_L=0.75,
    t_final=20,
)


def run_N_baseline(s_H=None, rho=None):
    b = _N_BASELINE
    s_H = b['s_H'] if s_H is None else s_H
    rho = b['rho'] if rho is None else rho
    return run_two_issue_system(
        b['ic'], s_H, b['s_L'], b['sigma'], b['alpha'], rho,
        b['I_H'], b['I_L'],
        t_final=b['t_final'], n_points=501
    )


def figureR4_scenarios():
    ivs = (0.55, 0.45, 0.45, 0.55)
    return [
        {
            'label': r'Small jump ($\rho: 0 \to 0.2$)',
            'initial_conditions': ivs,
            's_H': 0.9, 's_L': 0.4, 'sigma': 0.25, 'alpha': 3,
            'rho': [(0, 0.0), (50, 0.2)],
            'I_H': 0.75,
            'I_L': 0.75,
            'color': COLORS['primary1'],
            'linestyle': LINESTYLES['solid'],
            'linewidth': 2.5,
        },
        {
            'label': r'Large jump ($\rho: 0 \to 0.5$)',
            'initial_conditions': ivs,
            's_H': 0.9, 's_L': 0.4, 'sigma': 0.25, 'alpha': 3,
            'rho': [(0, 0.0), (50, 0.5)],
            'I_H': 0.75,
            'I_L': 0.75,
            'color': COLORS['primary4'],
            'linestyle': LINESTYLES['dashed'],
            'linewidth': 2.5,
        }
    ]


# Custom dotted linestyle used by R3 / D1 for the most extreme scenario.
_BIG_DOTTED = (0, (2.5, 2.5))


def figureR3_scenarios():
    """R3's three rising-severity scenarios (s_H sweep) with baked-in styles."""
    ivs = (0, 0, 0.5, 0.5)
    rho0 = 0
    I_H_ramp = lambda t: np.clip(1/60*(t-20), 0, 1)
    styled = [
        (r'$s_H = 0.6$', 0.6, '#7f0000',          _BIG_DOTTED),
        (r'$s_H = 0.7$', 0.7, COLORS['primary4'], LINESTYLES['dashdot']),
        (r'$s_H = 0.9$', 0.9, COLORS['primary1'], LINESTYLES['dashed']),
    ]
    return [
        {
            'label': lbl,
            'initial_conditions': ivs,
            's_H': s_H, 's_L': 0.4, 'sigma': 0.25, 'alpha': 3, 'rho': rho0,
            'I_H': I_H_ramp,
            'I_L': 0.5,
            'color': color,
            'linestyle': ls,
            'linewidth': 2.5,
        }
        for (lbl, s_H, color, ls) in styled
    ]


def figureD1_scenarios():
    """D1's three step-drop-in-s_H scenarios with baked-in styles."""
    ic = (0, 0, 0.5, 0.5)
    s_H_start = 0.9
    t_step = 50

    def make_step(floor):
        return lambda t: s_H_start if t < t_step else floor

    styled = [
        (r'Small drop ($s_H: 0.9 \to 0.8$)',    0.8, COLORS['primary1'], LINESTYLES['solid']),
        (r'Moderate drop ($s_H: 0.9 \to 0.7$)', 0.7, COLORS['primary4'], LINESTYLES['dashdot']),
        (r'Large drop ($s_H: 0.9 \to 0.6$)',    0.6, '#7f0000',          _BIG_DOTTED),
    ]
    return [
        {
            'label': lbl,
            'initial_conditions': ic,
            's_H': make_step(floor),
            's_L': 0.4, 'sigma': 0.25, 'alpha': 3.0, 'rho': 0,
            'I_H': 1.0,
            'I_L': 0.5,
            'color': color,
            'linestyle': ls,
            'linewidth': 2.5,
        }
        for (lbl, floor, color, ls) in styled
    ]
