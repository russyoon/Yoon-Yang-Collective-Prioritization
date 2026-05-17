"""Pure-numerics core for the social-learning / opinion-dynamics model."""

import numpy as np
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm


def f(x, alpha):
    """Conformity response function.

    Linear for alpha=1; an S-shaped normative-conformity curve for alpha > 1.
    """
    return x**alpha / (x**alpha + (1 - x)**alpha)


def compute_priority(C_H, C_L, sigma):
    """Compute priority of issue H over L using logit choice."""
    return np.exp(C_H / sigma) / (np.exp(C_H / sigma) + np.exp(C_L / sigma))


def make_schedule(param):
    """Coerce param into a callable f(t).

    Accepts a scalar, a callable, or a list of (t_break, value) tuples
    for a piecewise-constant schedule (held constant before the first break).
    """
    if callable(param):
        return param
    if np.isscalar(param):
        v = float(param)
        return lambda t: v
    try:
        sched = [(float(tb), float(v)) for tb, v in param]
    except Exception as e:
        raise ValueError(
            "schedule must be scalar, callable, or list of (t_break, value)"
        ) from e
    sched.sort(key=lambda x: x[0])
    breaks = np.array([tb for tb, _ in sched], dtype=float)
    values = np.array([v for _, v in sched], dtype=float)
    if breaks[0] > 0.0:
        breaks = np.insert(breaks, 0, 0.0)
        values = np.insert(values, 0, values[0])

    def of_t(t):
        idx = np.searchsorted(breaks, t, side='right') - 1
        idx = max(0, min(idx, len(values) - 1))
        return values[idx]
    return of_t


def two_group_equilibrium(C_1_0, C_2_0, s, I, alpha, rho, max_iter=500, tol=1e-8):
    """Fixed-point iteration for two-group equilibrium on a single issue."""
    C_1, C_2 = float(C_1_0), float(C_2_0)
    for _ in range(max_iter):
        next_C_1 = (1 - s) * I + s * f((C_1 + rho * C_2) / (1 + rho), alpha)
        next_C_2 = (1 - s) * I + s * f((C_2 + rho * C_1) / (1 + rho), alpha)
        diff = np.hypot(next_C_1 - C_1, next_C_2 - C_2)
        C_1, C_2 = next_C_1, next_C_2
        if diff < tol:
            break
    return C_1, C_2


def run_two_issue_system(initial_conditions, s_H, s_L, sigma, alpha, rho,
                         I_H, I_L, t_final=300, n_points=501):
    """Integrate the two-issue (H, L) social-learning system via solve_ivp.

    Implements manuscript Eq. (1) with tau = 1, so dC/dt = C* - C and
    "Time" on figure axes is in units of tau.

    Each of s_H, s_L, rho, I_H, I_L may be a scalar, a callable f(t), or
    a list of (t_break, value) tuples (piecewise-constant schedule).

    Returns a dict with t, group/population concerns for H and L, the
    population priority P, and the time-resolved I_H, I_L, rho.
    """
    C_1_H_0, C_2_H_0, C_1_L_0, C_2_L_0 = initial_conditions
    t_eval = np.linspace(0, t_final, n_points)

    s_H_func = make_schedule(s_H)
    s_L_func = make_schedule(s_L)
    rho_func = make_schedule(rho)
    I_H_func = make_schedule(I_H)
    I_L_func = make_schedule(I_L)

    def make_rhs(s_func, I_func):
        def rhs(t, y):
            C_1, C_2 = y
            s = s_func(t)
            I = I_func(t)
            r = rho_func(t)
            dC_1 = (1 - s) * I + s * f((C_1 + r * C_2) / (1 + r), alpha) - C_1
            dC_2 = (1 - s) * I + s * f((C_2 + r * C_1) / (1 + r), alpha) - C_2
            return [dC_1, dC_2]
        return rhs

    sol_H = solve_ivp(make_rhs(s_H_func, I_H_func),
                      [0, t_final], [C_1_H_0, C_2_H_0], t_eval=t_eval)
    sol_L = solve_ivp(make_rhs(s_L_func, I_L_func),
                      [0, t_final], [C_1_L_0, C_2_L_0], t_eval=t_eval)

    t = sol_H.t
    C_1_H, C_2_H = sol_H.y
    C_1_L, C_2_L = sol_L.y

    C_H = 0.5 * (C_1_H + C_2_H)
    C_L = 0.5 * (C_1_L + C_2_L)
    P = compute_priority(C_H, C_L, sigma)

    I_H_t = np.array([I_H_func(ti) for ti in t])
    I_L_t = np.array([I_L_func(ti) for ti in t])
    rho_t = np.array([rho_func(ti) for ti in t])

    return {
        't': t,
        'C_1_H': C_1_H, 'C_2_H': C_2_H, 'C_H': C_H,
        'C_1_L': C_1_L, 'C_2_L': C_2_L, 'C_L': C_L,
        'P': P,
        'I_H': I_H_t, 'I_L': I_L_t, 'rho_t': rho_t,
    }


def compute_priority_heatmap(s_L, sigma, alpha, rho_val,
                              ic=(0.55, 0.45, 0.45, 0.55), n_I=501):
    """Compute equilibrium priority P over the (s_H, I=I_H=I_L) grid.

    Shared by figureR2 (zoomed y-range view) and figureR2S (full y-range view).
    """
    C_1_H_0, C_2_H_0, C_1_L_0, C_2_L_0 = ic
    s_H_values = np.linspace(s_L, 1, int((1 - s_L) * 1000) + 1)
    I_values = np.linspace(0, 1, n_I)
    heatmap_P = np.zeros((len(I_values), len(s_H_values)))

    print(f"Computing heatmap for rho={rho_val}...")
    for i, I_val in tqdm(enumerate(I_values), total=len(I_values)):
        I_H = I_L = I_val
        for j, s_H in enumerate(s_H_values):
            C_1_H_f, C_2_H_f = two_group_equilibrium(
                C_1_H_0, C_2_H_0, s_H, I_H, alpha, rho_val,
                max_iter=1000, tol=1e-8)
            C_1_L_f, C_2_L_f = two_group_equilibrium(
                C_1_L_0, C_2_L_0, s_L, I_L, alpha, rho_val,
                max_iter=1000, tol=1e-8)
            C_H = 0.5 * (C_1_H_f + C_2_H_f)
            C_L = 0.5 * (C_1_L_f + C_2_L_f)
            heatmap_P[i, j] = compute_priority(C_H, C_L, sigma)
    return s_H_values, I_values, heatmap_P
