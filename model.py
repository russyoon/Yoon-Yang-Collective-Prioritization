"""Pure-numerics core for the social-learning / opinion-dynamics model."""

import numpy as np
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm


def f(x, alpha):
    """Nonlinear conformity response function."""
    return x**alpha / (x**alpha + (1 - x)**alpha)


def compute_priority(C_H, C_L, sigma):
    """Compute priority of issue H over L using logit choice."""
    return np.exp(C_H / sigma) / (np.exp(C_H / sigma) + np.exp(C_L / sigma))


def make_rho_func(rho):
    """
    Convert rho (scalar, callable, or schedule) into a callable.

    Parameters:
    -----------
    rho : float, callable, or list of (t_break, rho_value) tuples

    Returns:
    --------
    callable : function of time returning rho value
    """
    if callable(rho):
        return rho
    if np.isscalar(rho):
        r = float(rho)
        return lambda t: r
    # schedule: list of (t_break, rho_value)
    try:
        sched = [(float(tb), float(rv)) for tb, rv in rho]
    except Exception as e:
        raise ValueError("rho must be scalar, callable, or list of (t_break, rho_value)") from e
    sched.sort(key=lambda x: x[0])
    breaks = np.array([tb for tb, _ in sched], dtype=float)
    values = np.array([rv for _, rv in sched], dtype=float)
    if breaks[0] > 0.0:
        breaks = np.insert(breaks, 0, 0.0)
        values = np.insert(values, 0, values[0])

    def rho_of_t(t):
        idx = np.searchsorted(breaks, t, side='right') - 1
        idx = max(0, min(idx, len(values) - 1))
        return values[idx]
    return rho_of_t


def make_s_H_func(s_H):
    """
    Convert s_H (scalar, callable, or schedule) into a callable.

    Parameters:
    -----------
    s_H : float, callable, or list of (t_break, s_H_value) tuples

    Returns:
    --------
    callable : function of time returning s_H value
    """
    if callable(s_H):
        return s_H
    if np.isscalar(s_H):
        s = float(s_H)
        return lambda t: s
    # schedule: list of (t_break, s_H_value)
    try:
        sched = [(float(tb), float(sv)) for tb, sv in s_H]
    except Exception as e:
        raise ValueError("s_H must be scalar, callable, or list of (t_break, s_H_value)") from e
    sched.sort(key=lambda x: x[0])
    breaks = np.array([tb for tb, _ in sched], dtype=float)
    values = np.array([sv for _, sv in sched], dtype=float)
    if breaks[0] > 0.0:
        breaks = np.insert(breaks, 0, 0.0)
        values = np.insert(values, 0, values[0])

    def s_H_of_t(t):
        idx = np.searchsorted(breaks, t, side='right') - 1
        idx = max(0, min(idx, len(values) - 1))
        return values[idx]
    return s_H_of_t


def two_group_equilibrium(C_1_0, C_2_0, s, I, alpha, rho, max_iter=500, tol=1e-8):
    """
    Fixed-point iteration for two-group equilibrium on a single issue.
    Returns (C_1, C_2) at equilibrium.
    """
    C_1, C_2 = float(C_1_0), float(C_2_0)

    for _ in range(max_iter):
        next_C_1 = (1 - s) * I + s * f((C_1 + rho * C_2) / (1 + rho), alpha)
        next_C_2 = (1 - s) * I + s * f((C_2 + rho * C_1) / (1 + rho), alpha)

        diff = np.hypot(next_C_1 - C_1, next_C_2 - C_2)
        C_1, C_2 = next_C_1, next_C_2

        if diff < tol:
            break

    return C_1, C_2


def run_two_issue_system(initial_conditions, s_H, s_L, sigma, alpha, rho, I_H_func, I_L_func,
                         t_final=300, n_points=501):
    """
    Run the two-issue (H and L) system with time-varying I and possibly rho.

    Parameters:
    -----------
    initial_conditions : tuple
        (C_1_H_0, C_2_H_0, C_1_L_0, C_2_L_0) initial conditions
    s_H, s_L : float
        Social learning parameters for issue H and L
    sigma : float
        Logit choice sensitivity
    alpha : float
        Conformity exponent
    rho : float, callable, or list of (t_break, rho_val) tuples
        Inter-group connectivity
    I_H_func, I_L_func : callable
        Functions of time returning the objective severity I for issue H and L
    t_final : float
        Final simulation time
    n_points : int
        Number of time points

    Returns:
    --------
    dict with keys: 't', 'C_1_H', 'C_2_H', 'C_1_L', 'C_2_L', 'P_1', 'P_2', 'P'
    """
    C_1_H_0, C_2_H_0, C_1_L_0, C_2_L_0 = initial_conditions
    t_eval = np.linspace(0, t_final, n_points)

    # Handle rho as scalar, callable, or schedule
    def get_rho(t):
        if callable(rho):
            return rho(t)
        elif isinstance(rho, list):
            # Schedule: [(t_break, rho_val), ...]
            current_rho = rho[0][1] if rho else 0
            for t_break, rho_val in rho:
                if t >= t_break:
                    current_rho = rho_val
            return current_rho
        else:
            return rho

    # Euler integration with time-varying parameters
    dt = t_eval[1] - t_eval[0]
    C_1_H_list, C_2_H_list = [C_1_H_0], [C_2_H_0]
    C_1_L_list, C_2_L_list = [C_1_L_0], [C_2_L_0]

    C_1_H, C_2_H = C_1_H_0, C_2_H_0
    C_1_L, C_2_L = C_1_L_0, C_2_L_0

    for t in t_eval[1:]:
        rho_t = get_rho(t)
        I_H = I_H_func(t)
        I_L = I_L_func(t)

        # Issue H dynamics
        dC_1_H = (1 - s_H) * I_H + s_H * f((C_1_H + rho_t * C_2_H) / (1 + rho_t), alpha) - C_1_H
        dC_2_H = (1 - s_H) * I_H + s_H * f((C_2_H + rho_t * C_1_H) / (1 + rho_t), alpha) - C_2_H
        C_1_H += dC_1_H * dt
        C_2_H += dC_2_H * dt

        # Issue L dynamics
        dC_1_L = (1 - s_L) * I_L + s_L * f((C_1_L + rho_t * C_2_L) / (1 + rho_t), alpha) - C_1_L
        dC_2_L = (1 - s_L) * I_L + s_L * f((C_2_L + rho_t * C_1_L) / (1 + rho_t), alpha) - C_2_L
        C_1_L += dC_1_L * dt
        C_2_L += dC_2_L * dt

        C_1_H_list.append(C_1_H)
        C_2_H_list.append(C_2_H)
        C_1_L_list.append(C_1_L)
        C_2_L_list.append(C_2_L)

    C_1_H_arr = np.array(C_1_H_list)
    C_2_H_arr = np.array(C_2_H_list)
    C_1_L_arr = np.array(C_1_L_list)
    C_2_L_arr = np.array(C_2_L_list)

    # Per-group priorities, then average across groups
    P_1 = compute_priority(C_1_H_arr, C_1_L_arr, sigma)
    P_2 = compute_priority(C_2_H_arr, C_2_L_arr, sigma)
    P = (P_1 + P_2) / 2  # Mean of group priorities (P_bar)

    return {
        't': t_eval,
        'C_1_H': C_1_H_arr, 'C_2_H': C_2_H_arr,
        'C_1_L': C_1_L_arr, 'C_2_L': C_2_L_arr,
        'P_1': P_1, 'P_2': P_2, 'P': P
    }


def run_two_issue_system_ivp(initial_conditions, s_H, s_L, sigma, alpha, rho,
                              I_H_func, I_L_func, t_final=300, n_points=501):
    """
    Run the two-issue system using solve_ivp with time-varying I and possibly rho.
    rho can be scalar, callable, or schedule list of (t_break, rho_value).
    """
    C_1_H_0, C_2_H_0, C_1_L_0, C_2_L_0 = initial_conditions
    t_eval = np.linspace(0, t_final, n_points)

    rho_func = make_rho_func(rho)

    def issue_H_sys(t, y):
        C_1, C_2 = y
        I_H = I_H_func(t)
        rho_t = rho_func(t)
        dC_1 = (1 - s_H) * I_H + s_H * f((C_1 + rho_t * C_2) / (1 + rho_t), alpha) - C_1
        dC_2 = (1 - s_H) * I_H + s_H * f((C_2 + rho_t * C_1) / (1 + rho_t), alpha) - C_2
        return [dC_1, dC_2]

    def issue_L_sys(t, y):
        C_1, C_2 = y
        I_L = I_L_func(t)
        rho_t = rho_func(t)
        dC_1 = (1 - s_L) * I_L + s_L * f((C_1 + rho_t * C_2) / (1 + rho_t), alpha) - C_1
        dC_2 = (1 - s_L) * I_L + s_L * f((C_2 + rho_t * C_1) / (1 + rho_t), alpha) - C_2
        return [dC_1, dC_2]

    sol_H = solve_ivp(issue_H_sys, [0, t_final], [C_1_H_0, C_2_H_0], t_eval=t_eval)
    sol_L = solve_ivp(issue_L_sys, [0, t_final], [C_1_L_0, C_2_L_0], t_eval=t_eval)

    t = sol_H.t
    C_1_H, C_2_H = sol_H.y
    C_1_L, C_2_L = sol_L.y

    C_H = 0.5 * (C_1_H + C_2_H)
    C_L = 0.5 * (C_1_L + C_2_L)
    P_1 = np.exp(C_1_H / sigma) / (np.exp(C_1_H / sigma) + np.exp(C_1_L / sigma))
    P_2 = np.exp(C_2_H / sigma) / (np.exp(C_2_H / sigma) + np.exp(C_2_L / sigma))
    P_bar = 0.5 * (P_1 + P_2)
    P = np.exp(C_H / sigma) / (np.exp(C_H / sigma) + np.exp(C_L / sigma))

    I_H = np.array([I_H_func(ti) for ti in t])
    I_L = np.array([I_L_func(ti) for ti in t])
    rho_t = np.array([rho_func(ti) for ti in t])

    return {
        't': t,
        'C_1_H': C_1_H, 'C_2_H': C_2_H, 'C_H': C_H,
        'C_1_L': C_1_L, 'C_2_L': C_2_L, 'C_L': C_L,
        'P_1': P_1, 'P_2': P_2, 'P_bar': P_bar, 'P': P,
        'I_H': I_H, 'I_L': I_L, 'rho_t': rho_t,
    }


def run_two_issue_system_dynamic_s_H(initial_conditions, s_H, s_L, sigma, alpha, rho,
                                      I_H_func, I_L_func, t_final=300, n_points=501):
    """
    Run the two-issue system with time-varying s_H.
    s_H can be scalar, callable, or schedule list of (t_break, s_H_value).
    """
    C_1_H_0, C_2_H_0, C_1_L_0, C_2_L_0 = initial_conditions
    t_eval = np.linspace(0, t_final, n_points)

    s_H_func = make_s_H_func(s_H)
    s_L_func = make_s_H_func(s_L)
    rho_func = make_rho_func(rho)

    def issue_H_sys(t, y):
        C_1, C_2 = y
        I_H = I_H_func(t)
        rho_t = rho_func(t)
        s_H_t = s_H_func(t)
        dC_1 = (1 - s_H_t) * I_H + s_H_t * f((C_1 + rho_t * C_2) / (1 + rho_t), alpha) - C_1
        dC_2 = (1 - s_H_t) * I_H + s_H_t * f((C_2 + rho_t * C_1) / (1 + rho_t), alpha) - C_2
        return [dC_1, dC_2]

    def issue_L_sys(t, y):
        C_1, C_2 = y
        I_L = I_L_func(t)
        rho_t = rho_func(t)
        s_L_t = s_L_func(t)
        dC_1 = (1 - s_L_t) * I_L + s_L_t * f((C_1 + rho_t * C_2) / (1 + rho_t), alpha) - C_1
        dC_2 = (1 - s_L_t) * I_L + s_L_t * f((C_2 + rho_t * C_1) / (1 + rho_t), alpha) - C_2
        return [dC_1, dC_2]

    sol_H = solve_ivp(issue_H_sys, [0, t_final], [C_1_H_0, C_2_H_0], t_eval=t_eval)
    sol_L = solve_ivp(issue_L_sys, [0, t_final], [C_1_L_0, C_2_L_0], t_eval=t_eval)

    t = sol_H.t
    C_1_H, C_2_H = sol_H.y
    C_1_L, C_2_L = sol_L.y

    C_H = 0.5 * (C_1_H + C_2_H)
    C_L = 0.5 * (C_1_L + C_2_L)
    P_1 = np.exp(C_1_H / sigma) / (np.exp(C_1_H / sigma) + np.exp(C_1_L / sigma))
    P_2 = np.exp(C_2_H / sigma) / (np.exp(C_2_H / sigma) + np.exp(C_2_L / sigma))
    P = 0.5 * (P_1 + P_2)

    return {
        't': t_eval,
        'C_1_H': C_1_H, 'C_2_H': C_2_H,
        'C_1_L': C_1_L, 'C_2_L': C_2_L,
        'P_1': P_1, 'P_2': P_2, 'P': P
    }


def compute_priority_heatmap(s_L, sigma, alpha, rho_val,
                              ic=(0.55, 0.45, 0.45, 0.55), n_I=501):
    """Compute priority P over the (s_H, I=I_H=I_L) grid.

    Returns (s_H_values, I_values, heatmap_P). Shared by figureR2 (zoomed
    y-range view) and figureR2S (full y-range view).
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
