"""Figure 4 and Table 4 -- effect of the batch size b (linear decay schedule).

Parameters: L = Delta = sigma = 1, beta = 2, d = 100, alpha in S_alpha, base
learning-rate gamma~*_{T,b} = b^p gamma~*_T (Eq. (24)).
  (a) excess risk at fixed sample budget N = bT = 2e5, normalised by its value at b = 1;
  (b) number of steps T needed to reach R_T^(b) <= r = 1e-5, and the horizon T at
      which b = b~_max(T);
  Table 4: critical batch size b~_crit(N) = max{ b : b <= b~_max(N / b) } for N = 2e5.

Output: results/fig4a_fixed_budget.json, results/fig4b_steps_to_risk.json,
        results/table4_critical_batch.json
"""
import numpy as np

from experiments._common import parse_args, make_problem, S_ALPHA, DELTA
from lsq_schedules.asymptotics import SlockLinear
from lsq_schedules.io import save_results

args = parse_args(__doc__)

DIM, SIGMA, BETA = 100, 1.0, 2
N_BUDGET = 20_000 if args.quick else 200_000
RISK_TARGET = 1e-5


def batch_exponent(alpha, beta):
    """Exponent p of Eq. (24) for the linear / WSD schedule."""
    return alpha / (alpha + beta) if alpha < 2 else 2 * alpha / (3 * alpha + 2 * beta - 2)


model, x0 = make_problem(DIM, SIGMA, S_ALPHA[0], BETA)
lin = SlockLinear(model, x0, beta=BETA, T_max=1000)


def exact_risk(T, b):
    """Exact last-iterate risk with batch size b, horizon T and gamma = b^p gamma~*_T."""
    eta = b ** batch_exponent(lin.alpha, BETA) * lin.compute_best_slock_eta(T, m_constant=DELTA)
    lin._setup_for_T(T, base_lr=eta)
    bias, var = lin.compute_slock_biases_and_variances([T], batch=b)
    return float(bias[T] + var[T])


# (a) fixed sample budget N = bT ---------------------------------------------
batches = np.unique(np.logspace(0, 3 if args.quick else 4, 20, dtype=int))
risk_fixed_budget = {}
for alpha in S_ALPHA:
    lin._update_model_for_alpha(alpha)
    risk_fixed_budget[alpha] = {}
    for b in batches:
        T = N_BUDGET // b
        print(f"[4a] alpha={alpha}, b={b}, T={T}")
        risk_fixed_budget[alpha][int(b)] = exact_risk(T, int(b))
save_results({"N": N_BUDGET, "risk": risk_fixed_budget}, "fig4a_fixed_budget", args.results_dir)

# (b) steps needed to reach a target risk ------------------------------------
T_MAX = 50_000
batches = np.unique(np.logspace(2.5, 5, 10 if args.quick else 40, dtype=int))
horizons = np.unique(np.logspace(2, 5, 50 if args.quick else 200, dtype=int))
steps, b_max = {}, {}
for alpha in S_ALPHA:
    lin._update_model_for_alpha(alpha)
    steps[alpha] = {}
    for b in batches:
        print(f"[4b] alpha={alpha}, b={b}")
        steps[alpha][int(b)] = T_MAX  # value used when the target is never reached
        for T in horizons:
            if exact_risk(int(T), int(b)) <= RISK_TARGET:
                steps[alpha][int(b)] = int(T)
                break
    # b~_max(T) for the horizons found above (dashed lines of Fig. 4b)
    b_max[alpha] = {int(T): float(lin.compute_exact_critical_batch(T, m_constant=DELTA))
                    for T in sorted(set(steps[alpha].values()))}
save_results({"r": RISK_TARGET, "steps": steps, "b_max": b_max}, "fig4b_steps_to_risk", args.results_dir)

# Table 4: critical batch size for the budget N ------------------------------
# b_crit       : exact value, largest integer b with b <= b~_max(N // b).
# b_crit_grid50: value reported in Table 4 of the paper, i.e. the first point of the
#                coarse grid np.linspace(1, 10000, 50) at which b > b~_max(N // b)
#                (an over-estimate of b_crit by at most one grid step ~ 204).
b_crit, b_crit_grid = {}, {}
for alpha in S_ALPHA:
    lin._update_model_for_alpha(alpha)
    b_max_of = lambda b: lin.compute_exact_critical_batch(N_BUDGET // b, m_constant=DELTA)
    b = 1
    while b + 1 <= b_max_of(b + 1):
        b += 1
    b_crit[alpha] = b
    b_crit_grid[alpha] = next((int(g) for g in np.linspace(1, 10000, dtype=int) if b_max_of(g) < g), None)
    print(f"[Table 4] alpha={alpha}: b_crit = {b} (grid value: {b_crit_grid[alpha]})")
save_results({"N": N_BUDGET, "b_crit": b_crit, "b_crit_grid50": b_crit_grid},
             "table4_critical_batch", args.results_dir)
