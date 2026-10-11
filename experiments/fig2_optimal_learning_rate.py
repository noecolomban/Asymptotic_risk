"""Figure 2 -- exact risk R_T(gamma) vs. base learning-rate, and the asymptotic optimum.

T = 1e5, beta = 2, d = 100, sigma = 0.1, L = Delta = 1, alpha in S_alpha.
  (a) constant schedule, (b) linear decay (c = 1).
The vertical lines are the closed-form optima gamma~*_T of Table 2.

Output: results/fig2_constant.json, results/fig2_linear.json
"""
import numpy as np

from experiments._common import parse_args, make_problem, S_ALPHA, DELTA
from lsq_schedules.asymptotics import SlockConstant, SlockLinear
from lsq_schedules.io import save_results

args = parse_args(__doc__)

DIM, SIGMA, BETA = 100, 0.1, 2
T = 2000 if args.quick else 100_000
GRIDS = {  # grid of base learning-rates for each schedule
    "constant": np.logspace(-4, -1, 8 if args.quick else 50),
    "linear": np.logspace(-3, -1, 5 if args.quick else 10),
}
model, x0 = make_problem(DIM, SIGMA, S_ALPHA[0], BETA)
analyses = {
    "constant": SlockConstant(model, x0, T_max=T, beta=BETA),
    "linear": SlockLinear(model, x0, beta=BETA, T_max=T),
}

for name, analysis in analyses.items():
    risks, eta_star = {}, {}
    for alpha in S_ALPHA:
        analysis._update_model_for_alpha(alpha)
        eta_star[alpha] = float(analysis.compute_best_slock_eta(T, m_constant=DELTA))
        risks[alpha] = {}
        for gamma in GRIDS[name]:
            print(f"[{name}] alpha={alpha}, gamma={gamma:.2e}")
            analysis._setup_for_T(T, base_lr=gamma)
            b, v = analysis.compute_slock_biases_and_variances([T])
            risks[alpha][float(gamma)] = float(b[T] + v[T])
    save_results({"T": T, "risks": risks, "eta_star": eta_star}, f"fig2_{name}", args.results_dir)
