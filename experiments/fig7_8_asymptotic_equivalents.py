"""Figures 7 and 8 -- ratio of the asymptotic equivalents to the exact bias / variance.

gamma = 1e-3, L = Delta = 1, beta = 2, sigma = 0.1, d = 1000, alpha in S_alpha.
  Figure 7: constant schedule (Theorem 4.2);
  Figure 8: WSD schedule with c = 0.2 (Theorem 4.3).

Output: results/fig7_equivalents_constant.json, results/fig8_equivalents_wsd.json
"""
from experiments._common import parse_args, make_problem, S_ALPHA, DELTA
from lsq_schedules.asymptotics import SlockConstant, SlockWSD, Mode
from lsq_schedules.io import save_results

args = parse_args(__doc__)

DIM, SIGMA, BETA, GAMMA, COOLDOWN = 1000, 0.1, 2, 1e-3, 0.2
if args.quick:
    DIM = 200
T_VALUES = [10, 100, 1000] if args.quick else [10, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000]

model, x0 = make_problem(DIM, SIGMA, S_ALPHA[0], BETA)
analyses = {
    "fig7_equivalents_constant": SlockConstant(model, x0, T_max=max(T_VALUES), base_lr=GAMMA, beta=BETA),
    "fig8_equivalents_wsd": SlockWSD(model, x0, beta=BETA, T_max=max(T_VALUES), base_lr=GAMMA, cooldown_len=COOLDOWN),
}
for name, analysis in analyses.items():
    approx_var, exact_var, approx_bias, exact_bias = analysis.compare_biases_variances_trajectories_different_alphas(
        T_VALUES, S_ALPHA, m_exponent=BETA, m_constant=DELTA, mode=Mode.SLOCK)
    save_results({
        "approx_bias": approx_bias, "exact_bias": exact_bias,
        "approx_variance": approx_var, "exact_variance": exact_var,
    }, name, args.results_dir)
