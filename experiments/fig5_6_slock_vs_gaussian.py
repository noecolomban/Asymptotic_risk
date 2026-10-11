"""Figures 5 and 6 -- Slock's model vs. Gaussian features x ~ N(0, H) (Appendix D).

Ratios B_T / B_T^gauss and V_T / V_T^gauss of the exact risks (Theorem 3.1 vs.
Lemma D.1) under the linear decay schedule; d = 100, beta = 2, L = Delta = 1,
sigma = 1, alpha in {1.1, 1.3, ..., 2.5}.
  Figure 5: fixed base learning-rate gamma = 1e-2;
  Figure 6: vanishing base learning-rate gamma = gamma~*_T (Table 2).

Output: results/fig5_slock_vs_gaussian_fixed_lr.json,
        results/fig6_slock_vs_gaussian_optimal_lr.json
"""
from experiments._common import parse_args, make_problem, DELTA
from lsq_schedules.asymptotics import SlockLinear
from lsq_schedules.io import save_results

args = parse_args(__doc__)

DIM, SIGMA, BETA, GAMMA_FIXED = 100, 1.0, 2, 1e-2
ALPHAS = [1.1, 1.5, 2.5] if args.quick else [1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5]
T_VALUES = [100, 500, 1000] if args.quick else [100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 200000]

model, x0 = make_problem(DIM, SIGMA, ALPHAS[0], BETA)
lin = SlockLinear(model, x0, beta=BETA, T_max=max(T_VALUES))

for name, optimal_lr in [("fig5_slock_vs_gaussian_fixed_lr", False),
                         ("fig6_slock_vs_gaussian_optimal_lr", True)]:
    bias_ratio, var_ratio = {}, {}
    for alpha in ALPHAS:
        lin._update_model_for_alpha(alpha)
        bias_ratio[alpha], var_ratio[alpha] = {}, {}
        for T in T_VALUES:
            print(f"[{name}] alpha={alpha}, T={T}")
            gamma = lin.compute_best_slock_eta(T, DELTA) if optimal_lr else GAMMA_FIXED
            lin._setup_for_T(T, base_lr=gamma)
            b_slock, v_slock = lin.compute_slock_biases_and_variances([T])
            b_gauss, v_gauss = lin.compute_true_biases_and_variances([T])  # Gaussian recursion (Lemma D.1)
            bias_ratio[alpha][T] = float(b_slock[T] / b_gauss[T])
            var_ratio[alpha][T] = float(v_slock[T] / v_gauss[T])
    save_results({"bias_ratio": bias_ratio, "variance_ratio": var_ratio}, name, args.results_dir)
