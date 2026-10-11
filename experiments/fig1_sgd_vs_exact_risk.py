"""Figure 1 -- empirical SGD risk vs. exact risk of Theorem 3.1 (Corollary F.4).

Slock's model, gamma = 0.1, L = Delta = 1, alpha = 1.5, beta = 2, sigma = 0.1, d = 100,
averaged over n = 50 runs.
  (a) constant schedule: one run, theory evaluated along it;
  (b) linear decay: one run per horizon T, theory evaluated at t = T.

Output: results/fig1_constant.json, results/fig1_linear.json
"""
from experiments._common import parse_args, make_problem, log_indices
from lsq_schedules.asymptotics import SlockConstant, SlockLinear
from lsq_schedules.io import save_results
from lsq_schedules.sgd import SGD
from scheduled import ConstantSchedule, WSDSchedule

args = parse_args(__doc__)

DIM, SIGMA, ALPHA, BETA, GAMMA = 100, 0.1, 1.5, 2, 0.1
N_RUNS = 5 if args.quick else 50
T_VALUES = [1, 10, 100, 500, 1000] if args.quick else [1, 10, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000]
T_MAX = max(T_VALUES)

model, x0 = make_problem(DIM, SIGMA, ALPHA, BETA)

# (a) Constant schedule ------------------------------------------------------
sgd = SGD(model, x0, ConstantSchedule(steps=T_MAX, base_lr=GAMMA))
losses = sgd.sample_slock(show=False, n_samples=N_RUNS)          # length T_MAX + 1
theory = SlockConstant(model, x0, T_max=T_MAX, base_lr=GAMMA, beta=BETA)
biases, variances = theory.compute_slock_biases_and_variances(T_VALUES)
save_results({
    "sgd": {int(t): float(losses[t - 1]) for t in log_indices(T_MAX)},
    "true": {T: float(biases[T] + variances[T]) for T in T_VALUES},
}, "fig1_constant", args.results_dir)

# (b) Linear decay: a separate run for each horizon T ------------------------
theory = SlockLinear(model, x0, beta=BETA, T_max=T_MAX, base_lr=GAMMA)
biases, variances = theory.compute_slock_biases_and_variances(T_VALUES)
sgd_runs = {}
for T in T_VALUES:
    sgd = SGD(model, x0, WSDSchedule(steps=T, cooldown_len=1, base_lr=GAMMA))
    run = sgd.sample_slock(show=False, n_samples=N_RUNS)
    sgd_runs[T] = {int(t): float(run[t - 1]) for t in log_indices(T)}
save_results({
    "sgd": sgd_runs,
    "true": {T: float(biases[T] + variances[T]) for T in T_VALUES},
}, "fig1_linear", args.results_dir)
