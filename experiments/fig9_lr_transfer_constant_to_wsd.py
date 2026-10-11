"""Figure 9 -- reusing for WSD (c = 0.2) the base learning-rate tuned for the constant schedule.

T = 1e5, alpha in S_alpha, beta = 2, sigma = 0.1, d = 100, L = Delta = 1,
gamma = gamma~*_T of the constant schedule (Table 2) for both schedules.
The risk trajectories R_{t|T} are stored every `STRIDE` steps.

Output: results/fig9_constant_vs_wsd.json
"""
from experiments._common import parse_args, make_problem, S_ALPHA, DELTA
from lsq_schedules.asymptotics import SlockConstant, SlockWSD
from lsq_schedules.io import save_results

args = parse_args(__doc__)

DIM, SIGMA, BETA, COOLDOWN = 100, 0.1, 2, 0.2
T = 2000 if args.quick else 100_000
STRIDE = 10

model, x0 = make_problem(DIM, SIGMA, S_ALPHA[0], BETA)
constant = SlockConstant(model, x0, T_max=T, beta=BETA)
wsd = SlockWSD(model, x0, beta=BETA, T_max=T, cooldown_len=COOLDOWN)

risks = {"constant": {}, "wsd": {}}
gammas = {}
for alpha in S_ALPHA:
    print(f"alpha={alpha}")
    constant._update_model_for_alpha(alpha)
    gammas[alpha] = float(constant.compute_best_slock_eta(T, m_constant=DELTA))
    constant._setup_for_T(T, base_lr=gammas[alpha])
    wsd._update_model_for_alpha(alpha)
    wsd._setup_for_T(T, base_lr=gammas[alpha])
    for name, analysis in [("constant", constant), ("wsd", wsd)]:
        traj = analysis.sgd.compute_all_slock_risks()
        idx = list(range(0, T, STRIDE)) + [T - 1]
        risks[name][alpha] = {t: float(traj[t]) for t in idx}
save_results({"T": T, "gamma": gammas, **risks}, "fig9_constant_vs_wsd", args.results_dir)
