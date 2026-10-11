"""Figure 3 -- learning-rate transfer across cooldown lengths.

  (a) log(gamma~*_T(1) / gamma~*_T(c)) from the closed form of Table 2 / Eq. (19),
      for alpha in S_alpha (independent of T).
  (b) the same ratio for the optimum sqrt(T1/T2) of the convex bound of
      Schaipp et al. (2025) (G = D = 1, T = 10^4), compared with the empirical
      values stored in results/external/schaipp2025_best_lr.json.

Output: results/fig3a_eta_ratio_vs_cooldown.json, results/fig3b_convex_bound.json
"""
from math import floor

import numpy as np

from experiments._common import parse_args, make_problem, S_ALPHA, DELTA
from lsq_schedules.asymptotics import SlockWSD
from lsq_schedules.io import save_results

args = parse_args(__doc__)

# (a) Our model --------------------------------------------------------------
BETA, SIGMA, T = 2, 0.1, 100  # T is irrelevant: the ratio does not depend on it
c_list = np.linspace(0.01, 1.0, 100)
model, x0 = make_problem(100, SIGMA, S_ALPHA[0], BETA)
wsd = SlockWSD(model, x0, beta=BETA, T_max=T, cooldown_len=1.0)
ratios = {}
for alpha in S_ALPHA:
    wsd._update_model_for_alpha(alpha)
    eta_1 = wsd.compute_best_slock_eta(T=T, m_constant=DELTA, c=1.0)
    ratios[alpha] = {float(c): float(np.log(eta_1 / wsd.compute_best_slock_eta(T=T, m_constant=DELTA, c=c)))
                     for c in c_list}
save_results(ratios, "fig3a_eta_ratio_vs_cooldown", args.results_dir)


# (b) Convex bound of Schaipp et al. (2025) ---------------------------------
def wsd_schedule(T, c):
    eta = np.ones(T)
    T0 = floor((1 - c) * T)
    eta[T0:] = np.linspace(1.0, 0.0, T - T0 + 2)[1:-1]
    return eta


def gamma_star_bound(eta, G=1.0, D=1.0):
    """Minimiser sqrt(T1 / T2) of the last-iterate bound (their Theorem / notation)."""
    G = np.full_like(eta, G)
    s = np.sum(eta)
    T1 = D ** 2 / (2.0 * s)
    w = eta ** 2 * G ** 2
    tail_eta = np.cumsum(eta[::-1])[::-1]        # sum_{t>=k} eta_t
    tail_w = np.cumsum(w[::-1])[::-1]            # sum_{t>=k} eta_t^2 G_t^2
    k = np.arange(len(eta) - 1)
    second = 0.5 * np.sum(eta[k] / tail_eta[k + 1] / tail_eta[k] * tail_w[k])
    T2 = np.sum(w) / (2.0 * s) + second
    return np.sqrt(T1 / T2)


T_BOUND = 1000 if args.quick else 10_000
bound = {float(c): float(gamma_star_bound(wsd_schedule(T_BOUND, c))) for c in np.linspace(0.0, 0.98, 20)}
save_results(bound, "fig3b_convex_bound", args.results_dir)
