# Asymptotic Risk of Learning-Rate Schedules for Least-Squares Regression with Discretely Distributed Features

Code to reproduce the figures and tables of

> Noé Colomban, Francis Bach, Fabian Schaipp.
> *Asymptotic Risk of Learning-Rate Schedules for Least-Squares Regression with Discretely Distributed Features.*

We study the last iterate of SGD on streaming least squares under Slock's model (each sample activates
a single eigendirection), with power-law source and capacity conditions. The code computes

* the **exact finite-time risk** $\mathcal R_{t\mid T} = B_{t\mid T} + V_{t\mid T}$ of SGD for any schedule
  (Theorem 3.1 / Corollary F.4), together with Monte-Carlo SGD simulations;
* the **asymptotic equivalents** $\widetilde B_T$ and $\widetilde V_T$ for the constant and WSD schedules
  (Theorems 4.2 and 4.3), and the asymptotically optimal base learning-rate $\widetilde\gamma^*_T$ (Table 2);
* the extension to a **batch size** $b$ and the critical batch size (Appendix B);
* the exact risk under **Gaussian features**, for comparison (Appendix D).

## Installation

Python ≥ 3.11 is required.

```bash
git clone <URL-of-this-repository>
cd <repository>
pip install -e .            # or: pip install -r requirements.txt
```

Figures use LaTeX for text rendering when a `latex` executable is available
(the `type1ec` and `cm-super` packages are needed). Pass `--no-tex` to `make_figures`
or set `LSQ_NO_TEX=1` to use Matplotlib's built-in renderer instead.

## Reproducing the paper

All commands are run from the repository root.

**1. Draw the figures from the stored results** (a few seconds):

```bash
python -m experiments.make_figures            # all figures -> figures/
python -m experiments.make_figures fig2 fig4  # a subset
```

**2. Recompute the results** (overwrites `results/`). Every script is self-contained and its
parameters are those given in the caption of the corresponding figure:

```bash
python -m experiments.fig2_optimal_learning_rate
```

Each script accepts `--quick` for a smoke test with small horizons: results are then written to
`results/_quick/` (and `make_figures --quick` draws them to `figures/_quick/`) without
touching the results of the paper. All experiments were run on a single CPU.

| Paper | Script (`experiments/`) | Results (`results/`) | Figures (`figures/`) |
|---|---|---|---|
| Fig. 1 — SGD vs. exact risk | `fig1_sgd_vs_exact_risk.py` | `fig1_constant.json`, `fig1_linear.json` | `fig1a_*`, `fig1b_*` |
| Fig. 2 — $\mathcal R_T(\gamma)$ and $\widetilde\gamma^*_T$ | `fig2_optimal_learning_rate.py` | `fig2_constant.json`, `fig2_linear.json` | `fig2a_*`, `fig2b_*` |
| Fig. 3 — LR transfer across cooldown lengths | `fig3_lr_transfer_cooldown.py` | `fig3a_eta_ratio_vs_cooldown.json`, `fig3b_convex_bound.json`, `external/schaipp2025_best_lr.json` | `fig3a_*`, `fig3b_*` |
| Fig. 4 and Table 4 — batch size | `fig4_batch_size.py` | `fig4a_fixed_budget.json`, `fig4b_steps_to_risk.json`, `table4_critical_batch.json` | `fig4a_*`, `fig4b_*` (Table 4 is printed) |
| Figs. 5–6 — Slock vs. Gaussian features | `fig5_6_slock_vs_gaussian.py` | `fig5_*.json`, `fig6_*.json` | `fig5*`, `fig6*` |
| Figs. 7–8 — equivalents vs. exact values | `fig7_8_asymptotic_equivalents.py` | `fig7_equivalents_constant.json`, `fig8_equivalents_wsd.json` | `fig7*`, `fig8*` |
| Fig. 9 — constant-schedule LR reused for WSD | `fig9_lr_transfer_constant_to_wsd.py` | `fig9_constant_vs_wsd.json` | `fig9a_*`, `fig9b_*` |

Notes:

* `results/external/schaipp2025_best_lr.json` contains the empirical optimal learning-rates
  of Schaipp et al. (2025) shown as "Empirical" in Fig. 3b; it is not produced by this code.
* Long trajectories (Figs. 1 and 9) are stored on a thinned grid of steps (log-spaced for Fig. 1,
  every 10 steps for Fig. 9), which is visually indistinguishable from the full trajectory.
* Table 4: `table4_critical_batch.json` stores both the exact critical batch size
  ($\max\{b : b \le \widetilde b_{\max}(N/b)\}$, key `b_crit`) and the first value of the grid
  `np.linspace(1, 10000, 50)` exceeding the stability limit (key `b_crit_grid50`).
* In Fig. 4, beyond the critical batch size the prescribed learning-rate violates the stability
  condition (Assumption B.2) and the recursion diverges; NumPy overflow warnings are expected there.

## Repository structure

```
lsq_schedules/            library
├── models.py             least-squares problems; power-law spectrum; Slock sampling
├── sgd.py                SGD simulation and exact risk recursions (Slock: Eq. (8); Gaussian: Lemma D.1)
├── asymptotics/          asymptotic equivalents and optimal learning-rates
│   ├── constant_asymptotics.py   SlockConstant  (Theorem 4.2)
│   ├── linear_asymptotics.py     SlockLinear    (Theorem 4.3, c = 1; batch size; critical batch)
│   └── wsd_asymptotics.py        SlockWSD       (Theorem 4.3)
├── risk_computations.py  helpers to evaluate several schedules
├── io.py, paths.py       storage of results
└── plotting.py           figure style
scheduled/                learning-rate schedules (vendored, see scheduled/README.md)
experiments/              one script per figure + make_figures.py
results/                  JSON results used in the paper
figures/                  PDF figures of the paper
tests/                    sanity checks (exact recursion, Monte-Carlo check of Theorem 3.1, ...)
```

Conventions: $\lambda_i = L i^{-\alpha}$ and $m_{0,i} = \Delta i^{-\beta}$ with $L = \Delta = 1$; in the code
the spectral exponent $\alpha$ is called `exponent`.

## Tests

```bash
pytest          # or: python -m tests.test_risk
```


