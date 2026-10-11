"""Draw every figure of the paper from the JSON files stored in ``results/``.

Usage (from the repository root):
    python -m experiments.make_figures            # all figures
    python -m experiments.make_figures fig2 fig9  # a subset
    python -m experiments.make_figures --quick    # use results/_quick/ (smoke test)

Figures are written to ``figures/`` (``figures/_quick/`` with --quick).
"""
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from lsq_schedules.io import convert_key, load_results
from lsq_schedules.paths import FIGURES_DIR, RESULTS_DIR
from lsq_schedules.plotting import ScheduleCmap, finish, grouped_legend, set_style, shade

S_ALPHA_LABEL = r"$\alpha \in \mathcal{S}_\alpha$"
# Font size of legends whose labels list every value of alpha (Figs. 4 to 6).
SMALL_LEGEND = 8

RESULTS = RESULTS_DIR
OUT = FIGURES_DIR


def keys_to_numbers(d):
    """Recursively convert the string keys of a JSON dict to numbers / tuples."""
    if isinstance(d, dict):
        return {convert_key(k): keys_to_numbers(v) for k, v in d.items()}
    return d


def load(name):
    return keys_to_numbers(load_results(name, RESULTS))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved {OUT / name}.pdf")


def alpha_set_label(alphas):
    return r"$\alpha \in \{" + ", ".join(f"{a}" for a in alphas) + r"\}$"


# ---------------------------------------------------------------------------
# Figure 1: empirical SGD loss vs. exact risk
# ---------------------------------------------------------------------------
def fig1():
    # (a) constant schedule
    d = load("fig1_constant")
    fig, ax = plt.subplots()
    t = sorted(d["sgd"])
    ax.plot(t, [d["sgd"][s] for s in t], color=ScheduleCmap.CONSTANT(0.5), lw=2, label="SGD Loss")
    T = sorted(d["true"])
    ax.plot(T, [d["true"][s] for s in T], color=ScheduleCmap.CONSTANT(1.0), lw=1, ls="--", marker=".",
            label=r"Risk $\mathcal{R}_T$")
    finish(ax, "Step", "Loss / Risk", "log", "log")
    ax.legend()
    save(fig, "fig1a_sgd_vs_formula_constant")

    # (b) linear decay: one SGD trajectory per horizon T
    d = load("fig1_linear")
    fig, ax = plt.subplots()
    horizons = sorted(d["sgd"])
    intensities = np.linspace(0.4, 0.8, len(horizons)) ** 2
    for i, H in enumerate(horizons):
        t = sorted(d["sgd"][H])
        ax.plot(t, [d["sgd"][H][s] for s in t], color=ScheduleCmap.LINEAR(intensities[i]), lw=2,
                label="SGD Loss" if H == horizons[-1] else None)
    T = sorted(d["true"])
    ax.plot(T, [d["true"][s] for s in T], color=ScheduleCmap.LINEAR(1.0), lw=1, ls="--", marker=".",
            label=r"Risk $\mathcal{R}_T$")
    finish(ax, "Step", "Loss / Risk", "log", "log")
    ax.legend()
    save(fig, "fig1b_sgd_vs_formula_linear")


# ---------------------------------------------------------------------------
# Figure 2: R_T(gamma) and the asymptotic optimum gamma~*_T
# ---------------------------------------------------------------------------
def _fig2_panel(name, cmap, xlim, ylim):
    d = load(name)
    alphas = sorted(d["risks"])
    fig, ax = plt.subplots()
    solid, dashed = [], []
    for i, a in enumerate(alphas):
        etas = sorted(d["risks"][a])
        (l,) = ax.plot(etas, [d["risks"][a][e] for e in etas], marker=".", color=cmap(shade(i, len(alphas))))
        solid.append(l)
    for i, a in enumerate(alphas):
        dashed.append(ax.axvline(d["eta_star"][a], ls="--", color=cmap(shade(i, len(alphas)))))
    finish(ax, r"$\gamma$", "Risk", "log", "log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    grouped_legend(ax, [solid, dashed],
                   [r"$\mathcal{R}_T, \alpha \in \mathcal{S}_\alpha$", r"$\widetilde{\gamma}^*_T, \alpha \in \mathcal{S}_\alpha$"])
    return fig, ax


def fig2():
    fig, ax = _fig2_panel("fig2_constant", ScheduleCmap.CONSTANT, (1e-3, 1e-2), (2e-5, 6e-5))
    ax.set_xticks([1e-3, 1e-2])
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    save(fig, "fig2a_eta_opt_constant")
    fig, ax = _fig2_panel("fig2_linear", ScheduleCmap.LINEAR, (2e-3, 1e-1), (4e-6, 1e-4))
    save(fig, "fig2b_eta_opt_linear")


# ---------------------------------------------------------------------------
# Figure 3: learning-rate transfer across cooldown lengths
# ---------------------------------------------------------------------------
def fig3():
    # (a) our model
    d = load("fig3a_eta_ratio_vs_cooldown")
    alphas = sorted(d)
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    lines = []
    for i, a in enumerate(alphas):
        c = sorted(d[a])
        (l,) = ax.plot(c, [d[a][x] for x in c], color=ScheduleCmap.WSD(shade(i, len(alphas))))
        lines.append(l)
    finish(ax, r"Cooldown Length ($c$)", r"$\log(\widetilde{\gamma}^*_T(1)/\widetilde{\gamma}^*_T(c))$")
    ax.set_ylim(-0.1, 1.75)
    grouped_legend(ax, [lines], [S_ALPHA_LABEL])
    save(fig, "fig3a_eta_ratio_vs_cooldown")

    # (b) convex bound of Schaipp et al. (2025) vs. their empirical optimum
    bound = load("fig3b_convex_bound")
    emp = keys_to_numbers(load_results("schaipp2025_best_lr", RESULTS_DIR / "external"))
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    c_th = sorted(bound)
    ref = bound[max(c_th)]  # largest cooldown of the grid (c = 0.98), proxy for c = 1
    ax.plot(c_th, [np.log(ref) - np.log(bound[x]) for x in c_th],
            ls="--", color=ScheduleCmap.WSD(0.6), label="Theoretical")
    c_emp = sorted(emp)
    ref_emp = emp[max(c_emp)]  # c = 0.994, proxy for c = 1
    ax.plot(c_emp, [np.log(ref_emp) - np.log(emp[x]) for x in c_emp], marker=".",
            color=ScheduleCmap.WSD(0.9), label="Empirical")
    finish(ax, r"Cooldown Length ($c$)", r"$\log(\gamma^*(1)/\gamma^*(c))$")
    ax.set_ylim(-0.1, 1.75)
    ax.legend()
    save(fig, "fig3b_schaipp_bound")


# ---------------------------------------------------------------------------
# Figure 4: batch size
# ---------------------------------------------------------------------------
def fig4():
    # (a) fixed sample budget
    d = load("fig4a_fixed_budget")
    alphas = sorted(d["risk"])
    fig, ax = plt.subplots()
    lines = []
    for i, a in enumerate(alphas):
        b = sorted(d["risk"][a])
        (l,) = ax.plot(b, [d["risk"][a][x] / d["risk"][a][1] for x in b], marker=".",
                       color=ScheduleCmap.LINEAR(shade(i, len(alphas))))
        lines.append(l)
    finish(ax, r"Batch Size $b$", r"$\mathcal{R}_{\lfloor N/b \rfloor}^{(b)} / \mathcal{R}_{N}^{(1)}$", "log", "linear")
    ax.set_xlim(1, 10000)
    ax.set_ylim(0.995, 1.1)
    grouped_legend(ax, [lines], [alpha_set_label(alphas)], loc="upper left", fontsize=SMALL_LEGEND)
    save(fig, "fig4a_batch_fixed_budget")

    # (b) steps needed to reach a target risk, and b~_max(T)
    d = load("fig4b_steps_to_risk")
    alphas = sorted(d["steps"])
    fig, ax = plt.subplots()
    solid, dashed = [], []
    for i, a in enumerate(alphas):
        color = ScheduleCmap.LINEAR(shade(i, len(alphas)))
        b = sorted(d["steps"][a])
        (l,) = ax.plot(b, [d["steps"][a][x] for x in b], marker=".", color=color)
        solid.append(l)
        T = sorted(d["b_max"][a], key=lambda s: d["b_max"][a][s])
        (l,) = ax.plot([d["b_max"][a][s] for s in T], T, ls="--", lw=1.0, color=color)
        dashed.append(l)
    finish(ax, r"Batch Size $b$", r"$T$", "log", "log")
    ax.set_xlim(500, 100000)
    ax.set_ylim(100, 10000)
    grouped_legend(ax, [solid, dashed],
                   [r"$T$ to reach $\mathcal{R}_T^{(b)} = r$, " + alpha_set_label(alphas),
                    r"$\widetilde{b}_{\mathrm{max}}$, " + alpha_set_label(alphas)],
                   fontsize=SMALL_LEGEND, handleheight=2.0)
    save(fig, "fig4b_steps_to_fixed_risk")


# ---------------------------------------------------------------------------
# Figures 5 and 6: Slock vs. Gaussian features
# ---------------------------------------------------------------------------
def _slock_vs_gaussian(name, out_suffix):
    d = load(name)
    for key, ylabel, ylim, prefix in [
        ("bias_ratio", r"$B_T / B_T^{\mathrm{gauss}}$", (0.9, 1.01), "a_bias"),
        ("variance_ratio", r"$V_T / V_T^{\mathrm{gauss}}$", (0.96, 1.005), "b_variance"),
    ]:
        alphas = sorted(d[key])
        fig, ax = plt.subplots()
        lines = []
        for i, a in enumerate(alphas):
            T = sorted(d[key][a])
            (l,) = ax.plot(T, [d[key][a][s] for s in T], marker=".", color=ScheduleCmap.LINEAR(shade(i, len(alphas))))
            lines.append(l)
        finish(ax, r"$T$", ylabel, "log", "linear")
        ax.set_ylim(*ylim)
        grouped_legend(ax, [lines], [alpha_set_label(alphas)], loc="lower right", handleheight=3,
                       fontsize=SMALL_LEGEND)
        save(fig, f"{out_suffix}{prefix}_ratio_slock_vs_gaussian")


def fig5():
    _slock_vs_gaussian("fig5_slock_vs_gaussian_fixed_lr", "fig5")


def fig6():
    _slock_vs_gaussian("fig6_slock_vs_gaussian_optimal_lr", "fig6")


# ---------------------------------------------------------------------------
# Figures 7 and 8: asymptotic equivalents / exact values
# ---------------------------------------------------------------------------
def _equivalents(name, cmap, prefix):
    d = load(name)
    for part, ylabel, suffix in [("bias", r"$\widetilde{B}_T / B_T$", "a_bias"),
                                 ("variance", r"$\widetilde{V}_T / V_T$", "b_variance")]:
        approx, exact = d[f"approx_{part}"], d[f"exact_{part}"]
        alphas = sorted({a for a, _ in exact})
        T_values = sorted({T for _, T in exact})
        fig, ax = plt.subplots()
        lines = []
        for i, a in enumerate(alphas):
            (l,) = ax.plot(T_values, [approx[(a, T)] / exact[(a, T)] for T in T_values], marker=".",
                           color=cmap(shade(i, len(alphas))))
            lines.append(l)
        finish(ax, r"$T$ (log scale)", ylabel, "log", "log")
        grouped_legend(ax, [lines], [S_ALPHA_LABEL])
        save(fig, f"{prefix}{suffix}_ratio")


def fig7():
    _equivalents("fig7_equivalents_constant", ScheduleCmap.CONSTANT, "fig7")


def fig8():
    _equivalents("fig8_equivalents_wsd", ScheduleCmap.WSD, "fig8")


# ---------------------------------------------------------------------------
# Figure 9: WSD with the learning-rate tuned for the constant schedule
# ---------------------------------------------------------------------------
def fig9():
    d = load("fig9_constant_vs_wsd")
    alphas = sorted(d["constant"])
    t = sorted(d["constant"][alphas[0]])

    fig, ax = plt.subplots()
    cst, wsd = [], []
    for i, a in enumerate(alphas):
        (l,) = ax.plot(t, [d["constant"][a][s] for s in t], color=ScheduleCmap.CONSTANT(shade(i, len(alphas))))
        cst.append(l)
        (l,) = ax.plot(t, [d["wsd"][a][s] for s in t], color=ScheduleCmap.WSD(shade(i, len(alphas))))
        wsd.append(l)
    finish(ax, r"Step $t$", r"$\mathcal{R}_{t \mid T}$", "linear", "log")
    grouped_legend(ax, [cst, wsd], ["Constant", "WSD"])
    save(fig, "fig9a_constant_vs_wsd")

    fig, ax = plt.subplots()
    lines = []
    for i, a in enumerate(alphas):
        (l,) = ax.plot(t, [d["wsd"][a][s] / d["constant"][a][s] for s in t],
                       color=ScheduleCmap.MULTI(shade(i, len(alphas))))
        lines.append(l)
    finish(ax, r"Step $t$", r"$\mathcal{R}_{t\mid T}^{\mathrm{wsd}} / \mathcal{R}_{t\mid T}^{\mathrm{constant}}$")
    grouped_legend(ax, [lines], [S_ALPHA_LABEL], loc="upper left")
    save(fig, "fig9b_ratio_wsd_constant")


# ---------------------------------------------------------------------------
def table4():
    d = load("table4_critical_batch")
    print(f"\nTable 4 (N = {d['N']}):")
    print(f"  {'alpha':>6} {'b_crit (exact)':>15} {'grid value (paper)':>19}")
    for a in sorted(d["b_crit"]):
        print(f"  {a:>6} {d['b_crit'][a]:>15} {d['b_crit_grid50'][a]:>19}")


FIGURES = {f.__name__: f for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, table4]}


def main():
    global RESULTS, OUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figures", nargs="*", choices=[[]] + list(FIGURES), help="Subset of figures to draw.")
    parser.add_argument("--quick", action="store_true", help="Read results/_quick/ and write figures/_quick/.")
    parser.add_argument("--no-tex", action="store_true", help="Do not use LaTeX to render text.")
    args = parser.parse_args()
    if args.quick:
        RESULTS, OUT = RESULTS_DIR / "_quick", FIGURES_DIR / "_quick"
    set_style(usetex=False if args.no_tex else None)
    for name in args.figures or FIGURES:
        FIGURES[name]()


if __name__ == "__main__":
    main()
