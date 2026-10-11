"""Sanity checks of the exact risk (Theorem 3.1) and of its asymptotic equivalents.

Run with `pytest` (or `python -m tests.test_risk` if pytest is not installed).
"""
import numpy as np

from experiments._common import make_problem
from lsq_schedules.asymptotics import SlockConstant
from lsq_schedules.io import load_results, save_results
from lsq_schedules.sgd import SGD
from scheduled import ConstantSchedule, WSDSchedule


def _exact_slock_risk(lam, m0, sigma, etas):
    """Corollary F.4 written directly: product over steps of 1 - 2 eta lam + eta^2 Tr(Lam) lam."""
    tr = lam.sum()
    m = m0.copy()
    for eta in etas:
        m = (1 - 2 * eta * lam + eta ** 2 * tr * lam) * m + eta ** 2 * sigma ** 2 * lam
    return 0.5 * lam @ m


def test_slock_recursion_matches_corollary_F4():
    np.random.seed(0)
    model, x0 = make_problem(dim=30, sigma=0.3, alpha=1.5, beta=2)
    T, gamma = 200, 0.2
    schedule = WSDSchedule(steps=T, base_lr=gamma, cooldown_len=0.3)
    risks = SGD(model, x0, schedule).compute_all_slock_risks()
    m0 = np.arange(1, 31) ** -2.0
    # compute_all_slock_risks()[t] is the risk after t updates
    expected = _exact_slock_risk(model.Lambda_vals, m0, 0.3, schedule.schedule[: T - 1])
    assert np.isclose(risks[-1], expected, rtol=1e-10)


def test_theorem_3_1_monte_carlo():
    """Average of empirical SGD runs under Slock's model vs. exact risk."""
    np.random.seed(1)
    model, x0 = make_problem(dim=10, sigma=0.5, alpha=1.5, beta=2)
    T, gamma = 300, 0.1
    sgd = SGD(model, x0, ConstantSchedule(steps=T, base_lr=gamma))
    empirical = sgd.sample_slock(show=False, n_samples=400)        # empirical[t]: after t updates
    exact = sgd.compute_all_slock_risks()                           # exact[t]: after t updates
    rel_err = np.abs(empirical[1:T] - exact[1:T]) / exact[1:T]
    assert np.median(rel_err) < 0.1


def test_wsd_schedule_shape():
    T = 1000
    lin = WSDSchedule(steps=T, base_lr=1.0, cooldown_len=1.0).schedule
    cst = ConstantSchedule(steps=T, base_lr=1.0).schedule
    wsd = WSDSchedule(steps=T, base_lr=1.0, cooldown_len=0.2).schedule
    assert np.allclose(cst, 1.0)
    assert np.all(np.diff(lin) < 0) and lin[-1] < 0.01
    assert np.allclose(wsd[: int(0.8 * T)], 1.0) and np.all(np.diff(wsd[int(0.8 * T):]) < 0)


def test_constant_bias_equivalent_theorem_4_2():
    """B~_T / B_T -> 1 for the constant schedule (Theorem 4.2), checked at T = 1e4."""
    np.random.seed(0)
    model, x0 = make_problem(dim=1000, sigma=0.1, alpha=1.8, beta=2)
    T = 10_000
    analysis = SlockConstant(model, x0, T_max=T, base_lr=1e-2, beta=2)
    exact_bias, _ = analysis.compute_slock_biases_and_variances([T])
    approx_bias = analysis.compute_laplace_approx_bias(T, T, m_exponent=2, m_constant=1.0)
    assert abs(approx_bias / exact_bias[T] - 1) < 0.05


def test_results_roundtrip(tmp_path=None):
    import tempfile
    from pathlib import Path
    d = Path(tmp_path or tempfile.mkdtemp())
    save_results({(1.4, 10): 0.5, 2: {"3": 1.0}}, "roundtrip", d)
    loaded = load_results("roundtrip", d)
    assert loaded[(1.4, 10)] == 0.5 and loaded[2] == {"3": 1.0}


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(f"{name}: OK")
