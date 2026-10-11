"""Utilities shared by the experiment scripts."""
import argparse

import numpy as np

from lsq_schedules.models import PowerLawRegression, compute_power_x0
from lsq_schedules.paths import RESULTS_DIR

# Set S_alpha of spectral exponents used throughout the paper.
S_ALPHA = [1.4, 1.8, 2.2, 2.6]
# L = Delta = 1 in all experiments (lambda_i = i^-alpha, m_{0,i} = i^-beta).
DELTA = 1.0


def parse_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test with small horizons; results go to results/_quick/ "
                             "and do not overwrite the results of the paper.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    args = parser.parse_args()
    args.results_dir = RESULTS_DIR / "_quick" if args.quick else RESULTS_DIR
    np.random.seed(args.seed)
    return args


def make_problem(dim: int, sigma: float, alpha: float, beta: float):
    """Power-law problem of Assumption 3.2 with L = Delta = 1.

    Returns the model (lambda_i = i^-alpha, noise std sigma) and an initial point
    x0 such that m_{0,i} = (x0 - x*)_i^2 = i^-beta in the eigenbasis.
    """
    model = PowerLawRegression(dim=dim, sigma=sigma, exponent=alpha)
    # compute_power_x0 sets delta0_i = i^{-b}; b = beta/2 gives m_{0,i} = i^{-beta}.
    x0 = compute_power_x0(dim, model.x_star.flatten(), model.Q, beta=beta / 2)
    return model, x0


def log_indices(n: int, num: int = 2000) -> np.ndarray:
    """~``num`` log-spaced integer indices in [1, n] (used to thin long trajectories)."""
    return np.unique(np.geomspace(1, n, num).astype(int))
