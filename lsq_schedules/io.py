"""Small JSON helpers used to store and reload experiment results.

Results are stored in ``results/<name>.json`` (see :mod:`lsq_schedules.paths`).
Dictionary keys are written as strings and converted back to ``int``, ``float``
or ``tuple`` when read, so that e.g. ``{(alpha, T): value}`` round-trips.
"""
import ast
import json
import math
from pathlib import Path

from scipy.special import zeta

from lsq_schedules.paths import RESULTS_DIR


def convert_key(key):
    """Convert a JSON string key back to int / float / tuple of numbers if possible."""
    if isinstance(key, (int, float, tuple)):
        return key
    if isinstance(key, str):
        try:
            return int(key)
        except ValueError:
            pass
        try:
            return float(key)
        except ValueError:
            pass
        try:
            parsed = ast.literal_eval(key)
            if isinstance(parsed, tuple) and all(isinstance(x, (int, float)) for x in parsed):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return key


def _stringify_keys(obj):
    if isinstance(obj, dict):
        return {str(k): _stringify_keys(v) for k, v in obj.items()}
    return obj


def results_path(name: str, results_dir: Path | None = None) -> Path:
    """Path of ``results/<name>.json``."""
    results_dir = RESULTS_DIR if results_dir is None else Path(results_dir)
    return results_dir / f"{name}.json"


def save_results(d: dict, name: str, results_dir: Path | None = None) -> Path:
    """Save a (possibly nested) dictionary to ``results/<name>.json``."""
    path = results_path(name, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_stringify_keys(d), f, indent=1, ensure_ascii=False)
    print(f"Saved results to {path}")
    return path


def load_results(name: str, results_dir: Path | None = None) -> dict:
    """Load ``results/<name>.json``; top-level keys are converted with :func:`convert_key`."""
    path = results_path(name, results_dir)
    with open(path) as f:
        d = json.load(f)
    return {convert_key(k): v for k, v in d.items()}


# --------------------------------------------------------------------------
# Kept for backward compatibility with `risk_computations.py` (no-op).
# --------------------------------------------------------------------------
def save_optimization_results(*args, **kwargs):
    pass


def read_optimization_results(*args, **kwargs):
    pass


def constant_zeta_correction(alpha, d=10):
    """Finite-dimension correction factor sum_{i<=d} i^{-alpha/2} / int_1^d x^{-alpha/2} dx."""
    assert alpha > 1, "Alpha should be greater than 1."
    if abs(alpha - 2.0) < 1e-5:
        euler_gamma = 0.5772156649
        return (math.log(d) + euler_gamma) / math.log(d)
    pure_integral = (d ** (1 - alpha / 2) - 1) / (1 - alpha / 2)
    discrete_sum = zeta(alpha / 2) + (d ** (1 - alpha / 2)) / (1 - alpha / 2)
    return discrete_sum / pure_integral if pure_integral != 0 else 1.0
