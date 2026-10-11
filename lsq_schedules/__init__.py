"""Exact and asymptotic risk of learning-rate schedules for streaming least squares
under Slock's model (discretely distributed features).

Companion code of: N. Colomban, F. Bach, F. Schaipp,
"Asymptotic Risk of Learning-Rate Schedules for Least-Squares Regression with
Discretely Distributed Features".
"""
from lsq_schedules.models import LinearRegression, PowerLawRegression, compute_power_x0
from lsq_schedules.sgd import SGD
from lsq_schedules.asymptotics import SlockConstant, SlockLinear, SlockWSD, Mode

__all__ = [
    "LinearRegression", "PowerLawRegression", "compute_power_x0",
    "SGD", "SlockConstant", "SlockLinear", "SlockWSD", "Mode",
]
