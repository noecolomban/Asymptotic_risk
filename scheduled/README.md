# `scheduled`

Learning-rate schedule classes (constant, WSD, cosine, ...) vendored from the code of

> F. Schaipp, A. Hägele, A. Taylor, U. Şimşekli, F. Bach (2025). *The Surprising Agreement
> Between Convex Optimization Theory and Learning-Rate Scheduling for Large Model Training.* ICML.

Only the schedule definitions are kept (`fit.py` and `utils.py`, which depend on pandas and
are not used here, were removed). In this repository only `ConstantSchedule` and
`WSDSchedule` (with `cooldown_len=1` for the linear decay) are used.
