import math
from .gaussian_model import joint_log_density

def bounds(constraint):
    kind = constraint["kind"]
    if kind == "left":
        return -math.inf, constraint["upper"]
    if kind == "right":
        return constraint["lower"], math.inf
    if kind == "interval":
        return constraint["lower"], constraint["upper"]
    return constraint["value"], constraint["value"]

def representative(constraint):
    lower, upper = bounds(constraint)
    if not math.isfinite(lower):
        return upper
    if not math.isfinite(upper):
        return lower
    return lower / 2 + upper / 2

def observation_log_likelihood(row, model, engine):
    x, y = (representative(c) for c in row["x"])
    return joint_log_density(model, x, y)

def weighted_log_likelihood(rows, model, engine):
    return math.fsum(row["count"] * observation_log_likelihood(row, model, engine)
                     for row in rows)
