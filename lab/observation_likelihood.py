import math
from .gaussian_model import (LOG2PI, joint_log_density, conditional_normal,
                             normal_interval_probability, rectangle_probability)
from .numerics import NumericalFailure

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

def _log_allowed_probability(probability):
    if not math.isfinite(probability) or probability <= 0:
        raise NumericalFailure("unresolvable censored observation probability")
    return math.log(probability)

def observation_log_likelihood(row, model, engine):
    x, y = row["x"]
    exact = [constraint["kind"] == "exact" for constraint in (x, y)]
    if all(exact):
        return joint_log_density(model, x["value"], y["value"])
    if any(exact):
        axis = 0 if exact[0] else 1
        known, censored = (x, y) if exact[0] else (y, x)
        variance = model.covariance[axis][axis]
        value = known["value"]
        log_marginal = -.5*(LOG2PI+math.log(variance)+(value-model.mu[axis])**2/variance)
        mean, sd = conditional_normal(model, axis, value)
        lower, upper = bounds(censored)
        mass = normal_interval_probability((lower-mean)/sd, (upper-mean)/sd)
        return log_marginal+_log_allowed_probability(mass)
    probability = rectangle_probability(model, (bounds(x), bounds(y)), engine)
    return _log_allowed_probability(probability)

def weighted_log_likelihood(rows, model, engine):
    return math.fsum(row["count"]*observation_log_likelihood(row, model, engine)
                     for row in rows)
