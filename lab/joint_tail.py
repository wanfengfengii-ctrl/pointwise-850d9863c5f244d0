import math
from .numerics import NumericalFailure

def joint_exceedance_probability(model, thresholds, engine):
    engine.guard("risk")
    marginal = []
    for j in range(2):
        sd = math.sqrt(model.covariance[j][j])
        z = (thresholds[j] - model.mu[j]) / sd
        marginal.append(.5 * math.erfc(z / math.sqrt(2)))
    probability = marginal[0] * marginal[1]
    if not math.isfinite(probability):
        raise NumericalFailure("nonfinite joint tail")
    return probability
