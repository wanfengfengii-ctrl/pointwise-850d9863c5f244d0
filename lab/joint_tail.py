import math
from .gaussian_model import rectangle_probability
from .numerics import NumericalFailure

def joint_exceedance_probability(model, thresholds, engine):
    engine.guard("risk")
    sd = math.sqrt(model.covariance[0][0])
    if (thresholds[0]-model.mu[0])/sd >= 12:
        return 0.
    probability = rectangle_probability(
        model, ((thresholds[0], math.inf), (thresholds[1], math.inf)), engine)
    if not math.isfinite(probability):
        raise NumericalFailure("nonfinite joint tail")
    return probability
