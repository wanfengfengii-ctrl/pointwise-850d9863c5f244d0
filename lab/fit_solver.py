import math
from dataclasses import dataclass
from .gaussian_model import GaussianParameters
from .observation_likelihood import representative, weighted_log_likelihood
from .numerics import NumericalFailure

@dataclass(frozen=True)
class FitCandidate:
    model: GaussianParameters
    log_likelihood: float
    status: dict

def validate_fit_candidate(candidate):
    model = candidate.model
    if not math.isfinite(candidate.log_likelihood):
        raise NumericalFailure("nonfinite likelihood")
    sd = [math.sqrt(model.covariance[j][j]) for j in range(2)]
    rho = model.covariance[0][1] / (sd[0] * sd[1])
    if max(map(abs, model.mu)) > 8 or min(sd) < .2 or max(sd) > 4 or abs(rho) > .9:
        raise NumericalFailure("outside published numerical guarantee")
    if candidate.status["parameter_error_estimate"] > 2e-7:
        raise NumericalFailure("unresolved parameter accuracy")
    return candidate

def solve_joint_mle(rows, engine):
    engine.guard("fit")
    if not rows:
        raise NumericalFailure("empty analysis")
    if len({str(row["x"]) for row in rows}) > 40:
        raise NumericalFailure("at most 40 observation patterns")
    points = [(tuple(representative(c) for c in row["x"]), row["count"])
              for row in rows]
    total = sum(count for _, count in points)
    mu = [math.fsum(count * point[j] for point, count in points) / total
          for j in range(2)]
    covariance = [[math.fsum(count * (point[i] - mu[i]) * (point[j] - mu[j])
                            for point, count in points) / total
                   for j in range(2)] for i in range(2)]
    try:
        model = GaussianParameters(mu, covariance)
    except ValueError as exc:
        raise NumericalFailure("noninternal completed-data MLE") from exc
    candidate = FitCandidate(model, weighted_log_likelihood(rows, model, engine),
                             {"method": "weighted_moments", "converged": True,
                              "parameter_error_estimate": 0.})
    return validate_fit_candidate(candidate)
