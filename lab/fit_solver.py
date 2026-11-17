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

def _to_model(theta):
    mu1, mu2, log_sd1, log_sd2, z = theta
    sd1, sd2 = math.exp(log_sd1), math.exp(log_sd2)
    rho = math.tanh(z)
    return GaussianParameters((mu1, mu2),
                              ((sd1*sd1, rho*sd1*sd2), (rho*sd1*sd2, sd2*sd2)))

def _initial_theta(rows):
    points = [(tuple(representative(c) for c in row["x"]), row["count"])
              for row in rows]
    total = sum(count for _, count in points)
    mu = [math.fsum(count * point[j] for point, count in points) / total
          for j in range(2)]
    covariance = [[math.fsum(count * (point[i] - mu[i]) * (point[j] - mu[j])
                            for point, count in points) / total
                   for j in range(2)] for i in range(2)]
    try:
        GaussianParameters(mu, covariance)
    except ValueError as exc:
        raise NumericalFailure("noninternal completed-data MLE") from exc
    sd1, sd2 = math.sqrt(covariance[0][0]), math.sqrt(covariance[1][1])
    rho = min(1-1e-15, max(1e-15-1, covariance[0][1]/(sd1*sd2)))
    return [mu[0], mu[1], math.log(sd1), math.log(sd2), math.atanh(rho)], total

def _cholesky_pivots(matrix):
    n = len(matrix)
    factors = [[0.]*n for _ in range(n)]
    pivots = []
    for k in range(n):
        for j in range(k):
            factors[k][j] = (matrix[k][j]-math.fsum(factors[k][m]*factors[j][m]
                                                    for m in range(j)))/factors[j][j]
        pivot = matrix[k][k]-math.fsum(factors[k][m]**2 for m in range(k))
        if not math.isfinite(pivot) or pivot <= 0:
            raise NumericalFailure("objective curvature not positive definite")
        pivots.append(pivot)
        factors[k][k] = math.sqrt(pivot)
    return pivots

def _validate_curvature(objective, theta, engine):
    gradient = engine.gradient(objective, theta)
    hessian = engine.hessian(objective, theta)
    if min(_cholesky_pivots(hessian)) <= .002:
        raise NumericalFailure("ill-conditioned objective curvature")
    columns = []
    for k in range(len(theta)):
        unit = [0.]*len(theta)
        unit[k] = 1.
        columns.append(engine.solve_linear(hessian, unit))
    norm = max(math.fsum(abs(columns[j][i]) for j in range(len(theta)))
               for i in range(len(theta)))
    if not norm <= 100:
        raise NumericalFailure("curvature inverse beyond published support")
    estimate = max(abs(math.fsum(columns[k][i]*gradient[k] for k in range(len(theta))))
                   for i in range(len(theta)))
    return norm, estimate

def solve_joint_mle(rows, engine):
    engine.guard("fit")
    if not rows:
        raise NumericalFailure("empty analysis")
    if len({str(row["x"]) for row in rows}) > 40:
        raise NumericalFailure("at most 40 observation patterns")
    initial, total = _initial_theta(rows)
    def objective(theta):
        try:
            model = _to_model(theta)
        except (OverflowError, ValueError) as exc:
            raise NumericalFailure("parameters outside resolved model range") from exc
        return -weighted_log_likelihood(rows, model, engine)/total
    outcome = engine.minimize(objective, initial)
    if not outcome.converged:
        raise NumericalFailure("optimizer did not converge")
    theta = [float(v) for v in outcome.value]
    norm, estimate = _validate_curvature(objective, theta, engine)
    model = _to_model(theta)
    candidate = FitCandidate(model, weighted_log_likelihood(rows, model, engine),
                             {"method": "joint_likelihood_bfgs", "converged": True,
                              "iterations": outcome.iterations,
                              "parameter_error_estimate": estimate,
                              "hessian_inverse_infinity_norm": norm})
    return validate_fit_candidate(candidate)
