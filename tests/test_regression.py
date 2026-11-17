import math
import unittest
from lab.numerics import NumericEngine
from lab.gaussian_model import GaussianParameters
from lab.fit_solver import solve_joint_mle
from lab.joint_tail import joint_exceedance_probability
from lab.observation_likelihood import weighted_log_likelihood
from lab.workflow import Workflow
from samples import exact_rows,mixed_rows,REFERENCE

class TestJointFitRiskRegression(unittest.TestCase):
    def test_fit_maximizes_joint_likelihood_not_imputed_readings(self):
        engine = NumericEngine()
        candidate = solve_joint_mle(mixed_rows(),engine)
        for a,b in zip(candidate.model.mu,REFERENCE["mu"]):
            self.assertAlmostEqual(a,b,delta=1e-6)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(candidate.model.covariance[i][j],
                                       REFERENCE["covariance"][i][j],delta=1e-6)
        self.assertAlmostEqual(candidate.log_likelihood,REFERENCE["log_likelihood"],delta=1e-8)
        # Pre-fix answer: censored constraints imputed as readings at their
        # representative points, then weighted moments. Its true joint
        # likelihood must be strictly worse than the maximizer's.
        imputed = GaussianParameters(
            (0.030555555555555558,0.05555555555555555),
            ((0.5147608024691358,0.25969135802469134),
             (0.25969135802469134,0.4791358024691358)))
        self.assertGreater(candidate.log_likelihood,
                           weighted_log_likelihood(mixed_rows(),imputed,engine))
        self.assertGreater(max(abs(a-b) for a,b in zip(candidate.model.mu,imputed.mu)),1e-3)

    def test_joint_tail_uses_full_covariance_not_marginal_product(self):
        engine = NumericEngine()
        for rho in (0.5,-0.7,0.25):
            model = GaussianParameters((0,0),((1,rho),(rho,1)))
            probability = joint_exceedance_probability(model,[0,0],engine)
            analytic = .25+math.asin(rho)/(2*math.pi)
            self.assertAlmostEqual(probability,analytic,delta=1e-10)
            self.assertGreater(abs(probability-.25),.01)
        with Workflow() as w:
            w.put_batch("batch",exact_rows())
            w.freeze("task",["exact"])
            fit = w.fit("task")
            report = w.risk(fit["result_id"],[0,0])
            self.assertAlmostEqual(report["probability"],1/3,delta=1e-7)
            self.assertGreater(report["probability"],.3)
