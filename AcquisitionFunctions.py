import numpy as np
from scipy.stats import norm




class ConstrainedEI:
    """Expected Improvement with probabilistic constraints (CEI)."""

    def __init__(self, gp_obj, gp_cons, tau=0.5):
        self.gp_obj = gp_obj
        self.gp_cons = gp_cons
        self.tau = tau

    def _posterior_mu_std(self, gp, X):
        mu, std = gp.predict(X, return_std=True)
        return mu.ravel(), std.ravel()

    def probability_of_feasibility(self, mus_c, stds_c):
        PF_all = [norm.cdf(-mu / (std + 1e-12))
                  for mu, std in zip(mus_c, stds_c)]
        PF = np.prod(np.clip(PF_all, 1e-12, 1.0), axis=0)
        return PF

    def compute(self, X, best_feas):
        mu_y, std_y = self._posterior_mu_std(self.gp_obj, X)
        mus_c, stds_c = [], []
        for gp in self.gp_cons:
            mu_c, std_c = self._posterior_mu_std(gp, X)
            mus_c.append(mu_c)
            stds_c.append(std_c)

        PF = self.probability_of_feasibility(mus_c, stds_c)
        
        improvement= mu_y - best_feas
        Z =  improvement/ (std_y + 1e-12)
        EI = improvement * norm.cdf(Z) + std_y * norm.pdf(Z)
        EI[std_y < 1e-9] = 0.0
        CEI = EI * PF
        return CEI