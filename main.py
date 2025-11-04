import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import os
from scipy.stats import qmc

from benchmarks import BENCHMARKS, expand_constraints

# -------------------------------------------------------------
# Thresholded Constrained Knowledge Gradient acquisition
# -------------------------------------------------------------
class ThresholdedConstrainedKG:
    """Multi-dimensional, multi-constraint Thresholded Constrained KG."""

    def __init__(self, gp_obj, gp_cons, M=10, sigma_e2_obj=0.0, sigma_e2_cons=None,
                 tau=0.5, rng=None):
        self.gp_obj = gp_obj
        self.gp_cons = gp_cons
        self.M = M
        self.sigma_e2_obj = sigma_e2_obj
        self.sigma_e2_cons = sigma_e2_cons or [0.0] * len(gp_cons)
        self.tau = tau
        self.rng = np.random.RandomState(3) if rng is None else rng

    # --- utilities ---
    def _posterior_mu_std(self, gp, X):
        mu, std = gp.predict(X, return_std=True)
        return mu.ravel(), std.ravel()

    def _tilde_sigma(self, gp, X, xc, sigma_e2):
        X_all = np.vstack([X, xc.reshape(1, -1)])
        _, Cov = gp.predict(X_all, return_cov=True)
        k_xprime_xc = Cov[:-1, -1]
        var_xc = Cov[-1, -1]
        denom = np.sqrt(np.maximum(var_xc + sigma_e2, 1e-12))
        return k_xprime_xc / denom

    def probability_of_feasibility(self, mus_c, stds_c):
        PF_all = [norm.cdf(-mu / (std + 1e-12))
                  for mu, std in zip(mus_c, stds_c)]
        PF = np.prod(np.clip(PF_all, 1e-12, 1.0), axis=0)
        return PF

    # --- acquisition computation ---
    def compute(self, X_grid, candidates):
        print("computing")
        X_grid = np.atleast_2d(X_grid)
        candidates = np.atleast_2d(candidates)

        mu_y, std_y = self._posterior_mu_std(self.gp_obj, X_grid)

        mus_c, stds_c = [], []
        for gp in self.gp_cons:
            mu_c, std_c = self._posterior_mu_std(gp, X_grid)
            mus_c.append(mu_c)
            stds_c.append(std_c)

        PF_n = self.probability_of_feasibility(mus_c, stds_c)
        feas_mask = PF_n >= self.tau
        baseline = np.min(mu_y[feas_mask]) if np.any(feas_mask) else np.inf

        Zy = self.rng.randn(self.M)
        Zc = [self.rng.randn(self.M) for _ in self.gp_cons]
        tckg = np.zeros(len(candidates))


        for i, xc in enumerate(candidates):

            tilde_y = self._tilde_sigma(self.gp_obj, X_grid, xc, self.sigma_e2_obj)
            tilde_cs = [self._tilde_sigma(gp_c, X_grid, xc, sig2)
                        for gp_c, sig2 in zip(self.gp_cons, self.sigma_e2_cons)]
            vals = []

            stds_c_next = [np.sqrt(np.maximum(std_c**2 - tilde_c**2, 1e-12))
                           for std_c, tilde_c in zip(stds_c, tilde_cs)]

            for m in range(self.M):
                mu_y_next = mu_y + tilde_y * Zy[m]
                mus_c_next = [mu_c + tilde_c * Zc_i[m]
                              for mu_c, tilde_c, Zc_i in zip(mus_c, tilde_cs, Zc)]
                PF_next = self.probability_of_feasibility(mus_c_next, stds_c_next)
                feas_mask_next = PF_next >= self.tau

                if np.any(feas_mask_next):
                    vals.append(np.min(mu_y_next[feas_mask_next]))
                else:
                    vals.append(np.inf)

            vals = np.array(vals)
            finite_vals = vals[np.isfinite(vals)]
            if not np.isfinite(baseline) or len(finite_vals) == 0:
                print("[WARN] No feasible baseline found — skipping acquisition update")
                tckg[i] = 0.0  # fallback: no feasible region known yet
            else:
                tckg[i] = baseline - np.mean(finite_vals)
        return tckg


# -------------------------------------------------------------
# -------------------------------------------------------------
# Constrained Expected Improvement (CEI)
# -------------------------------------------------------------
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

        Z = (best_feas - mu_y) / (std_y + 1e-12)
        EI = (best_feas - mu_y) * norm.cdf(Z) + std_y * norm.pdf(Z)
        EI[std_y < 1e-9] = 0.0
        CEI = EI * PF
        return CEI



class BayesianOptimizer:
    def __init__(self, func, constraints, n_steps=8, init_points=5, m_mc=10, tau=0.5, dim= 2, seed= 0):
        """
        func : callable f(X)
        constraints : list of callables [c1(X), c2(X), ...]
        """
        self.func = func
        self.constraints = constraints
        self.n_steps = n_steps
        self.m_mc = m_mc
        self.tau = tau
        self.dim= dim

        # --- define grid and initial points ---
        if self.dim == 1:
            self.grid_n = 200
            self.candidates = np.linspace(0, 1, 101).reshape(-1, 1)
            self.X_grid = np.linspace(0, 1, self.grid_n).reshape(-1, 1)
            self.X_train = np.linspace(0, 1, init_points).reshape(-1, 1)
        else:

            self.grid_n= 25
            sampler = qmc.LatinHypercube(d=self.dim, seed=seed)
            self.X_train = sampler.random(n=init_points)

            # Build a 2D or 3D grid (square grid for first 2 dims)
            x1 = np.linspace(0, 1, self.grid_n)
            x2 = np.linspace(0, 1, self.grid_n)
            self.X_grid = np.array(np.meshgrid(x1, x2)).reshape(2, -1).T
            self.candidates = self.X_grid.copy()

        # --- evaluate function and constraints on init points ---
        self.y_train = func(self.X_train).ravel()
        self.yc_trains = [con(self.X_train).ravel() for con in constraints]
        self.progress = []


    # --- GP fit helper ---
    def _fit_gp(self, X, y):
        kernel = C(1.0, (1e-3, 10)) * RBF(length_scale=0.3,
                                          length_scale_bounds=(1e-2, 10))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10,
                                      normalize_y=True, random_state=0,
                                      n_restarts_optimizer=3)
        gp.fit(X, y)
        return gp

    def _visualize(self, gpr_obj, gpr_cons, acq_values, x_next, step, folder="figures"):
        """
        Save visualization plots for objective, constraint, and acquisition function.
        Works for both 1D and 2D cases.
        """
        os.makedirs(folder, exist_ok=True)
        fname = os.path.join(folder, f"step_{step+1:02d}.png")

        dim = self.X_train.shape[1]

        # -----------------------------
        # 1D visualization (same as before)
        # -----------------------------
        if dim == 1:
            X_plot = np.linspace(0, 1, self.grid_n).reshape(-1, 1)
            mu, std = gpr_obj.predict(X_plot, return_std=True)
            true_y = self.func(X_plot)

            mu_c, std_c = gpr_cons[0].predict(X_plot, return_std=True)
            true_c = self.constraints[0](X_plot)
            feas_mask = true_c <= 0

            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            # (1) True function
            axes[0, 0].plot(X_plot, true_y, "k", label="True f(x)")
            axes[0, 0].fill_between(X_plot.ravel(), np.min(true_y), np.max(true_y),
                                    where=feas_mask.ravel(), color="green", alpha=0.15,
                                    label="Feasible region")
            axes[0, 0].scatter(self.X_train, self.y_train, color="tab:blue", marker="x")
            axes[0, 0].axvline(x_next, color="red", linestyle="--", lw=1)
            axes[0, 0].set_xlim(0, 1)
            axes[0, 0].set_title(f"True f(x) with Feasible Region (Step {step+1})")
            axes[0, 0].legend()

            # (2) Objective GP
            axes[0, 1].plot(X_plot, mu, color="tab:orange", label="GP mean")
            axes[0, 1].fill_between(X_plot.ravel(), mu - 2 * std, mu + 2 * std,
                                    color="tab:orange", alpha=0.2)
            axes[0, 1].scatter(self.X_train, self.y_train, color="k", s=15)
            axes[0, 1].axvline(x_next, color="red", linestyle="--")
            axes[0, 1].set_xlim(0, 1)
            axes[0, 1].set_title("Objective GP Posterior ±2σ")

            # (3) Constraint GP
            axes[1, 0].plot(X_plot, true_c, "k", label="True c(x)")
            axes[1, 0].plot(X_plot, mu_c, color="tab:red", label="GP mean")
            axes[1, 0].fill_between(X_plot.ravel(), mu_c - 2 * std_c, mu_c + 2 * std_c,
                                    color="tab:red", alpha=0.2)
            axes[1, 0].axhline(0, color="gray", lw=1, linestyle="--")
            axes[1, 0].scatter(self.X_train, self.yc_trains[0], color="black", s=15)
            axes[1, 0].axvline(x_next, color="red", linestyle="--")
            axes[1, 0].set_xlim(0, 1)
            axes[1, 0].set_title("Constraint GP Posterior ±2σ")
            axes[1, 0].legend()

            # (4) Acquisition function
            axes[1, 1].plot(self.candidates, acq_values, color="tab:green")
            axes[1, 1].axvline(x_next, color="red", linestyle="--", label="max TCKG")
            axes[1, 1].legend()
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_title("TCKG Acquisition Function")

            plt.tight_layout()
            plt.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"[Saved] Visualization for step {step+1} → {fname}")
            return

        # -----------------------------
        # 2D visualization
        # -----------------------------
        elif dim == 2:
            n = int(np.sqrt(len(self.X_grid)))
            X1 = self.X_grid[:, 0].reshape(n, n)
            X2 = self.X_grid[:, 1].reshape(n, n)

            mu, _ = gpr_obj.predict(self.X_grid, return_std=True)
            mu = mu.reshape(n, n)

            mu_c, _ = gpr_cons[0].predict(self.X_grid, return_std=True)
            mu_c = mu_c.reshape(n, n)

            true_y = self.func(self.X_grid).reshape(n, n)
            true_c = self.constraints[0](self.X_grid).reshape(n, n)

            acq_map = acq_values.reshape(n, n)

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # (1) True function + feasible region
            im1 = axes[0, 0].contourf(X1, X2, true_y, 30, cmap="viridis")
            cs1 = axes[0, 0].contour(X1, X2, true_c, levels=[0], colors="red", linewidths=2)
            axes[0, 0].scatter(self.X_train[:, 0], self.X_train[:, 1], color="white", s=20)
            axes[0, 0].set_title(f"True f(x,y) with Feasibility (Step {step+1})")
            plt.colorbar(im1, ax=axes[0, 0])

            # (2) Objective GP mean
            im2 = axes[0, 1].contourf(X1, X2, mu, 30, cmap="viridis")
            axes[0, 1].scatter(self.X_train[:, 0], self.X_train[:, 1], color="white", s=20)
            axes[0, 1].set_title("Objective GP Mean")
            plt.colorbar(im2, ax=axes[0, 1])

            # (3) Constraint GP mean
            im3 = axes[1, 0].contourf(X1, X2, mu_c, 30, cmap="coolwarm")
            axes[1, 0].contour(X1, X2, mu_c, levels=[0], colors="black", linewidths=2)
            axes[1, 0].scatter(self.X_train[:, 0], self.X_train[:, 1], color="white", s=20)
            axes[1, 0].set_title("Constraint GP Mean")
            plt.colorbar(im3, ax=axes[1, 0])

            # (4) Acquisition function
            im4 = axes[1, 1].contourf(X1, X2, acq_map, 30, cmap="Greens")
            axes[1, 1].scatter(self.X_train[:, 0], self.X_train[:, 1], color="white", s=20)
            axes[1, 1].scatter(x_next[0], x_next[1], color="red", s=50, marker="x", label="Next point")
            axes[1, 1].set_title("TCKG Acquisition Function")
            axes[1, 1].legend()
            plt.colorbar(im4, ax=axes[1, 1])

            plt.tight_layout()
            plt.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"[Saved] 2D Visualization for step {step+1} → {fname}")

    # --- optimization loop ---
    def run(self, visualize=True, folder="figures"):
        """
        Run Bayesian Optimization and record the best *feasible* sample
        (lowest f(x) with all constraints <= 0) at each iteration.
        """
        for step in range(self.n_steps):
            print(f"\n--- Step {step + 1}/{self.n_steps} ---")

            # Fit Gaussian Processes
            gp_obj = self._fit_gp(self.X_train, self.y_train)
            gp_cons = [self._fit_gp(self.X_train, yc) for yc in self.yc_trains]

            # Compute acquisition
            acq = ThresholdedConstrainedKG(gp_obj, gp_cons, M=self.m_mc, tau=self.tau)
            acq_values = acq.compute(self.X_grid, self.candidates)

            # Pick next point
            x_next = self.candidates[np.argmax(acq_values)]
            y_next = self.func(x_next)
            yc_next = [con(x_next) for con in self.constraints]

            print(f"x_next={x_next}, f(x)={float(y_next):.4f}, c(x)={[float(v) for v in yc_next]}")

            # Visualization (optional)
            if visualize:
                self._visualize(gp_obj, gp_cons, acq_values, x_next, step, folder)

            # Update training data
            self.X_train = np.vstack([self.X_train, x_next])
            self.y_train = np.append(self.y_train, y_next)
            for k in range(len(self.constraints)):
                self.yc_trains[k] = np.append(self.yc_trains[k], yc_next[k])

            # === Compute best feasible sample so far ===
            feas_mask = np.all(np.vstack(self.yc_trains) <= 0, axis=0)
            if np.any(feas_mask):
                best_feas = np.min(self.y_train[feas_mask])
            else:
                best_feas = np.inf  # no feasible yet

            self.progress.append(best_feas)

            # === Log ===
            if np.isfinite(self.progress[-1]):
                print(f"[Progress] Best feasible f(x): {self.progress[-1]:.6f}")
            else:
                print("[Progress] No feasible sample yet.")

        print("\nOptimization finished.")

# ==============================================================
# 1. Toy Function and Constraint
# ==============================================================

def f(x):
        x = np.asarray(x, dtype=float)
        return -1*(
            0.6 * np.sin(2 * np.pi * x)
            + 0.3 * np.sin(8 * np.pi * x + 0.3)
            + 0.25 * np.cos(14 * np.pi * (x + 0.2))
            + 0.2 * np.sin(30 * np.pi * x ** 1.2 + 0.5)
            + 0.15 * np.cos(50 * np.pi * (x ** 1.1 + 0.1))
            + 0.45 * np.exp(-60 * (x - 0.22) ** 2)
            - 0.4 * np.exp(-80 * (x - 0.78) ** 2)
            + 0.25 * np.exp(-200 * (x - 0.55) ** 4)
            - 0.15 * np.exp(-150 * (x - 0.9) ** 6)
            + 0.1 * x ** 2 - 0.05 * x ** 3
        )


def c1(x):
        x = np.asarray(x, dtype=float)
        return 0.3 * np.sin(6 * np.pi * (x - 0.2)) + 0.1 * np.cos(3 * np.pi * (x - 0.2)) - 0.05
    

def branin_2d(X):
    """Normalized 2D Branin function in [0,1]^2 domain."""
    X = np.atleast_2d(X)
    x1 = X[:, 0] * 15 - 5      # map [0,1] -> [-5,10]
    x2 = X[:, 1] * 15          # map [0,1] -> [0,15]
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    y = a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s
    return y.reshape(-1, 1)

def constraint_ring(X):
    """Feasible if inside a ring around (0.5,0.5)."""
    X = np.atleast_2d(X)
    return (X[:, 0] - 0.5)**2 + (X[:, 1] - 0.5)**2 - 0.15**2


def run_experiments_with_acq(acq_type="tckg",problem="ackley3_shell" ,n_runs=10, seed_base=31, visualize= True):
    
    obj, cons = BENCHMARKS[problem]
    cons = expand_constraints(cons)
    all_progress = []

    for run in range(n_runs):
        print(f"\n==============================")
        print(f"  {acq_type.upper()} RUN {run + 1}/{n_runs}")
        print(f"==============================")

        bo = BayesianOptimizer(obj, cons,
                               n_steps=15, init_points=5, m_mc=15, tau=0.5, seed=run)

        for step in range(bo.n_steps):
            gp_obj = bo._fit_gp(bo.X_train, bo.y_train)
            gp_cons = [bo._fit_gp(bo.X_train, yc) for yc in bo.yc_trains]

            feas_mask = np.all(np.vstack(bo.yc_trains) <= 0, axis=0)
            best_feas = np.min(bo.y_train[feas_mask]) if np.any(feas_mask) else np.inf

            if acq_type == "tckg":
                acq = ThresholdedConstrainedKG(gp_obj, gp_cons, M=bo.m_mc, tau=bo.tau)
                acq_values = acq.compute(bo.X_grid, bo.candidates)
                x_next = bo.candidates[np.argmax(acq_values)]
            elif acq_type == "cei":
                acq = ConstrainedEI(gp_obj, gp_cons, tau=bo.tau)
                acq_values = acq.compute(bo.candidates, best_feas)
                x_next = bo.candidates[np.argmax(acq_values)]
            elif acq_type == "random":
                x_next = bo.candidates[np.random.randint(len(bo.candidates))]
            else:
                raise ValueError(f"Unknown acq_type {acq_type}")
            
            # Visualization (optional)
            if visualize and acq_type == "tckg":
                bo._visualize(gp_obj, gp_cons, acq_values, x_next, step, problem+"/experiment_"+acq_type+"_"+str(run))

            y_next = bo.func(x_next)
            yc_next = [con(x_next) for con in bo.constraints]
            bo.X_train = np.vstack([bo.X_train, x_next])
            bo.y_train = np.append(bo.y_train, y_next)
            for k in range(len(bo.constraints)):
                bo.yc_trains[k] = np.append(bo.yc_trains[k], yc_next[k])

            feas_mask = np.all(np.vstack(bo.yc_trains) <= 0, axis=0)
            best_feas = np.min(bo.y_train[feas_mask]) if np.any(feas_mask) else np.inf
            bo.progress.append(best_feas)
            print(f"[{acq_type.upper()} step {step+1}] best feasible f(x)={best_feas:.4f}")

        all_progress.append(bo.progress)

    all_progress = np.array(all_progress)

    # Replace inf with NaN first
    all_progress = np.where(np.isinf(all_progress), np.nan, all_progress)

    # --- Compute penalty M (once) from all finite values ---
    finite_vals = all_progress[np.isfinite(all_progress)]
    if finite_vals.size > 0:
        penalty_M = np.max(finite_vals) + 0.1 * np.std(finite_vals)
    else:
        penalty_M = 1e6  # fallback if everything is infeasible
    print(f"[Info] Using penalty_M = {penalty_M:.4f}")

    # --- Replace NaNs with constant penalty M ---
    all_progress = np.where(np.isnan(all_progress), penalty_M, all_progress)

    # --- Regular mean/std ---
    mean_progress = np.mean(all_progress, axis=0)
    std_progress = np.std(all_progress, axis=0)

    return mean_progress, std_progress, all_progress



def conduct_experiment(problem="branin_wavy", n_runs= 10):
    os.makedirs(problem, exist_ok=True)
    problem= "branin_wavy"
    mean_tckg, std_tckg, _ = run_experiments_with_acq("tckg", n_runs=n_runs, problem= problem)
    mean_cei, std_cei, _ = run_experiments_with_acq("cei", n_runs=n_runs, problem= problem)
    mean_rand, std_rand, _ = run_experiments_with_acq("random", n_runs=n_runs, problem= problem)

    steps = np.arange(1, len(mean_tckg) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(steps, mean_tckg, label="TCKG")
    plt.fill_between(steps, mean_tckg - std_tckg, mean_tckg + std_tckg, alpha=0.2)

    plt.plot(steps, mean_cei, label="CEI")
    plt.fill_between(steps, mean_cei - std_cei, mean_cei + std_cei, alpha=0.2)

    plt.plot(steps, mean_rand, label="Random Sampling")
    plt.fill_between(steps, mean_rand - std_rand, mean_rand + std_rand, alpha=0.2)

    plt.xlabel("Iteration")
    plt.ylabel("Best feasible f(x)")
    plt.title("Constrained Optimization Progress Comparison")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(problem+"/cei_vs_tckg_vs_random.png", dpi=150)



if __name__ == "__main__":
    conduct_experiment(problem="branian_wavy", n_runs=2)
    conduct_experiment(problem="sixhump_wedge", n_runs=2)