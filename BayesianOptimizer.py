import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
import os
from scipy.stats import norm
from AcquisitionFunctions import ConstrainedEI, SigmoidConstrainedEI
import matplotlib.pyplot as plt
from scipy.optimize import minimize



class BayesianOptimizer:
    def __init__(self, func, constraints, problem ,n_steps=8, init_points=5, m_mc=10, tau=0.5, dim= 2, seed= 0):
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
        self.seed= seed
        self.problem = problem
        
        if self.dim == 1:
            self.grid_n = 200
            self.candidates = np.linspace(0, 1, 101).reshape(-1, 1)
            self.X_grid = np.linspace(0, 1, self.grid_n).reshape(-1, 1)
            self.X_train = np.linspace(0, 1, init_points).reshape(-1, 1)

        elif self.dim == 2:
            self.grid_n = 100
            sampler = qmc.LatinHypercube(d=self.dim, seed=seed)
            self.X_train = sampler.random(n=init_points)

            # regular 2D grid for visualization & candidate set
            x1 = np.linspace(0, 1, self.grid_n)
            x2 = np.linspace(0, 1, self.grid_n)
            self.X_grid = np.array(np.meshgrid(x1, x2)).reshape(2, -1).T
            self.candidates = self.X_grid.copy()

        elif self.dim == 3:
            # use LHS for both train and candidates
            sampler = qmc.LatinHypercube(d=self.dim, seed=seed)
            self.X_train = sampler.random(n=init_points)

            # candidates and grid both via LHS (space-filling 3D design)
            self.grid_n = 2000
            self.X_grid = sampler.random(n=self.grid_n)
            self.candidates = self.X_grid.copy()

        else:
            raise ValueError("This implementation currently supports dim = 1, 2, or 3.")


        # --- evaluate function and constraints on init points ---
        self.y_train = func(self.X_train).ravel()
        self.yc_trains = [con(self.X_train).ravel() for con in constraints]
        
    
    def _optimize_acquisition(self, acq, best_feas, n_restarts=20):
        """
        Continuous optimization of the acquisition function using L-BFGS-B.
        """
        dim = self.dim
        bounds = [(0.0, 1.0)] * dim

        best_x = None
        best_val = -np.inf

        # random restart loop
        for _ in range(n_restarts):

            # random starting point
            x0 = np.random.rand(dim)

            def obj(x):
                return -acq.evaluate_single(x, best_feas)

            res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)

            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_x = res.x

        return best_x, best_val



    # --- GP fit helper ---
    def _fit_gp(self, X, y):
        # FIX: Increased alpha for stability, adjusted length_scale_bounds to prevent overfitting
        kernel = C(1.0, (1e-2, 100)) * RBF(length_scale=0.5, length_scale_bounds=(0.05, 2))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, 
                                      normalize_y=True, random_state=self.seed, 
                                      n_restarts_optimizer=5)
        gp.fit(X, y)
        return gp

    def _visualize(self, gpr_obj, gpr_cons, acq_values, x_next, step, folder="figures"):
        """
        Visualization: 
        - Plot 1: True objective with infeasible regions faded out.
        - Plot 2: GP Objective Mean.
        - Plot 3: GP Probability of Feasibility.
        - Plot 4: Acquisition Function.
        """
        os.makedirs(folder, exist_ok=True)
        fname = os.path.join(folder, f"step_{step+1:02d}.png")

        dim = self.X_train.shape[1]

        if dim == 2:
            n = int(np.sqrt(len(self.X_grid)))
            X1 = self.X_grid[:, 0].reshape(n, n)
            X2 = self.X_grid[:, 1].reshape(n, n)

            # 1. Predict Objective
            mu_obj, _ = gpr_obj.predict(self.X_grid, return_std=True)
            mu_obj = mu_obj.reshape(n, n)

            # 2. Calculate True Feasibility (Boolean Mask)
            # feasible = True where ALL constraints <= 0
            true_constraints_evaluated = np.array([con(self.X_grid) for con in self.constraints])
            true_feasible_mask = np.all(true_constraints_evaluated <= 0, axis=0).reshape(n, n)
            
            # Create INFEASIBLE mask (1 where infeasible, 0 where feasible)
            # We will use this to plot the white overlay
            true_infeasible_mask = (~true_feasible_mask).astype(float)

            # 3. Calculate GP-predicted Probability of Feasibility
            PF_gp_flat = np.ones(len(self.X_grid))
            for gpr in gpr_cons:
                mu_c_gp, std_c_gp = gpr.predict(self.X_grid, return_std=True)
                std_c_gp = np.maximum(std_c_gp, 1e-9)
                PF_gp_i = norm.cdf((0 - mu_c_gp) / std_c_gp)
                PF_gp_flat *= PF_gp_i
            PF_gp_map = PF_gp_flat.reshape(n, n)

            # 4. Evaluate True Objective Function
            true_y = self.func(self.X_grid).reshape(n, n)

            # 5. Acquisition Map
            acq_map = acq_values.reshape(n, n)

            # --- PLOTTING ---
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # =================================================================
            # (1) True function (Infeasible Faded)
            # =================================================================
            # A. Plot the full function first
            im1 = axes[0, 0].contourf(X1, X2, true_y, 30, cmap="viridis")
            
            # B. Overlay White on Infeasible Regions (Alpha controls "fade" amount)
            # levels=[0.5, 1.5] selects the region where mask == 1 (Infeasible)
            axes[0, 0].contourf(X1, X2, true_infeasible_mask, levels=[0.5, 1.5], colors='white', alpha=0.7)
            
            # C. Plot training points
            axes[0, 0].scatter(self.X_train[:, 0], self.X_train[:, 1], color="white", edgecolors='k', s=30)
            axes[0, 0].set_title(f"True f(x) (Feasible Region Highlighted)")
            plt.colorbar(im1, ax=axes[0, 0])


            # =================================================================
            # (2) Objective GP Mean
            # =================================================================
            im2 = axes[0, 1].contourf(X1, X2, mu_obj, 30, cmap="viridis")
            axes[0, 1].scatter(self.X_train[:, 0], self.X_train[:, 1], color="white", edgecolors='k', s=30)
            axes[0, 1].set_title("Objective GP Mean prediction")
            plt.colorbar(im2, ax=axes[0, 1])

            # =================================================================
            # (3) GP Probability of Feasibility (PF)
            # =================================================================
            im3 = axes[1, 0].contourf(X1, X2, PF_gp_map, 30, cmap="Blues", vmin=0, vmax=1)
            axes[1, 0].contour(X1, X2, PF_gp_map, levels=[0.5], colors="black", linewidths=2, linestyles='--') 
            axes[1, 0].scatter(self.X_train[:, 0], self.X_train[:, 1], color="white", edgecolors='k', s=30)
            axes[1, 0].set_title("GP Probability of Feasibility (PF)")
            plt.colorbar(im3, ax=axes[1, 0])

            # =================================================================
            # (4) Acquisition Function
            # =================================================================
            im4 = axes[1, 1].contourf(X1, X2, acq_map, 30, cmap="Greens")
            axes[1, 1].scatter(self.X_train[:, 0], self.X_train[:, 1], color="white", edgecolors='k', s=30)
            if x_next is not None:
                axes[1, 1].scatter(x_next[0], x_next[1], color="red", s=80, marker="x", label="Next", linewidth=2)
            axes[1, 1].set_title(f"Acquisition")
            axes[1, 1].legend()
            plt.colorbar(im4, ax=axes[1, 1])

            plt.tight_layout()
            plt.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"[Saved] 2D Visualization for step {step+1} -> {fname}")
            
    def run(self, visualize= True, acq_type= "cei", scei_params= None):
        progress= []

        for step in range(self.n_steps):
            gp_obj = self._fit_gp(self.X_train, self.y_train)
            gp_cons = [self._fit_gp(self.X_train, yc) for yc in self.yc_trains]

            feas_mask = np.all(np.vstack(self.yc_trains) <= 0, axis=0)
            best_feas = np.max(self.y_train[feas_mask]) if np.any(feas_mask) else -np.inf
            
            acq= None
            if acq_type == "cei":
                acq = ConstrainedEI(gp_obj, gp_cons, tau=self.tau)
                
            elif acq_type == "scei":
                acq = SigmoidConstrainedEI(gp_obj, gp_cons, tau=self.tau, k= scei_params["k"], alpha= scei_params["alpha"])
            else:
                raise ValueError(f"Unknown acq_type {acq_type}")
            
            x_next= None
            #x_next, acq_values = self._optimize_acquisition(acq, best_feas)
            if x_next is None:
                print("L-BFGS-B failed → using grid.")
                acq_values = acq.compute(self.candidates, best_feas)
                x_next = self.candidates[np.argmax(acq_values)]

            
            # Visualization (optional)
            if visualize  and self.dim==2:
                visualize_acq_values = acq.compute(self.candidates, best_feas)
                self._visualize(gp_obj, gp_cons, visualize_acq_values, x_next, step, self.problem+"/experiment_"+acq_type+"_"+str(self.seed))

            y_next = self.func(x_next)
            yc_next = [con(x_next) for con in self.constraints]
            self.X_train = np.vstack([self.X_train, x_next])
            self.y_train = np.append(self.y_train, y_next)
            for k in range(len(self.constraints)):
                self.yc_trains[k] = np.append(self.yc_trains[k], yc_next[k])

            feas_mask = np.all(np.vstack(self.yc_trains) <= 0, axis=0)
            best_feas = np.max(self.y_train[feas_mask]) if np.any(feas_mask) else -np.inf
            progress.append(best_feas)
            progress.append(best_feas)
            print(f"[{acq_type.upper()} step {step+1}] best feasible f(x)={best_feas:.4f}")
        return progress