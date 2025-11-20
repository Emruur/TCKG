import numpy as np
from BayesianOptimizer import BayesianOptimizer
import os
import json
import matplotlib.pyplot as plt
import cocoex
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")


from scipy.optimize import minimize
import numpy as np
import cocoex

import numpy as np

def find_feasible_maximum_grid(func, cons_fns, dim, num_points_per_dim=200, tolerance=0):
    """
    Finds the feasible maximum using a uniform grid search.
    
    Args:
        func (callable): The objective function (maximization target).
        cons_fns (list): List of constraint functions (g(x) <= 0 is feasible).
        dim (int): Dimensionality of the problem.
        num_points_per_dim (int): Number of grid points along each dimension.
        tolerance (float): Numerical tolerance for the constraint check (g(x) <= tolerance).

    Returns:
        tuple: (best_x_list, best_y_value).
    """
    
    # 1. Generate the Grid
    # Create an array of coordinates for each dimension
    linspaces = [np.linspace(0, 1, num_points_per_dim) for _ in range(dim)]
    
    # Use meshgrid to get all combinations, then reshape to (N_total, dim)
    grid_coords = np.meshgrid(*linspaces, indexing='ij')
    X_grid = np.vstack([x.ravel() for x in grid_coords]).T
    
    # 2. Evaluate Feasibility on the Grid
    feasible_indices = np.arange(X_grid.shape[0])
    
    # Check each constraint function
    for g in cons_fns:
        # g is expected to take a batch (N, D) and return a 1D array (N,)
        g_values = g(X_grid[feasible_indices])
        
        # Identify points that violate the constraint (g(x) > tolerance)
        # We only keep points where g(x) <= tolerance
        is_feasible = g_values <= tolerance
        
        # Filter the indices to only include feasible points remaining
        feasible_indices = feasible_indices[is_feasible]
        
        # Early break if no points are feasible
        if len(feasible_indices) == 0:
            break

    # 3. Evaluate Objective Function for Feasible Points
    
    if len(feasible_indices) == 0:
        print("Warning: Grid search found no feasible points.")
        # Return center of domain and negative infinity for max value
        return np.full(dim, 0.5).tolist(), -np.inf
    
    # Get the subset of feasible points
    X_feasible = X_grid[feasible_indices]
    
    # Evaluate the objective function on the feasible subset
    Y_feasible = func(X_feasible) 
    
    # 4. Find the Maximum
    
    best_index_in_subset = np.argmax(Y_feasible)
    best_x = X_feasible[best_index_in_subset]
    best_y = Y_feasible[best_index_in_subset].item()
    
    print(f"Grid search completed. Found {len(X_feasible)} feasible points out of {X_grid.shape[0]}.")

    return best_x.tolist(), float(best_y)

def find_feasible_maximum(func, cons_fns, dim):
    """Find feasible minimum for a benchmark using constrained optimization."""


    # Objective wrapper
    def obj(x):
        return -func(np.array(x).reshape(1, -1))[0]

    # Convert constraint functions g(x) ≤ 0 to scipy style
    constraints = [{"type": "ineq", "fun": lambda x, g=g: -g(np.array(x).reshape(1, -1))[0]} for g in cons_fns]

    # Try several starting points (since many problems are nonconvex)
    best_val = np.inf
    best_x = None
    for _ in range(50):
        x0 = np.random.rand(dim)
        res = minimize(obj, x0, bounds=[(0, 1)] * dim, constraints=constraints, method="SLSQP", options={"maxiter": 500})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_x = res.x

    print("Constraints: ", )
    return best_x.tolist(), -float(best_val)


class BBOBWrapper:
    """
    Wrapper class to interface the cocoex bbob-constrained problems 
    with the BayesianOptimizer class.

    It dynamically handles initialization, objective, constraints, and scaling
    from [0, 1]^n (BO domain) to the problem's defined search space [L, U]^n.
    """

    def __init__(self, problem_id, instance_id=1, dimension=2, observer=None):
        
        # COCO requires a string definition for the suite
        suite_name = 'bbob-constrained'
        filter_options = (
            f'function_indices:{problem_id}'
            f' instance_indices:{instance_id}'
            f' dimensions:{dimension}'
        )

        suite = cocoex.Suite(suite_name, "", filter_options)
        
        # Get the single problem instance defined by the filters
        self.problem = suite.get_problem(0)
        
        if observer is not None:
            self.problem.observe_with(observer)

        self.dimension = self.problem.dimension
        self.problem_id = problem_id
        self.instance_id = instance_id

        # --- Dynamic Bound Lookup ---
        # The COCO problems define their search space on [problem.lower_bound, problem.upper_bound]^n
        # These are usually -5 and 5, but we retrieve them programmatically.
        self.lower_bound = self.problem.lower_bounds[0]
        self.upper_bound = self.problem.upper_bounds[1]
        
        print(self.problem.lower_bounds)
        print(self.problem.lower_bounds)
        
        self.range_size = self.upper_bound - self.lower_bound
        
        print(f"COCO problem search space bounds: [{self.lower_bound}, {self.upper_bound}]")


    def unscale(self, X_unit):
        """
        Transforms points X from the BO domain [0, 1]^n to the COCO domain [L, U]^n.
        X_unit: numpy array, shape (N, D) or (D,)
        """
        # X = L + X_unit * (U - L)
        X = self.lower_bound + X_unit * self.range_size
        return X

    def get_func(self):
        """Returns the objective function f(x) callable."""
        
        def objective(X_unit):
            X = self.unscale(X_unit)
            
            # The COCO problem requires a 1D array (single point) input.
            
            if X.ndim == 1:
                # Single point input (e.g., from minimize):
                # We return a 1D array of shape (1,) to match the batch style.
                return np.array([-self.problem(X)])
            
            else:
                # Array of points input (e.g., from BO loop):
                # Returns an array of results, shape (N,)
                return np.array([-self.problem(x) for x in X])

        return objective

    def get_constraints(self):
        """Returns a list of constraint functions [c1(x), c2(x), ...] callables."""
        
        num_constraints = self.problem.number_of_constraints
        print("NUM CONSTRAINTS", num_constraints)
        constraint_functions = []

        for k in range(num_constraints):
            
            # Define a function for the k-th constraint
            def constraint_k(X_unit, k=k):
                X = self.unscale(X_unit)

                # The COCO problem returns a vector of all constraint values
                if X.ndim == 1:
                    # single point
                    all_constraints = self.problem.constraint(X)
                    return all_constraints[k]
                else:
                    # array of points (must loop)
                    results = []
                    for x in X:
                        all_constraints = self.problem.constraint(x)
                        results.append(all_constraints[k])
                    return np.array(results)

            constraint_functions.append(constraint_k)

        return constraint_functions
    
def run_experiment(problem, obj,cons,acq_type="cei" ,n_runs=1, visualize= True, dim= 2, scei_params= None):
    
    all_progress = []
    
    init_points= 5
    n_steps= 15
    
    if dim==3:
        init_points= 10
        n_steps= 20

    for run in range(n_runs):
        print(f"\n==============================")
        print(f"  {acq_type.upper()} RUN {run + 1}/{n_runs}")
        print(f"==============================")

        bo = BayesianOptimizer(obj, cons, problem,
                               n_steps=n_steps, init_points=init_points, m_mc=15, tau=0.5, seed=run, dim=dim)
        progress= bo.run(visualize= visualize, acq_type= acq_type, scei_params= scei_params)
        
        all_progress.append(progress)

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


def generate_scei_hparams_list():
    """Generates the 100 combinations of k and alpha to be treated as distinct acquisitions."""
    # Log-spaced k values (0.01 to 100), 10 values
    k_values = np.logspace(-2, 2, 10) 
    
    # Linearly spaced alpha values (0.0 to 1.0), 10 values (excluding 1.0)
    alpha_values = np.linspace(0.0, 1.0, 10, endpoint=False) 

    hparam_list = []
    for k in k_values:
        for alpha in alpha_values:
            hparam_list.append({"k": k, "alpha": alpha})
    return hparam_list

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float_, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.ndarray)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def conduct_comparison_experiment(problem, n_runs=10, dim=2):
    """
    Runs SCEI variants and CEI, saves results, and performs an ablation analysis 
    on k and alpha, including the calculation of the variance of conditional means.
    """
    
    # 1. Set the new output directory
    output_dir = f"{problem}_kablation"
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    
    feasible_x, feasible_y = find_feasible_maximum(obj,cons, dim)
    print(f"Feasible Maximum for {problem}: f(x*) = {feasible_y:.5f} at {feasible_x}")

    # === Define Acquisition List ===
    acq_list = [("cei", {})]
    scei_hparams_list = generate_scei_hparams_list()
    for hparams in scei_hparams_list:
        # Use a consistent naming format for the keys in the results dict
        name = f"scei_k{hparams['k']:.2e}_a{hparams['alpha']:.2f}"
        acq_list.append((name, hparams))
    
    print(f"\n--- Running {len(acq_list)} different acquisition strategies ---")
    
    # Initialize lists for ablation analysis
    all_scei_regrets = []
    all_scei_aucs = []
    
    # Initialize dicts for conditional variance calculation
    k_values_set = np.logspace(-2, 2, 15)
    alpha_values_set = np.linspace(0.0, 1.0, 15, endpoint=False)
    regret_by_k = {k: [] for k in k_values_set}
    auc_by_k = {k: [] for k in k_values_set}
    regret_by_alpha = {a: [] for a in alpha_values_set}
    auc_by_alpha = {a: [] for a in alpha_values_set}

    # === Step 2: Run experiments for all acq types ===
    for i, (acq_name, hparams) in enumerate(acq_list):
        acq_type = "cei" if acq_name == "cei" else "scei"
        
        mean_raw, std_raw, all_runs = run_experiment(
            problem, acq_type=acq_type, n_runs=n_runs, visualize=False, dim=dim, scei_params=hparams
        )
        
        
        # Compute regret: Regret = f(x*) - f(x)
        mean_regret = feasible_y - mean_raw
        
        eps = 1e-12
        mean_regret = np.maximum(mean_regret, eps)
        
        # Compute asymmetric std (not used for the main plot but kept for completeness)
        diffs = np.array(all_runs) - mean_raw[None, :]
        pos_std = np.std(np.clip(diffs, 0, None), axis=0)
        neg_std = np.std(np.clip(-diffs, 0, None), axis=0)
        
        # Compute Area Under the Regret Curve (AUC)
        auc = np.trapz(mean_regret)

        results[acq_name] = {
            "mean_regret": mean_regret.tolist(),
            "std_regret": std_raw.tolist(), 
            "pos_std_regret": pos_std.tolist(),
            "neg_std_regret": neg_std.tolist(),
            "hparams": hparams,
            "auc": auc
        }
        
        if acq_type == "scei":
            # Store final regret and AUC for ablation analysis
            final_regret_value = mean_regret[-1]
            k_val = hparams['k']
            alpha_val = hparams['alpha']
            
            all_scei_regrets.append({'k': k_val, 'alpha': alpha_val, 'value': final_regret_value})
            all_scei_aucs.append({'k': k_val, 'alpha': alpha_val, 'value': auc})
            
            # Populate data structures for conditional variance calculation
            # Find the closest k and alpha in the defined sets due to potential float precision
            closest_k = k_values_set[np.argmin(np.abs(k_values_set - k_val))]
            closest_alpha = alpha_values_set[np.argmin(np.abs(alpha_values_set - alpha_val))]

            regret_by_k[closest_k].append(final_regret_value)
            regret_by_alpha[closest_alpha].append(final_regret_value)
            auc_by_k[closest_k].append(auc)
            auc_by_alpha[closest_alpha].append(auc)
        
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Completed {i+1}/{len(acq_list)}: {acq_name}")


    # === Step 3: Save JSON with all results ===
    results["feasible_max"] = {"x": feasible_x.tolist(), "y": feasible_y}
    with open(os.path.join(output_dir, "results_all.json"), "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    # === Step 4: Identify Best SCEI Variants & Ablation Analysis ===
    
    # Separate results for plotting
    cei_result = results["cei"]
    scei_results = {name: r for name, r in results.items() if name.startswith("scei")}

    # --- 4a. Find Min Regret & Min AUC ---
    final_regrets = {name: r["mean_regret"][-1] for name, r in scei_results.items()}
    min_regret_name = min(final_regrets, key=final_regrets.get)
    min_regret_result = scei_results[min_regret_name]
    
    auc_values = {name: r["auc"] for name, r in scei_results.items()}
    min_auc_name = min(auc_values, key=auc_values.get)
    min_auc_result = scei_results[min_auc_name]
    
    ablation_data = {
        "Optimal_Min_Regret": {
            "k": min_regret_result['hparams']['k'],
            "alpha": min_regret_result['hparams']['alpha'],
            "value": min_regret_result['mean_regret'][-1]
        },
        "Optimal_Min_AUC": {
            "k": min_auc_result['hparams']['k'],
            "alpha": min_auc_result['hparams']['alpha'],
            "value": min_auc_result['auc']
        }
    }
    
    # --- 4b. Single Marginalized Variance (Requested Calculation) ---

    # 1. Calculate the Conditional Means (the 'mean_a' and 'mean_k' lists)
    # The mean performance for a fixed alpha, averaged over all k values.
    mean_a_regrets = [np.mean(v) for a, v in regret_by_alpha.items() if v] # Check v is not empty
    mean_k_regrets = [np.mean(v) for k, v in regret_by_k.items() if v]
    mean_a_aucs = [np.mean(v) for a, v in auc_by_alpha.items() if v]
    mean_k_aucs = [np.mean(v) for k, v in auc_by_k.items() if v]
    
    # 2. Calculate the Objective Variance among these Conditional Means (the requested metric)
    ablation_data["Regret_Variance_Alpha_Marginalized"] = np.var(mean_a_regrets)
    ablation_data["Regret_Variance_K_Marginalized"] = np.var(mean_k_regrets)
    ablation_data["AUC_Variance_Alpha_Marginalized"] = np.var(mean_a_aucs)
    ablation_data["AUC_Variance_K_Marginalized"] = np.var(mean_k_aucs)

    # --- Retain Conditional Data for potential plotting/inspection ---
    # Convert keys to strings for JSON and include both mean and variance for each conditional setting.
    
    ablation_data["Regret_Mean_vs_K_Conditional"] = {str(k): np.mean(v) for k, v in regret_by_k.items() if v}
    ablation_data["Regret_Variance_vs_K_Conditional"] = {str(k): np.var(v) for k, v in regret_by_k.items() if v}

    ablation_data["Regret_Mean_vs_Alpha_Conditional"] = {str(a): np.mean(v) for a, v in regret_by_alpha.items() if v}
    ablation_data["Regret_Variance_vs_Alpha_Conditional"] = {str(a): np.var(v) for a, v in regret_by_alpha.items() if v}

    ablation_data["AUC_Mean_vs_K_Conditional"] = {str(k): np.mean(v) for k, v in auc_by_k.items() if v}
    ablation_data["AUC_Variance_vs_K_Conditional"] = {str(k): np.var(v) for k, v in auc_by_k.items() if v}

    ablation_data["AUC_Mean_vs_Alpha_Conditional"] = {str(a): np.mean(v) for a, v in auc_by_alpha.items() if v}
    ablation_data["AUC_Variance_vs_Alpha_Conditional"] = {str(a): np.var(v) for a, v in auc_by_alpha.items() if v}
    
    # Save ablation analysis
    with open(os.path.join(output_dir, "ablation_analysis.json"), "w") as f:
        json.dump(ablation_data, f, indent=4, cls=NumpyEncoder)


    # --- 4c. Prepare Plot Lines ---
    plot_lines = {
        "CEI_Baseline": {"result": cei_result, "color": 'tab:blue', "style": 'solid', "label": "CEI (Baseline)"},
        min_regret_name: {"result": min_regret_result, "color": 'tab:red', "style": 'dashed', "label": f"SCEI (Min Regret: k={min_regret_result['hparams']['k']:.2e}, α={min_regret_result['hparams']['alpha']:.2f})"},
    }
    if min_auc_name != min_regret_name:
        plot_lines[min_auc_name] = {"result": min_auc_result, "color": 'tab:green', "style": 'dashdot', "label": f"SCEI (Min AUC: k={min_auc_result['hparams']['k']:.2e}, α={min_auc_result['hparams']['alpha']:.2f})"}
    else:
        plot_lines[min_regret_name]["label"] = f"SCEI (Min Regret & AUC: k={min_regret_result['hparams']['k']:.2e}, α={min_regret_result['hparams']['alpha']:.2f})"
        
    steps = np.arange(1, len(cei_result["mean_regret"]) + 1)


    # ==========================================================
    # === Step 5: Plot Regret with Regular (Symmetric) STD ===
    # ==========================================================
    plt.figure(figsize=(10, 6))
    
    # Plot all SCEI regret lines in the background
    for name, r in scei_results.items():
        plt.plot(steps, r["mean_regret"], color='tab:orange', alpha=0.05, linewidth=1)
        
    # Plot the 1-3 highlighted lines
    for line_name, line_data in plot_lines.items():
        r = line_data["result"]
        plt.plot(steps, r["mean_regret"], label=line_data["label"], color=line_data["color"], linestyle=line_data["style"], linewidth=2)
        
        # Plot regular (symmetric) STD
        s = np.array(r["std_regret"])
        plt.fill_between(steps, 
                         np.array(r["mean_regret"]) - s, 
                         np.array(r["mean_regret"]) + s, 
                         alpha=0.2, color=line_data["color"])

    plt.xlabel("Iteration")
    plt.ylabel("Regret = $f(\mathbf{x}^*) - f(\mathbf{x})$")
    plt.title(f"{problem}: Regret Progress (Symmetric Std, Highlighted SCEI Variants)")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{problem}_regret_k_alpha_comparison.png"), dpi=150)
# Helper class to serialize numpy floats to standard JSON floats

    
def conduct_experiment(problem, n_runs=10, dim=2, visualize= True):
    """
    Compares CEI and a fixed-parameter SCEI (k=13, alpha=0.6) performance 
    on a constrained maximization problem, plotting both symmetric and 
    asymmetric standard deviation on the regret curve.
    """
    os.makedirs(problem, exist_ok=True)
    results = {}

    # === Step 1: Identify feasible maximum ===
    # --- Setup Parameters ---
    PROBLEM_ID = 4  # e.g., fsphere

    INSTANCE= 1

    # --- Initialize the COCO Wrapper ---
    coco_wrapper = BBOBWrapper(
        problem_id=PROBLEM_ID,
        dimension=dim,
        instance_id=INSTANCE,  # Use instance 1 for consistency
        # Optionally pass a cocoex.Observer here if you want full COCO benchmarking
    )

    # --- Get the Objective and Constraints from the Wrapper ---
    obj = coco_wrapper.get_func()
    cons = coco_wrapper.get_constraints()
    # TODO am i fckng buddy
    feasible_x, feasible_y = find_feasible_maximum_grid(obj,cons, dim)
    print(f"Feasible Maximum for {problem}: f(x*) = {feasible_y:.5f} at {feasible_x}")

    # === Step 2: Run experiments for all acq types ===
    
    acq_types = ["cei","scei"]
    styles = {"cei": "tab:blue", "scei": "tab:orange"} 
    
    for acq in acq_types:
        scei_params = {
            "k": 15, 
            "alpha": 0.5
        }
        
        # Run experiment
        mean_raw, std_raw, all_runs = run_experiment(problem,obj,cons ,acq_type=acq, n_runs=n_runs, visualize= visualize ,dim=dim, scei_params= scei_params)

        # Compute regret: Regret = f(x*) - f(x) for maximization
        mean_regret = feasible_y - mean_raw
        eps = 1e-12
        mean_regret = np.maximum(mean_regret, eps)
        std_regret = std_raw  # Regular (symmetric) standard deviation of the regret

        # Compute separate positive/negative std for asymmetric uncertainty
        diffs = np.array(all_runs) - mean_raw[None, :]
        pos_std = np.std(np.clip(diffs, 0, None), axis=0)
        neg_std = np.std(np.clip(-diffs, 0, None), axis=0)

        results[acq] = {
            "mean_raw": mean_raw.tolist(), # Kept for JSON record
            "std_raw": std_raw.tolist(),   # Kept for JSON record
            "mean_regret": mean_regret.tolist(),
            "std_regret": std_regret.tolist(),
            "pos_std_regret": pos_std.tolist(),
            "neg_std_regret": neg_std.tolist(),
            "params": scei_params if acq == "scei" else {}
        }

    # === Step 3: Save JSON with feasible maximum ===
    results["feasible_max"] = {"x": feasible_x, "y": feasible_y}
    with open(os.path.join(problem, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    steps = np.arange(1, len(next(iter(results.values()))["mean_raw"]) + 1)


    # ==========================================================
    # === PLOT 1: Regret with Regular (Symmetric) STD ===
    # ==========================================================
    plt.figure(figsize=(8, 5))
    for acq in acq_types:
        style = styles[acq]
        m = np.array(results[acq]["mean_regret"])
        s = np.array(results[acq]["std_regret"]) # Use regular std
        plt.plot(steps, m, label=acq.upper(), color=style)
        plt.fill_between(steps, m - s, m + s, alpha=0.2, color=style)
        
    plt.xlabel("Iteration")
    plt.ylabel("Regret = $f(\mathbf{x}^*) - f(\mathbf{x})$")
    plt.title(f"{problem}: Regret Progress (Symmetric Std)")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(problem, f"{problem}_regret_sym_std.png"), dpi=150)


    # ==========================================================
    # === PLOT 2: Regret with Asymmetric STD ===
    # ==========================================================
    plt.figure(figsize=(8, 5))
    for acq in acq_types:
        style = styles[acq]
        m = np.array(results[acq]["mean_regret"])
        s_pos = np.array(results[acq]["pos_std_regret"])
        s_neg = np.array(results[acq]["neg_std_regret"])
        plt.plot(steps, m, label=acq.upper(), color=style)
        # s_neg gives the lower bound, s_pos gives the upper bound
        plt.fill_between(steps, m - s_neg, m + s_pos, alpha=0.2, color=style)
        
    plt.xlabel("Iteration")
    plt.ylabel("Regret = $f(\mathbf{x}^*) - f(\mathbf{x})$")
    plt.title(f"{problem}: Regret Progress (Asymmetric Std)")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(problem, f"{problem}_regret_asym_std.png"), dpi=150)
    
#conduct_comparison_experiment("branin_wavy", n_runs=5)

conduct_experiment("hhas", n_runs=3, dim= 2)







