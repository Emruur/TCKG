import numpy as np
from benchmarks import BENCHMARKS, expand_constraints, find_feasible_maximum
from BayesianOptimizer import BayesianOptimizer
import os
import json
import matplotlib.pyplot as plt

def run_experiment(problem, acq_type="cei" ,n_runs=1, visualize= True, dim= 2, scei_params= None):
    
    obj, cons = BENCHMARKS[problem]
    cons = expand_constraints(cons)
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

def conduct_comparison_experiment(problem, n_runs=10, dim=2):
    
    os.makedirs(problem, exist_ok=True)
    results = {}

    # === Step 1: Identify feasible maximum ===
    feasible_x, feasible_y = find_feasible_maximum(problem, dim)
    print(f"Feasible Maximum for {problem}: f(x*) = {feasible_y:.5f} at {feasible_x}")

    # === Define Acquisition List ===
    # Baseline CEI (no params)
    acq_list = [("cei", {})]
    
    # SCEI variants (100 combinations)
    scei_hparams_list = generate_scei_hparams_list()
    for hparams in scei_hparams_list:
        name = f"scei_k{hparams['k']:.2e}_a{hparams['alpha']:.2f}"
        acq_list.append((name, hparams))
    
    print(f"\n--- Running {len(acq_list)} different acquisition strategies ({len(scei_hparams_list)} SCEI variants) ---")
    
    # === Step 2: Run experiments for all acq types ===
    for i, (acq_name, hparams) in enumerate(acq_list):
        acq_type = "cei" if acq_name == "cei" else "scei" # Identify the core type
        
        # Run experiment (setting visualize=False to avoid excessive plots during run)
        mean_raw, std_raw, all_runs = run_experiment(
            problem, acq_type=acq_type, n_runs=n_runs, visualize=False, dim=dim, scei_params=hparams
        )

        # Compute regret (f(x*) - f(x))
        mean_regret = feasible_y - mean_raw
        
        # Compute separate positive/negative std for asymmetric uncertainty visualization
        diffs = np.array(all_runs) - mean_raw[None, :]
        pos_std = np.std(np.clip(diffs, 0, None), axis=0)
        neg_std = np.std(np.clip(-diffs, 0, None), axis=0)

        results[acq_name] = {
            "mean_raw": mean_raw.tolist(),
            "std_raw": std_raw.tolist(),
            "mean_regret": mean_regret.tolist(),
            "std_regret": std_raw.tolist(), # standard deviation is saved here
            "pos_std_regret": pos_std.tolist(),
            "neg_std_regret": neg_std.tolist(),
            "hparams": hparams # Store the parameters for easy identification
        }
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Completed {i+1}/{len(acq_list)}: {acq_name}")


    # === Step 3: Save JSON with all results ===
    results["feasible_max"] = {"x": feasible_x, "y": feasible_y}
    with open(os.path.join(problem, "results_all.json"), "w") as f:
        json.dump(results, f, indent=4)

    # === Step 4: Plotting (Handling Many Lines) ===
    
    # Note: Plotting 101 lines on one graph can be messy. We'll use color/transparency
    # to highlight CEI and show the range of SCEI performance.

    steps = np.arange(1, len(results["cei"]["mean_raw"]) + 1)

    # Separate results for plotting
    cei_result = results["cei"]
    scei_results = [r for name, r in results.items() if name.startswith("scei")]

    # --- Plot 1: Raw Means (Maximized Objective) ---
    plt.figure(figsize=(10, 6))
    
    # Plot all SCEI lines in light gray/orange
    for r in scei_results:
        plt.plot(steps, r["mean_raw"], color='tab:orange', alpha=0.1, linewidth=1)
        
    # Plot the CEI baseline (dark blue)
    plt.plot(steps, cei_result["mean_raw"], label="CEI (Baseline)", color='tab:blue', linewidth=2)
    plt.fill_between(steps, 
                     np.array(cei_result["mean_raw"]) - np.array(cei_result["std_raw"]), 
                     np.array(cei_result["mean_raw"]) + np.array(cei_result["std_raw"]), 
                     alpha=0.2, color='tab:blue')

    # Optionally plot the best/worst performing SCEI line (based on final mean)
    final_means = np.array([r["mean_raw"][-1] for r in scei_results])
    best_scei_idx = np.argmax(final_means)
    best_scei_result = scei_results[best_scei_idx]
    best_hparams_str = f"k={best_scei_result['hparams']['k']:.2e}, a={best_scei_result['hparams']['alpha']:.2f}"
    plt.plot(steps, best_scei_result["mean_raw"], label=f"SCEI (Best: {best_hparams_str})", color='tab:red', linewidth=1.5)
    
    plt.xlabel("Iteration")
    plt.ylabel("Best feasible f(x)")
    plt.title(f"{problem}: Constrained Optimization (CEI vs. 100 SCEI Variants)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(problem, f"{problem}_raw_all_variants.png"), dpi=150)
    
    
    # --- Plot 2: Regret ---
    plt.figure(figsize=(10, 6))
    
    # Plot all SCEI regret lines
    for r in scei_results:
        plt.plot(steps, r["mean_regret"], color='tab:orange', alpha=0.1, linewidth=1)
        
    # Plot the CEI baseline
    plt.plot(steps, cei_result["mean_regret"], label="CEI (Baseline)", color='tab:blue', linewidth=2)
    
    # Plot the best performing SCEI regret line
    final_regrets = np.array([r["mean_regret"][-1] for r in scei_results])
    best_regret_idx = np.argmin(final_regrets)
    best_scei_regret_result = scei_results[best_regret_idx]
    best_regret_hparams_str = f"k={best_scei_regret_result['hparams']['k']:.2e}, a={best_scei_regret_result['hparams']['alpha']:.2f}"
    plt.plot(steps, best_scei_regret_result["mean_regret"], label=f"SCEI (Min Regret: {best_regret_hparams_str})", color='tab:red', linewidth=1.5)
    
    plt.xlabel("Iteration")
    plt.ylabel("Regret = f(x*) - f(x)")
    plt.title(f"{problem}: Regret Progress (CEI vs. 100 SCEI Variants)")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(problem, f"{problem}_regret_all_variants.png"), dpi=150)

conduct_comparison_experiment("branin_easy_circle", n_runs=5)







