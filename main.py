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
        
        # Compute Area Under the Curve (AUC) for the REGRET curve
        # Lower AUC means faster convergence / better overall performance.
        # We will track the MIN AUC (Max AUC of performance is Min AUC of Regret).
        # Use simple trapezoidal rule (np.trapz)
        auc = np.trapz(mean_regret)

        results[acq_name] = {
            "mean_raw": mean_raw.tolist(),
            "std_raw": std_raw.tolist(),
            "mean_regret": mean_regret.tolist(),
            "std_regret": std_raw.tolist(), 
            "pos_std_regret": pos_std.tolist(),
            "neg_std_regret": neg_std.tolist(),
            "hparams": hparams,
            "auc": auc
        }
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Completed {i+1}/{len(acq_list)}: {acq_name}")


    # === Step 3: Save JSON with all results ===
    results["feasible_max"] = {"x": feasible_x, "y": feasible_y}
    with open(os.path.join(problem, "results_all.json"), "w") as f:
        json.dump(results, f, indent=4)

    # === Step 4: Identify Best SCEI Variants ===
    
    # Separate results for plotting
    cei_result = results["cei"]
    scei_results = {name: r for name, r in results.items() if name.startswith("scei")}

    # Find MINIMUM REGRET (best final solution)
    final_regrets = {name: r["mean_regret"][-1] for name, r in scei_results.items()}
    min_regret_name = min(final_regrets, key=final_regrets.get)
    min_regret_result = scei_results[min_regret_name]
    
    # Find MINIMUM AUC (best overall speed/accumulated performance)
    auc_values = {name: r["auc"] for name, r in scei_results.items()}
    min_auc_name = min(auc_values, key=auc_values.get)
    min_auc_result = scei_results[min_auc_name]
    
    # Ensure min_regret and min_auc are plotted, even if they are the same variant
    # We create a dictionary to hold the three lines to plot
    plot_lines = {
        "CEI_Baseline": {"result": cei_result, "color": 'tab:blue', "style": 'solid', "label": "CEI (Baseline)"},
        min_regret_name: {"result": min_regret_result, "color": 'tab:red', "style": 'dashed', "label": f"SCEI (Min Regret: k={min_regret_result['hparams']['k']:.2e}, a={min_regret_result['hparams']['alpha']:.2f})"},
    }
    # Add the Min AUC line if it's different from Min Regret
    if min_auc_name != min_regret_name:
        plot_lines[min_auc_name] = {"result": min_auc_result, "color": 'tab:green', "style": 'dashdot', "label": f"SCEI (Min AUC: k={min_auc_result['hparams']['k']:.2e}, a={min_auc_result['hparams']['alpha']:.2f})"}
    else:
        # If they are the same, modify the Min Regret label to reflect both achievements
        plot_lines[min_regret_name]["label"] = f"SCEI (Min Regret & AUC: k={min_regret_result['hparams']['k']:.2e}, a={min_regret_result['hparams']['alpha']:.2f})"
        
    steps = np.arange(1, len(cei_result["mean_raw"]) + 1)


    # === Step 5: Plot regret with asymmetric stds ===
    plt.figure(figsize=(10, 6))
    
    # Plot all SCEI regret lines in the background
    for name, r in scei_results.items():
        plt.plot(steps, r["mean_regret"], color='tab:orange', alpha=0.05, linewidth=1)
        
    # Plot the 1-3 highlighted lines
    for line_name, line_data in plot_lines.items():
        r = line_data["result"]
        plt.plot(steps, r["mean_regret"], label=line_data["label"], color=line_data["color"], linestyle=line_data["style"], linewidth=2)
        # Only plot std for CEI and the best variants, using asymmetric std
        s_pos = np.array(r["pos_std_regret"])
        s_neg = np.array(r["neg_std_regret"])
        plt.fill_between(steps, 
                         np.array(r["mean_regret"]) - s_neg, 
                         np.array(r["mean_regret"]) + s_pos, 
                         alpha=0.2, color=line_data["color"])

    plt.xlabel("Iteration")
    plt.ylabel("Regret = f(x*) - f(x)")
    plt.title(f"{problem}: Regret Progress (Highlighted SCEI Variants)")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(problem, f"{problem}_regret_k_alpha_comparison.png"), dpi=150)
    
    
    # --- Optional: Plot 2 - Raw Means (Highlighted Lines) ---
    # This plot will show the absolute objective value progress.
    plt.figure(figsize=(10, 6))
    
    # Plot all SCEI raw lines in the background
    for name, r in scei_results.items():
        plt.plot(steps, r["mean_raw"], color='tab:orange', alpha=0.05, linewidth=1)
        
    # Plot the 1-3 highlighted lines
    for line_name, line_data in plot_lines.items():
        r = line_data["result"]
        plt.plot(steps, r["mean_raw"], label=line_data["label"], color=line_data["color"], linestyle=line_data["style"], linewidth=2)
        # Plot full symmetric std for raw values
        s = np.array(r["std_raw"])
        plt.fill_between(steps, 
                         np.array(r["mean_raw"]) - s, 
                         np.array(r["mean_raw"]) + s, 
                         alpha=0.2, color=line_data["color"])
        
    plt.xlabel("Iteration")
    plt.ylabel("Best feasible f(x)")
    plt.title(f"{problem}: Raw Mean Objective (Highlighted SCEI Variants)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(problem, f"{problem}_raw_k_alpha_comparison.png"), dpi=150)

def conduct_experiment(problem, n_runs=10, dim=2):
    os.makedirs(problem, exist_ok=True)
    results = {}

    # === Step 1: Identify feasible minimum ===
    # NOTE: This uses the function name 'find_feasible_maximum' based on your previous correction.
    feasible_x, feasible_y = find_feasible_maximum(problem, dim)
    print(f"Feasible minimum for {problem}: f(x*) = {feasible_y:.5f} at {feasible_x}")

    # === Step 2: Run experiments for all acq types ===
    
    acq_types = ["cei","scei"]
    for acq in acq_types:
        scei_params= {
            "k": 1, 
            "alpha": 0.5 
            # The following lines defining the parameter search space were comments in the original code,
            # but are critical for understanding the intent:
            # [0.01 -> 100] try 10 values
            # 0.5 -> 0-1 try 10 values
        }
        # 100 combinations to try
        
        # NOTE: In the original code, `scei_params` is defined as a fixed dictionary inside the loop, 
        # but the comments indicate an intention to iterate over 100 combinations.
        # Assuming run_experiment uses these fixed values for "scei" in this original code structure.
        mean_raw, std_raw, all_runs = run_experiment(problem ,acq_type=acq, n_runs=n_runs, visualize= True ,dim=dim, scei_params= scei_params)

        # Compute regret (raw - feasible min)
        mean_regret = feasible_y - mean_raw
        std_regret = std_raw  # same std applies numerically to regret

        # Compute separate positive/negative std for asymmetric uncertainty visualization
        diffs = np.array(all_runs) - mean_raw[None, :]
        pos_std = np.std(np.clip(diffs, 0, None), axis=0)
        neg_std = np.std(np.clip(-diffs, 0, None), axis=0)

        results[acq] = {
            "mean_raw": mean_raw.tolist(),
            "std_raw": std_raw.tolist(),
            "mean_regret": mean_regret.tolist(),
            "std_regret": std_regret.tolist(),
            "pos_std_regret": pos_std.tolist(),
            "neg_std_regret": neg_std.tolist()
        }

    # === Step 3: Save JSON with feasible minimum ===
    # NOTE: Renamed 'feasible_min' to 'feasible_max' for consistency with maximization objective.
    results["feasible_max"] = {"x": feasible_x, "y": feasible_y}
    with open(os.path.join(problem, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # === Step 4: Plot mean ± std ===
    steps = np.arange(1, len(next(iter(results.values()))["mean_raw"]) + 1)
    plt.figure(figsize=(8, 5))
    for acq, style in zip(acq_types, ["tab:blue", "tab:orange", "tab:green"]):
        m = np.array(results[acq]["mean_raw"])
        s = np.array(results[acq]["std_raw"])
        plt.plot(steps, m, label=acq.upper(), color=style)
        plt.fill_between(steps, m - s, m + s, alpha=0.2, color=style)
    plt.xlabel("Iteration")
    plt.ylabel("Best feasible f(x)")
    plt.title(f"{problem}: Constrained Optimization (raw means ± std)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(problem, f"{problem}_raw_std.png"), dpi=150)

    # === Step 5: Plot regret with asymmetric stds ===
    plt.figure(figsize=(8, 5))
    for acq, style in zip(acq_types, ["tab:blue", "tab:orange", "tab:green"]):
        m = np.array(results[acq]["mean_regret"])
        s_pos = np.array(results[acq]["pos_std_regret"])
        s_neg = np.array(results[acq]["neg_std_regret"])
        plt.plot(steps, m, label=acq.upper(), color=style)
        plt.fill_between(steps, m - s_neg, m + s_pos, alpha=0.2, color=style)
    plt.xlabel("Iteration")
    plt.ylabel("Regret = f(x) - f(x*)")
    plt.title(f"{problem}: Regret Progress (with asymmetric std)")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(problem, f"{problem}_regret_asym_std.png"), dpi=150)
    
    
conduct_comparison_experiment("branin_easy_circle", n_runs=3)







