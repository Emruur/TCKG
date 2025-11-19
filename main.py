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
    k_values = np.logspace(-2, 2, 15) 
    
    # Linearly spaced alpha values (0.0 to 1.0), 10 values (excluding 1.0)
    alpha_values = np.linspace(0.0, 1.0, 15, endpoint=False) 

    hparam_list = []
    for k in k_values:
        for alpha in alpha_values:
            hparam_list.append({"k": k, "alpha": alpha})
    return hparam_list


def conduct_comparison_experiment(problem, n_runs=10, dim=2):
    """
    Runs 100 SCEI variants and CEI, saves results to <problem>_kablation/, 
    plots regret with symmetric STD, and performs an ablation analysis on k and alpha.
    """
    
    # 1. Set the new output directory
    output_dir = f"{problem}_kablation"
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # === Step 1: Identify feasible maximum ===
    feasible_x, feasible_y = find_feasible_maximum(problem, dim)
    print(f"Feasible Maximum for {problem}: f(x*) = {feasible_y:.5f} at {feasible_x}")

    # === Define Acquisition List ===
    acq_list = [("cei", {})]
    scei_hparams_list = generate_scei_hparams_list()
    for hparams in scei_hparams_list:
        name = f"scei_k{hparams['k']:.2e}_a{hparams['alpha']:.2f}"
        acq_list.append((name, hparams))
    
    print(f"\n--- Running {len(acq_list)} different acquisition strategies ({len(scei_hparams_list)} SCEI variants) ---")
    
    # Initialize lists for ablation analysis
    all_scei_regrets = []
    all_scei_aucs = []
    
    # === Step 2: Run experiments for all acq types ===
    for i, (acq_name, hparams) in enumerate(acq_list):
        acq_type = "cei" if acq_name == "cei" else "scei"
        
        # Run experiment (setting visualize=False)
        mean_raw, std_raw, all_runs = run_experiment(
            problem, acq_type=acq_type, n_runs=n_runs, visualize=False, dim=dim, scei_params=hparams
        )

        # Compute regret: Regret = f(x*) - f(x)
        mean_regret = feasible_y - mean_raw
        
        # Compute asymmetric std (still needed for plotting best variant uncertainty, though not for the main plot)
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
            all_scei_regrets.append({'k': hparams['k'], 'alpha': hparams['alpha'], 'value': mean_regret[-1]})
            all_scei_aucs.append({'k': hparams['k'], 'alpha': hparams['alpha'], 'value': auc})
        
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Completed {i+1}/{len(acq_list)}: {acq_name}")


    # === Step 3: Save JSON with all results ===
    results["feasible_max"] = {"x": feasible_x, "y": feasible_y}
    with open(os.path.join(output_dir, "results_all.json"), "w") as f:
        json.dump(results, f, indent=4)

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
    
    # --- 4b. Variance of Objectives on alpha and k ---
    k_values = np.logspace(-2, 2, 10)
    alpha_values = np.linspace(0.0, 1.0, 10, endpoint=False)
    
    # Variance over K (Averaging Regret/AUC for fixed K across all Alpha)
    regret_by_k = {k: [] for k in k_values}
    auc_by_k = {k: [] for k in k_values}
    
    # Variance over Alpha (Averaging Regret/AUC for fixed Alpha across all K)
    regret_by_alpha = {a: [] for a in alpha_values}
    auc_by_alpha = {a: [] for a in alpha_values}

    # Populate variance data structures
    for r in all_scei_regrets:
        regret_by_k[r['k']].append(r['value'])
        regret_by_alpha[r['alpha']].append(r['value'])
        
    for r in all_scei_aucs:
        auc_by_k[r['k']].append(r['value'])
        auc_by_alpha[r['alpha']].append(r['value'])

    # Calculate variances
    ablation_data["Regret_Variance_vs_K"] = {k: np.var(v) for k, v in regret_by_k.items()}
    ablation_data["Regret_Variance_vs_Alpha"] = {a: np.var(v) for a, v in regret_by_alpha.items()}
    ablation_data["AUC_Variance_vs_K"] = {k: np.var(v) for k, v in auc_by_k.items()}
    ablation_data["AUC_Variance_vs_Alpha"] = {a: np.var(v) for a, v in auc_by_alpha.items()}
    
    # Save ablation analysis
    with open(os.path.join(output_dir, "ablation_analysis.json"), "w") as f:
        json.dump(ablation_data, f, indent=4, cls=NumpyEncoder) # Use custom encoder for numpy floats

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
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float_, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.ndarray)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def conduct_experiment(problem, n_runs=10, dim=2):
    """
    Compares CEI and a fixed-parameter SCEI (k=13, alpha=0.6) performance 
    on a constrained maximization problem, plotting both symmetric and 
    asymmetric standard deviation on the regret curve.
    """
    os.makedirs(problem, exist_ok=True)
    results = {}

    # === Step 1: Identify feasible maximum ===
    feasible_x, feasible_y = find_feasible_maximum(problem, dim)
    print(f"Feasible Maximum for {problem}: f(x*) = {feasible_y:.5f} at {feasible_x}")

    # === Step 2: Run experiments for all acq types ===
    
    acq_types = ["cei","scei"]
    styles = {"cei": "tab:blue", "scei": "tab:orange"} 
    
    for acq in acq_types:
        scei_params = {
            "k": 13, 
            "alpha": 0.4
        }
        
        # Run experiment
        mean_raw, std_raw, all_runs = run_experiment(problem ,acq_type=acq, n_runs=n_runs, visualize= False ,dim=dim, scei_params= scei_params)

        # Compute regret: Regret = f(x*) - f(x) for maximization
        mean_regret = feasible_y - mean_raw
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
    
conduct_comparison_experiment("goldstein_annulus", n_runs=10)

#conduct_experiment("goldstein_annulus", n_runs=10)







