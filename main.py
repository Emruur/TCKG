import numpy as np
from benchmarks import BENCHMARKS, expand_constraints, find_feasible_maximum
from BayesianOptimizer import BayesianOptimizer
import os
import json
import matplotlib.pyplot as plt

def run_experiment(problem, acq_type="cei" ,n_runs=1, visualize= True, dim= 2):
    
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
        progress= bo.run(visualize= visualize, acq_type= acq_type)
        
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

def conduct_comparison_experiment(problem, n_runs=10, dim=2):
    os.makedirs(problem, exist_ok=True)
    results = {}

    # === Step 1: Identify feasible minimum ===
    feasible_x, feasible_y = find_feasible_maximum(problem, dim)
    print(f"Feasible minimum for {problem}: f(x*) = {feasible_y:.5f} at {feasible_x}")

    # === Step 2: Run experiments for all acq types ===
    acq_types = ["cei","scei"]
    for acq in acq_types:
        mean_raw, std_raw, all_runs = run_experiment(problem ,acq_type=acq, n_runs=n_runs, visualize= True ,dim=dim)

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
    results["feasible_min"] = {"x": feasible_x, "y": feasible_y}
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







