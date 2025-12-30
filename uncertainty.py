"""
This script does the following:
1. Loads real measurement data from CSV.
2. Initializes the CalibrationModel.
3. Creates the cost() wrapper.
4. Configures and runs PSO.
5. Prints and saves optimization results.

"""

import numpy as np
import pandas as pd 
import json
import os 
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares

from calibration_model import CalibrationModel
from cost_function import init_cost_model, cost
from pso import Swarm
from cost_function import get_model

def load_data (csv_path: str):

    '''
    Columns are:
    0: Time
    1: refX
    2: refY
    3: refZ
    4: testX
    5: testY
    6: testZ
    7: Temp
    '''

    df = pd.read_csv(csv_path, header = None)

    df.columns = ["Time", "refX", "refY", "refZ", "testX", "testY", "testZ", "Temp"]

    df["Temp"] = df["Temp"] - 273.15

    B_meas = df[["testX", "testY", "testZ"]].to_numpy()
    B_ref = df[["refX", "refY", "refZ"]].to_numpy()
    T = df["Temp"].to_numpy()

    import numpy as np
    print("N rows:", len(df))
    print("Temp (C) min,max:", df["Temp"].min(), df["Temp"].max())
    print("ref ranges:", np.min(B_ref, axis=0), np.max(B_ref, axis=0))
    print("meas ranges:", np.min(B_meas, axis=0), np.max(B_meas, axis=0))


    return B_meas, B_ref, T

def initialise_model(csv_path):
    '''
    Initialise the CalibrationModel and connect it to the cost() function.
    '''

    B_meas, B_ref, T = load_data(csv_path)
    print("Number of samples: ", len(B_meas))

    # Initialise the global model inside cost_function.py

    init_cost_model(B_meas, B_ref, T)

    print ("Calibration model initialised.")

    return B_meas, B_ref, T

def make_bounds():
    # Returns (lower_bounds, upper_bounds) for the 24 parameters.

    S_diag_low = 0.5
    S_diag_high = 1.5
    S_off_low = -0.5
    S_off_high = 0.5

    lower_S = np.array([
        S_diag_low, S_off_low, S_off_low,
        S_off_low, S_diag_low, S_off_low,
        S_off_low, S_off_low, S_diag_low,
        ])

    upper_S = np.array([
        S_diag_high, S_off_high, S_off_high,
        S_off_high, S_diag_high, S_off_high,
        S_off_high, S_off_high, S_diag_high,
        ])

    lower_Ks = -0.05 * np.ones(9)
    upper_Ks = 0.05 * np.ones(9)

    lower_O = -1.0 * np.ones(3)
    upper_O = 1.0 * np.ones(3)

    lower_Ko = -0.05 * np.ones(3)
    upper_Ko = 0.05 * np.ones(3)

    # Concatenate all

    lower_bounds = np.concatenate([lower_S, lower_Ks, lower_O, lower_Ko])
    upper_bounds = np.concatenate([upper_S, upper_Ks, upper_O, upper_Ko])

    return lower_bounds, upper_bounds


def ls_residuals(params, B_meas, B_ref, T):
    
    # Residual Vector for LS refinement
    

    model = get_model()
    B_cal_ls = model.apply(params)

    return (B_cal_ls - B_ref).flatten()

def refine_least_squares(initial_params, B_meas, B_ref, T):

    result = least_squares(
        fun = ls_residuals,
        x0 = initial_params,
        args = (B_meas, B_ref, T),
        method = "trf",
        max_nfev = 10000,
        xtol = 1e-12,
        ftol = 1e-12,
        gtol = 1e-12
    )

    refined_params = result.x 
    residual_vec = ls_residuals(refined_params, B_meas, B_ref, T)
    refined_rmse = np.sqrt(np.mean(residual_vec**2))

    print("Refined RMSE: ", refined_rmse)

    return refined_params, refined_rmse


def run_pso(csv_path):
    # Full Pipeline

    B_meas, B_ref, T = initialise_model(csv_path)

    print("\nModel and data ready.")
    print("Running PSO on", len(B_meas), "samples...")

    lower_bounds, upper_bounds = make_bounds()
    bounds_main = (lower_bounds, upper_bounds)

    swarm = Swarm(n_particles = 200, dim = 24, bounds  = bounds_main, inertia = 0.5, cognitive = 1.5, social = 1.5)

    print("Swarm Initialised With 200 Particles.")

    t_pso_start = time.perf_counter()

    best_pos, best_cost = swarm.run(n_iterations = 200, verbose = True)

    t_pso_end = time.perf_counter()
    pso_time = t_pso_end - t_pso_start

    print(f"\nPSO computation time: {pso_time:.3f} seconds")

    print("\n===== PSO Finished ====")
    print("Best RMSE:", best_cost)
    print("Best Parameters:")
    print(best_pos)


    S = best_pos[0:9].reshape(3, 3)
    Ks = best_pos[9:18].reshape(3, 3)
    O = best_pos[18:21].reshape(3)
    Ko = best_pos[21:24].reshape(3)

    print("\n====================")
    print("  CALIBRATION MATRICES Before LS")
    print("====================\n")

    print("S (Soft-iron 3×3):")
    print(S, "\n")

    print("Ks (Temp soft-iron 3×3):")
    print(Ks, "\n")

    print("O (Hard-iron offset 3×1):")
    print(O, "\n")

    print("Ko (Temp offset 3×1):")
    print(Ko, "\n")


    # ---- Least Squares Refinement ----

    t_ls_start = time.perf_counter()

    refined_params, refined_rmse = refine_least_squares(best_pos, B_meas, B_ref, T)

    t_ls_end = time.perf_counter()
    ls_time = t_ls_end - t_ls_start

    print(f"Least Squares computation time: {ls_time:.3f} seconds")

    print("\n===== After Least-Squares Refinement =====")
    print("Refined RMSE:", refined_rmse)
    print("Refined Parameters:")
    print(refined_params)

    print("\n===== COMPUTATION TIME SUMMARY =====")
    print(f"PSO time: {pso_time:.3f} seconds")
    print(f"LS time: {ls_time:.3f} seconds")
    print(f"Total PSO + LS time: {(pso_time + ls_time):.3f} seconds")
    print("===================================\n")

    model = get_model()
    B_cal_refined = model.apply(refined_params)

    # RMSE Summary

    print("\n===== RMSE Summary =====")
    print("PSO RMSE:", best_cost)
    print("Refined LS RMSE:", refined_rmse)
    print("Improvement:", best_cost - refined_rmse)
    print("========================\n")


    # results = {"best_cost": float(best_cost), "best_params": best_pos.tolist(),}

    results = {
    "pso_best_cost": float(best_cost),
    "pso_best_params": best_pos.tolist(),
    "refined_rmse": float(refined_rmse),
    "refined_params": refined_params.tolist(),
    }


    # Parameters After Refinement

    S2 = refined_params[0:9].reshape(3, 3)
    Ks2 = refined_params[9:18].reshape(3, 3)
    O2 = refined_params[18:21].reshape(3)
    Ko2 = refined_params[21:24].reshape(3)

    print("\n====================")
    print("  CALIBRATION MATRICES After LS")
    print("====================\n")

    print("S (Soft-iron 3×3):")
    print(S2, "\n")

    print("Ks (Temp soft-iron 3×3):")
    print(Ks2, "\n")

    print("O (Hard-iron offset 3×1):")
    print(O2, "\n")

    print("Ko (Temp offset 3×1):")
    print(Ko2, "\n")




    return best_pos, best_cost, refined_params, refined_rmse

def compute_aleatoric_uncertainty(refined_params, B_ref):
    """
    Aleatoric uncertainty estimated from residual statistics.
    Assumes Gaussian measurement noise.

    Returns:
        alea_std: shape (3,) -> per-axis noise std
    """
    model = get_model()
    B_cal = model.apply(refined_params)

    residuals = B_cal - B_ref  # shape (N, 3)

    alea_std = residuals.std(axis=0)

    print("\n===== Aleatoric Uncertainty (Residual Noise) =====")
    print("X-axis std:", alea_std[0])
    print("Y-axis std:", alea_std[1])
    print("Z-axis std:", alea_std[2])

    return alea_std

def compute_epistemic_uncertainty_mc(param_samples, B_meas):
    """
    Monte Carlo epistemic uncertainty via parameter sampling.

    param_samples: shape (n_trials, 24)
    B_meas: shape (N, 3)

    Returns:
        epi_std: shape (N, 3) -> per-sample epistemic std
        mean_pred: shape (N, 3)
    """
    model = get_model()
    preds = []

    for p in param_samples:
        preds.append(model.apply(p))

    preds = np.stack(preds, axis=0)  # (n_trials, N, 3)

    mean_pred = preds.mean(axis=0)
    epi_std = preds.std(axis=0)

    print("\n===== Epistemic Uncertainty (MC over parameters) =====")
    print("Mean epistemic std:")
    print("X-axis:", epi_std[:,0].mean())
    print("Y-axis:", epi_std[:,1].mean())
    print("Z-axis:", epi_std[:,2].mean())

    return mean_pred, epi_std

def compute_total_uncertainty_and_ci(mean_pred, epi_std, alea_std, B_ref, z=1.96):
    """
    Combines aleatoric + epistemic uncertainty,
    computes CI and coverage.

    Returns:
        total_std, CI_lower, CI_upper, coverage
    """
    total_std = np.sqrt(epi_std**2 + alea_std**2)

    CI_lower = mean_pred - z * total_std
    CI_upper = mean_pred + z * total_std

    within_CI = np.logical_and(
        B_ref >= CI_lower,
        B_ref <= CI_upper
    )

    coverage = within_CI.mean(axis=0) * 100

    print("\n===== 95% Confidence Interval Coverage =====")
    print(f"X-axis: {coverage[0]:.2f}%")
    print(f"Y-axis: {coverage[1]:.2f}%")
    print(f"Z-axis: {coverage[2]:.2f}%")

    return total_std, CI_lower, CI_upper, coverage

def run_pso_ls_multiple(csv_path, n_trials=20):
    """
    Runs PSO + LS multiple times to estimate epistemic uncertainty.
    """

    B_meas, B_ref, T = initialise_model(csv_path)
    lower_bounds, upper_bounds = make_bounds()
    bounds = (lower_bounds, upper_bounds)

    refined_params_all = []
    rmse_all = []

    for trial in range(n_trials):
        print(f"\n--- PSO+LS Trial {trial+1}/{n_trials} ---")

        swarm = Swarm(
            n_particles=100,
            dim=24,
            bounds=bounds,
            inertia=0.5,
            cognitive=1.5,
            social=1.5
        )

        best_pos, _ = swarm.run(n_iterations=200, verbose=False)

        refined_params, refined_rmse = refine_least_squares(
            best_pos, B_meas, B_ref, T
        )

        refined_params_all.append(refined_params)
        rmse_all.append(refined_rmse)

    return (
        np.array(refined_params_all),
        np.array(rmse_all),
        B_meas,
        B_ref
    )


def run_ls_only_multiple(csv_path, n_trials=20):
    """
    Runs LS-only calibration multiple times to estimate
    epistemic uncertainty for LS alone.
    """

    B_meas, B_ref, T = initialise_model(csv_path)

    refined_params_all = []
    rmses = []

    for trial in range(n_trials):
        print(f"\n--- LS-only Trial {trial+1}/{n_trials} ---")

        # Identity-based initial guess + small noise
        init_params = np.zeros(24)
        init_params[0] = init_params[4] = init_params[8] = 1.0
        init_params += 0.01 * np.random.randn(24)

        refined_params, refined_rmse = refine_least_squares(
            init_params, B_meas, B_ref, T
        )

        refined_params_all.append(refined_params)
        rmses.append(refined_rmse)

    refined_params_all = np.array(refined_params_all)
    rmses = np.array(rmses)

    print("\n===== LS-only RMSE Summary =====")
    print("Mean RMSE:", rmses.mean())
    print("Std RMSE :", rmses.std())

    return refined_params_all, rmses, B_meas, B_ref

def make_local_bounds(center, radius=0.2):
    """
    Create local PSO bounds around an LS solution.
    """
    eps = 1e-6
    lower = center - radius * np.abs(center + eps)
    upper = center + radius * np.abs(center + eps)
    return lower, upper

def run_ls_then_pso(csv_path, local_radius=0.2):
    """
    Runs LS first, then PSO in a local neighborhood of the LS solution.
    """

    # ---- LS step ----
    B_meas, B_ref, T = initialise_model(csv_path)

    init_params = np.zeros(24)
    init_params[0] = init_params[4] = init_params[8] = 1.0
    init_params += 0.01 * np.random.randn(24)

    ls_params, ls_rmse = refine_least_squares(
        init_params, B_meas, B_ref, T
    )

    # ---- Local PSO step ----
    lower, upper = make_local_bounds(ls_params, radius=local_radius)

    swarm = Swarm(
        n_particles=100,
        dim=24,
        bounds=(lower, upper),
        inertia=0.5,
        cognitive=1.5,
        social=1.5
    )

    best_pos, best_cost = swarm.run(n_iterations=200, verbose=False)

    return ls_params, ls_rmse, best_pos, best_cost, B_meas, B_ref

def run_ls_pso_multiple(csv_path, n_trials=20, local_radius=0.2):
    """
    Runs LS → PSO multiple times to estimate epistemic uncertainty.
    """

    refined_params_all = []
    rmses = []

    for trial in range(n_trials):
        print(f"\n--- LS → PSO Trial {trial+1}/{n_trials} ---")

        _, _, pso_params, pso_rmse, B_meas, B_ref = run_ls_then_pso(
            csv_path, local_radius
        )

        refined_params_all.append(pso_params)
        rmses.append(pso_rmse)

    refined_params_all = np.array(refined_params_all)
    rmses = np.array(rmses)

    print("\n===== LS → PSO RMSE Summary =====")
    print("Mean RMSE:", rmses.mean())
    print("Std RMSE :", rmses.std())

    return refined_params_all, rmses, B_meas, B_ref


'''if (__name__ == "__main__"):
    run_pso("full_data.csv")'''

def build_pipeline_summary(
    name,
    param_samples,
    rmse_samples,
    B_meas,
    B_ref
):
    """
    Builds a standardized summary dict for a pipeline.
    """

    # RMSE statistics
    rmse_mean = rmse_samples.mean()
    rmse_std = rmse_samples.std()

    # Mean parameter estimate
    mean_params = param_samples.mean(axis=0)

    # Aleatoric uncertainty
    alea_std = compute_aleatoric_uncertainty(mean_params, B_ref)

    # Epistemic uncertainty
    mean_pred, epi_std = compute_epistemic_uncertainty_mc(
        param_samples, B_meas
    )
    epi_std_mean = epi_std.mean(axis=0)

    # CI coverage
    _, _, _, coverage = compute_total_uncertainty_and_ci(
        mean_pred, epi_std, alea_std, B_ref
    )

    return {
        "pipeline": name,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "alea_std": alea_std,
        "epi_std": epi_std_mean,
        "ci_coverage": coverage
    }


if __name__ == "__main__":

    summaries = []

    # ============================================================
    # PSO → LS
    # ============================================================

    print("\n===== PSO → LS =====")

    pso_ls_params, pso_ls_rmses, B_meas, B_ref = run_pso_ls_multiple(
        "full_data.csv",
        n_trials=20
    )

    summary_pso_ls = build_pipeline_summary(
        "PSO → LS",
        pso_ls_params,
        pso_ls_rmses,
        B_meas,
        B_ref
    )

    summaries.append(summary_pso_ls)

    # ============================================================
    # LS ONLY
    # ============================================================

    print("\n===== LS ONLY =====")

    ls_params, ls_rmses, B_meas, B_ref = run_ls_only_multiple(
        "full_data.csv",
        n_trials=20
    )

    summary_ls = build_pipeline_summary(
        "LS only",
        ls_params,
        ls_rmses,
        B_meas,
        B_ref
    )

    summaries.append(summary_ls)

    # ============================================================
    # LS → PSO
    # ============================================================

    print("\n===== LS → PSO =====")

    ls_pso_params, ls_pso_rmses, B_meas, B_ref = run_ls_pso_multiple(
        "full_data.csv",
        n_trials=20,
        local_radius=0.2
    )

    summary_ls_pso = build_pipeline_summary(
        "LS → PSO",
        ls_pso_params,
        ls_pso_rmses,
        B_meas,
        B_ref
    )

    summaries.append(summary_ls_pso)

    # ============================================================
    # FINAL COMPARISON TABLE
    # ============================================================

    print("\n================ PIPELINE COMPARISON ================\n")
    print(
        f"{'Pipeline':<10} | "
        f"{'RMSE μ':>10} | "
        f"{'RMSE σ':>10} | "
        f"{'Aleatoric (μ)':>20} | "
        f"{'Epistemic (μ)':>20} | "
        f"{'CI Cov (%)':>20}"
    )
    print("-" * 120)

    for s in summaries:
        print(
            f"{s['pipeline']:<10} | "
            f"{s['rmse_mean']:10.4e} | "
            f"{s['rmse_std']:10.4e} | "
            f"{np.mean(s['alea_std']):20.4e} | "
            f"{np.mean(s['epi_std']):20.4e} | "
            f"{np.mean(s['ci_coverage']):20.2f}"
        )

    print("\n=====================================================\n")



'''if __name__ == "__main__":

    # ---- Single best solution (existing pipeline) ----
    best_pos, best_cost, refined_params, refined_rmse = run_pso("full_data.csv")

    # ============================================================
    # PSO - LS UNCERTAINTY ANALYSIS
    # ============================================================

    # ---- Aleatoric uncertainty ----
    _, B_ref, _ = initialise_model("full_data.csv")
    mean_pso_ls_params = param_samples.mean(axis=0)
    alea_std = compute_aleatoric_uncertainty(mean_pso_ls_params, B_ref)


    # ---- Epistemic uncertainty via MC PSO+LS ----
    param_samples, B_meas, B_ref = run_pso_ls_multiple(
        "full_data.csv",
        n_trials=20
    )

    mean_pred, epi_std = compute_epistemic_uncertainty_mc(
        param_samples, B_meas
    )

    # ---- Total uncertainty + CI + coverage ----
    compute_total_uncertainty_and_ci(
        mean_pred, epi_std, alea_std, B_ref
    )

    # ============================================================
    # LS-ONLY UNCERTAINTY ANALYSIS
    # ============================================================

    print("\n===== LS ONLY =====")

    ls_param_samples, ls_rmses, B_meas, B_ref = run_ls_only_multiple(
        "full_data.csv",
        n_trials=20
    )

    # Mean LS RMSE
    print("\nLS-only Mean RMSE:", ls_rmses.mean())

    # Aleatoric uncertainty (use mean LS solution)
    mean_ls_params = ls_param_samples.mean(axis=0)
    alea_std_ls = compute_aleatoric_uncertainty(mean_ls_params, B_ref)

    # Epistemic uncertainty (MC over LS params)
    mean_pred_ls, epi_std_ls = compute_epistemic_uncertainty_mc(
        ls_param_samples, B_meas
    )

    # Total uncertainty + CI coverage
    compute_total_uncertainty_and_ci(
        mean_pred_ls,
        epi_std_ls,
        alea_std_ls,
        B_ref
    )

    # ============================================================
    # LS - PSO UNCERTAINTY ANALYSIS
    # ============================================================

    print("\n===== LS → PSO =====")

    ls_pso_params, ls_pso_rmses, B_meas, B_ref = run_ls_pso_multiple(
        "full_data.csv",
        n_trials=20,
        local_radius=0.2
    )

    # Mean LS → PSO RMSE
    print("\nLS → PSO Mean RMSE:", ls_pso_rmses.mean())

    # Aleatoric uncertainty (use mean LS → PSO params)
    mean_ls_pso_params = ls_pso_params.mean(axis=0)
    alea_std_ls_pso = compute_aleatoric_uncertainty(
        mean_ls_pso_params, B_ref
    )

    # Epistemic uncertainty
    mean_pred_ls_pso, epi_std_ls_pso = compute_epistemic_uncertainty_mc(
        ls_pso_params, B_meas
    )

    # Total uncertainty + CI coverage
    compute_total_uncertainty_and_ci(
        mean_pred_ls_pso,
        epi_std_ls_pso,
        alea_std_ls_pso,
        B_ref
    )
'''




