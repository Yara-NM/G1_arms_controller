#!/usr/bin/env python3
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG -----------------------------------------------------
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "results" / "task_space_log_2_real.csv"
# Task-space spring–damper gains (must match what you used in the test)
K = np.array([30.0, 30.0, 25.0])   # [Kx, Ky, Kz]
D = np.array([15.0, 15.0, 12.0])   # [Dx, Dy, Dz]

# ----------------------------------------------------------------

def load_log(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Log file not found: {path}")

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    def v3(prefix, arm, kind):
        # prefix: 'x', arm: 'R' or 'L', kind: 'meas' or 'ref'
        return np.vstack([
            np.array([float(r[f"{prefix}{arm}_{kind}_x"]) for r in rows]),
            np.array([float(r[f"{prefix}{arm}_{kind}_y"]) for r in rows]),
            np.array([float(r[f"{prefix}{arm}_{kind}_z"]) for r in rows]),
        ])  # shape (3, N)

    t = np.array([float(r["t"]) for r in rows])

    xR_meas = v3("x", "R", "meas")
    xL_meas = v3("x", "L", "meas")
    xR_ref  = v3("x", "R", "des")
    xL_ref  = v3("x", "L", "des")

    return t, xR_meas, xL_meas, xR_ref, xL_ref

def finite_diff(x, t):
    """
    x: (3, N), t: (N,)
    returns v: (3, N) with simple backward difference
    """
    v = np.zeros_like(x)
    dt = np.diff(t)
    dt[dt <= 1e-6] = 1e-6  # avoid division by zero

    # backward diff for i>=1, v[:,0] = 0
    v[:, 1:] = (x[:, 1:] - x[:, :-1]) / dt
    v[:, 0] = v[:, 1]  # or keep zero, up to you
    return v

def main():
    print(f"[INFO] Loading log from: {LOG_PATH}")
    t, xR_meas, xL_meas, xR_ref, xL_ref = load_log(LOG_PATH)

    # --- approximate velocities from log ---
    vR_meas = finite_diff(xR_meas, t)
    vL_meas = finite_diff(xL_meas, t)
    vR_ref  = finite_diff(xR_ref,  t)
    vL_ref  = finite_diff(xL_ref,  t)

    # --- compute virtual forces ---
    # F = K * (x_ref - x_meas) + D * (v_ref - v_meas)
    eR_pos = xR_ref - xR_meas
    eL_pos = xL_ref - xL_meas
    eR_vel = vR_ref - vR_meas
    eL_vel = vL_ref - vL_meas

    # broadcast K,D across time:
    FR = K[:, None] * eR_pos + D[:, None] * eR_vel  # (3,N)
    FL = K[:, None] * eL_pos + D[:, None] * eL_vel  # (3,N)

    FR_norm = np.linalg.norm(FR, axis=0)
    FL_norm = np.linalg.norm(FL, axis=0)

    # --- PLOTS ---------------------------------------------------

    # 1) Norms over time
    plt.figure(figsize=(8, 4))
    plt.plot(t, FR_norm, label="‖F_R‖")
    plt.plot(t, FL_norm, label="‖F_L‖")
    plt.xlabel("time [s]")
    plt.ylabel("virtual force norm [N]")
    plt.title("Virtual spring–damper force norms")
    plt.grid(True)
    plt.legend()

    # 2) Right arm forces
    fig2, axs2 = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    comp = ["x", "y", "z"]
    for i in range(3):
        axs2[i].plot(t, FR[i, :], label=f"F_R_{comp[i]}")
        axs2[i].set_ylabel(f"{comp[i]} [N]")
        axs2[i].grid(True)
        axs2[i].legend(loc="upper right")
    axs2[-1].set_xlabel("time [s]")
    fig2.suptitle("Right EE virtual force components")

    # 3) Left arm forces
    fig3, axs3 = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    for i in range(3):
        axs3[i].plot(t, FL[i, :], label=f"F_L_{comp[i]}")
        axs3[i].set_ylabel(f"{comp[i]} [N]")
        axs3[i].grid(True)
        axs3[i].legend(loc="upper right")
    axs3[-1].set_xlabel("time [s]")
    fig3.suptitle("Left EE virtual force components")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
