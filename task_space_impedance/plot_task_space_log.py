# plot_task_space_log.py
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

LOG_PATH = os.path.join(os.path.dirname(__file__),"task_space_log.csv")

def load_log(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError("Log file is empty.")

    # Adjust these names if your header is different
    t = np.array([float(r["t"]) for r in rows])

    def vec3(prefix, arm, kind):
        # prefix: "x", arm: "R" or "L", kind: "meas" or "ref"
        return np.vstack([
            np.array([float(r[f"{prefix}{arm}_{kind}_x"]) for r in rows]),
            np.array([float(r[f"{prefix}{arm}_{kind}_y"]) for r in rows]),
            np.array([float(r[f"{prefix}{arm}_{kind}_z"]) for r in rows]),
        ])  # shape (3, N)

    xR_meas = vec3("x", "R", "meas")
    xL_meas = vec3("x", "L", "meas")
    xR_ref  = vec3("x", "R", "des")
    xL_ref  = vec3("x", "L", "des")

    return t, xR_meas, xL_meas, xR_ref, xL_ref

def main():
    print(f"[INFO] Loading log from: {LOG_PATH}")
    t, xR_meas, xL_meas, xR_ref, xL_ref = load_log(LOG_PATH)

    # --- Right arm positions ---
    fig1, axs1 = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    labels = ["x", "y", "z"]

    for i in range(3):
        axs1[i].plot(t, xR_meas[i, :], label=f"R_meas_{labels[i]}")
        axs1[i].plot(t, xR_ref[i, :],  "--", label=f"R_ref_{labels[i]}")
        axs1[i].set_ylabel(f"{labels[i]} [m]")
        axs1[i].grid(True)
        axs1[i].legend(loc="best")

    axs1[-1].set_xlabel("time [s]")
    fig1.suptitle("Right arm EE position (measured vs reference)")
    fig1.tight_layout()

    # --- Left arm positions ---
    fig2, axs2 = plt.subplots(3, 1, sharex=True, figsize=(8, 8))

    for i in range(3):
        axs2[i].plot(t, xL_meas[i, :], label=f"L_meas_{labels[i]}")
        axs2[i].plot(t, xL_ref[i, :],  "--", label=f"L_ref_{labels[i]}")
        axs2[i].set_ylabel(f"{labels[i]} [m]")
        axs2[i].grid(True)
        axs2[i].legend(loc="best")

    axs2[-1].set_xlabel("time [s]")
    fig2.suptitle("Left arm EE position (measured vs reference)")
    fig2.tight_layout()

    # --- Position error norms (optional but useful) ---
    errR = np.linalg.norm(xR_ref - xR_meas, axis=0)
    errL = np.linalg.norm(xL_ref - xL_meas, axis=0)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(t, errR, label="‖e_R‖")
    ax3.plot(t, errL, label="‖e_L‖")
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("position error [m]")
    ax3.grid(True)
    ax3.legend(loc="best")
    fig3.suptitle("EE position error norms")
    fig3.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
