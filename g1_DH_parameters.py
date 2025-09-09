#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import csv
import math

np.set_printoptions(precision=6, suppress=True, linewidth=180)

# ---------- Small geometry helpers ----------
def homog(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

def rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

def rx(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,    c,   -s],
                     [0.0,    s,    c]])

def T_from_DH(a, alpha, d, theta):
    """Classic DH: Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)."""
    R = rz(theta) @ rx(alpha)
    p = np.array([a * np.cos(theta), a * np.sin(theta), d])
    return homog(R, p)

def DH_from_T(T):
    """
    Extract classic DH (a, alpha, d, theta0) from ^iT_{i+1}
    for convention: Rz(theta) Tz(d) Tx(a) Rx(alpha).
    """
    R = T[:3, :3]
    theta = np.arctan2(R[1, 0], R[0, 0])
    alpha = np.arctan2(R[2, 1], R[2, 2])
    a = float(np.hypot(T[0, 3], T[1, 3]))
    d = float(T[2, 3])

    def wrap(x): return (x + np.pi) % (2*np.pi) - np.pi
    return float(a), float(wrap(alpha)), float(d), float(wrap(theta))

def joint_short_type(jmodel):
    try:
        return jmodel.shortname()
    except Exception:
        return jmodel.__class__.__name__

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def _skew(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])

def _orth(v, n):
    # component of v orthogonal to unit n
    return v - np.dot(v, n) * n

def _safe_dir_from_seed(n):
    # pick any unit vector not parallel to n
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, n)) > 0.95:
        seed = np.array([0.0, 1.0, 0.0])
    return _unit(_orth(seed, n))

def _closest_points_on_lines(p1, z1, p2, z2, eps=1e-10):
    """
    Lines: L1: p1 + s z1,  L2: p2 + t z2  (z1,z2 unit).
    Returns closest points O1, O2 and 'parallel' flag.
    """
    z1 = _unit(z1); z2 = _unit(z2)
    w0 = p1 - p2
    c = np.dot(z1, z2)
    denom = 1.0 - c*c
    if abs(denom) < eps:
        # Parallel (or nearly)
        t = np.dot(z2, (p1 - p2))
        O1 = p1
        O2 = p2 + t * z2
        return O1, O2, True
    a = np.dot(z1, w0)
    b = np.dot(z2, w0)
    s = (a - c*b) / denom
    t = (a*c - b) / denom
    O1 = p1 - s * z1
    O2 = p2 - t * z2
    return O1, O2, False

def _recon_error_breakdown(T_true, T_guess):
    R = T_true[:3,:3]; p = T_true[:3,3]
    Rg = T_guess[:3,:3]; pg = T_guess[:3,3]
    e_frob = float(np.linalg.norm(T_true - T_guess, ord='fro'))
    Rdiff = R.T @ Rg
    tr = np.clip((np.trace(Rdiff) - 1.0)/2.0, -1.0, 1.0)
    ang = float(np.degrees(np.arccos(tr)))
    e_trans_mm = float(1000.0*np.linalg.norm(p - pg))
    return e_frob, ang, e_trans_mm


# ---------- Main class ----------
class G1_DH_Arms:
    """
    Emits classic DH (exact, common-normal frames), CSV export, and
    validates FK vs Pinocchio in the same canonical frame.
    """
    def __init__(self):
        base_path = os.path.abspath(os.path.dirname(__file__))
        urdf_path = os.path.join(base_path, "assets/g1/g1_body29_hand14.urdf")
        mesh_dir = os.path.join(base_path, "assets/g1")
        self.robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_dir])

        # Lock legs, waist, and fingers
        joints_to_lock = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint", "left_hand_middle_1_joint",
            "left_hand_index_0_joint", "left_hand_index_1_joint",
            "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
            "right_hand_index_0_joint", "right_hand_index_1_joint",
            "right_hand_middle_0_joint", "right_hand_middle_1_joint"
        ]
        self.rmodel = self.robot.buildReducedRobot(joints_to_lock, np.zeros(self.robot.model.nq)).model
        self.rdata = self.rmodel.createData()

        self.q0 = pin.neutral(self.rmodel)
        pin.forwardKinematics(self.rmodel, self.rdata, self.q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)

    def _active_chain_to(self, end_joint_name, side_hint=("left_",)):
        try:
            j = self.rmodel.getJointId(end_joint_name)
        except Exception:
            raise RuntimeError(f"Joint '{end_joint_name}' not found in reduced model.")

        ids = []
        side_ok = True
        while j != 0 and side_ok:
            jname = self.rmodel.names[j]
            if self.rmodel.joints[j].nv > 0:
                ids.append(j)
            parent = self.rmodel.parents[j]
            if side_hint is not None:
                side_ok = any(tag in jname for tag in side_hint) or any(tag in self.rmodel.names[parent] for tag in side_hint)
            j = parent

        ids.reverse()
        if len(ids) < 2:
            raise RuntimeError("Chain too short — did the reduced model lock arm joints?")
        return ids

    def _joint_axis_world(self, jid):
        """World pose of joint frame and its axis in world coords."""
        Tw = self.rdata.oMi[jid].homogeneous
        Rw = Tw[:3,:3]; pw = Tw[:3,3]
        jshort = joint_short_type(self.rmodel.joints[jid]).upper()
        if 'RX' in jshort: a_loc = np.array([1.0,0.0,0.0])
        elif 'RY' in jshort: a_loc = np.array([0.0,1.0,0.0])
        else: a_loc = np.array([0.0,0.0,1.0])
        return Tw, pw, _unit(Rw @ a_loc)

    def _build_true_dh_frames(self, chain_ids):
        """
        Build A_i such that: URDF_i = A_i * DH_i, with DH_i following classic rules:
          - z_i = joint axis
          - x_i along common normal between z_i and z_{i+1}
          - O_i on z_i at closest point to z_{i+1}
        """
        pin.forwardKinematics(self.rmodel, self.rdata, self.q0)

        Tw = {}; pw = {}; z_w = {}
        for jid in chain_ids:
            Tw[jid], pw[jid], z_w[jid] = self._joint_axis_world(jid)

        # For each i in [0..n-2], define O_i and x_i from pair (i, i+1)
        Oi = {}; xi = {}
        last_Ob = None
        for k in range(len(chain_ids) - 1):
            a = chain_ids[k]; b = chain_ids[k+1]
            Oa, Ob, _ = _closest_points_on_lines(pw[a], z_w[a], pw[b], z_w[b])
            v = Ob - Oa
            xdir = _safe_dir_from_seed(z_w[a]) if np.linalg.norm(v) < 1e-12 else _unit(v)
            Oi[a] = Oa
            xi[a] = _unit(_orth(xdir, z_w[a]))  # ensure ⟂ z_a
            last_Ob = Ob

        # Last joint: put origin at its closest point to previous; choose x_n ⟂ z_n near previous x
        last = chain_ids[-1]; prev = chain_ids[-2]
        if last not in Oi:
            Oi[last] = last_Ob if last_Ob is not None else pw[last]
        x_last = _unit(_orth(xi[prev], z_w[last]))
        if np.linalg.norm(x_last) < 1e-12:
            x_last = _safe_dir_from_seed(z_w[last])
        xi[last] = x_last

        # Build A_i (URDF = A * DH), i.e., ^URDF T_DH = A  (rotation+translation)
        A = {}
        for jid in chain_ids:
            Rw = Tw[jid][:3,:3]; pw_j = Tw[jid][:3,3]
            z_u = _unit(Rw.T @ z_w[jid])
            x_u = _unit(Rw.T @ xi[jid])
            x_u = _unit(_orth(x_u, z_u))
            y_u = _unit(np.cross(z_u, x_u))
            R_u = np.column_stack([x_u, y_u, z_u])
            p_u = Rw.T @ (Oi[jid] - pw_j)
            Ai = np.eye(4); Ai[:3,:3] = R_u; Ai[:3,3] = p_u
            A[jid] = Ai
        return A

    def _pairwise_transforms_dual(self, chain_ids, mode="exact"):
        """
        mode = "exact"      → true DH frames (A with rotation+translation via common normals)
        mode = "axis_only"  → rotation-only (kept for reference)
        """
        pin.forwardKinematics(self.rmodel, self.rdata, self.q0)
        oMi = self.rdata.oMi

        Tw = {jid: oMi[jid].homogeneous for jid in chain_ids}

        if mode == "axis_only":
            A = {}
            for jid in chain_ids:
                jshort = joint_short_type(self.rmodel.joints[jid]).upper()
                if 'RZ' in jshort or 'PZ' in jshort: ax = np.array([0,0,1],float)
                elif 'RY' in jshort or 'PY' in jshort: ax = np.array([0,1,0],float)
                else: ax = np.array([1,0,0],float)
                z = _unit(ax); x = _safe_dir_from_seed(z); y = _unit(np.cross(z, x))
                R_u = np.column_stack([x, y, z])
                Ai = np.eye(4); Ai[:3,:3] = R_u
                A[jid] = Ai
        else:
            A = self._build_true_dh_frames(chain_ids)

        mats = []
        first = chain_ids[0]
        Tw_first_urdf = Tw[first]
        # IMPORTANT: world→DH_first = world→URDF_first · A_first
        Tw_first_std  = Tw_first_urdf @ A[first]
        mats.append(("WORLD", self.rmodel.names[first],
                     {"urdf": Tw_first_urdf, "std": Tw_first_std, "jid": first,
                      "axis": A[first][:3,2], "A": A[first],
                      "jtype": joint_short_type(self.rmodel.joints[first])}))

        for i in range(len(chain_ids) - 1):
            a = chain_ids[i]; b = chain_ids[i + 1]
            T_ab_urdf = np.linalg.inv(Tw[a]) @ Tw[b]
            # IMPORTANT: ^DH_i T_DH_{i+1} = A_{i+1}^{-1} · ^URDF_i T_URDF_{i+1} · A_i
            T_ab_std  = np.linalg.inv(A[b]) @ T_ab_urdf @ A[a]
            mats.append((self.rmodel.names[a], self.rmodel.names[b],
                        {"urdf": T_ab_urdf, "std": T_ab_std, "jid_b": b,
                         "axis_b": A[b][:3,2], "A_b": A[b],
                         "jtype_b": joint_short_type(self.rmodel.joints[b])}))
        return mats

    def _dh_rows_dual(self, mats):
        rows = []
        for (from_name, to_name, payload) in mats:
            if from_name == "WORLD":
                rows.append({
                    "from": from_name, "to": to_name, "kind": "reference",
                    "T_urdf": payload["urdf"], "T_std": payload["std"],
                    "axis": payload["axis"], "A": payload["A"], "jtype": payload["jtype"],
                    "dh_urdf": None, "dh_std": None,
                    "err_urdf": None, "err_std": None,
                    "err_urdf_ang": None, "err_urdf_mm": None,
                    "err_std_ang": None, "err_std_mm": None
                })
                continue

            T_urdf = payload["urdf"]
            T_std  = payload["std"]
            a_u, alpha_u, d_u, theta0_u = DH_from_T(T_urdf)
            a_s, alpha_s, d_s, theta0_s = DH_from_T(T_std)

            Tu_chk = T_from_DH(a_u, alpha_u, d_u, theta0_u)
            Ts_chk = T_from_DH(a_s, alpha_s, d_s, theta0_s)

            err_u, ang_u_deg, trans_u_mm = _recon_error_breakdown(T_urdf, Tu_chk)
            err_s, ang_s_deg, trans_s_mm = _recon_error_breakdown(T_std,  Ts_chk)

            jtype_b = payload["jtype_b"]
            variable = "theta (revolute)" if "R" in jtype_b.upper() else ("d (prismatic)" if "P" in jtype_b.upper() else "unknown")

            rows.append({
                "from": from_name, "to": to_name, "kind": "link",
                "T_urdf": T_urdf, "T_std": T_std,
                "axis": payload["axis_b"], "A": payload["A_b"], "jtype": jtype_b,
                "dh_urdf": (a_u, alpha_u, d_u, theta0_u),
                "dh_std":  (a_s, alpha_s, d_s, theta0_s),
                "err_urdf": err_u, "err_std": err_s,
                "err_urdf_ang": ang_u_deg, "err_urdf_mm": trans_u_mm,
                "err_std_ang": ang_s_deg, "err_std_mm": trans_s_mm,
                "variable": variable
            })
        return rows

    def _canonical_table_from_rows(self, rows):
        out = []
        i = 1
        for r in rows[1:]:
            a, alpha, d, theta0 = r["dh_std"]
            out.append({
                "i": i, "from": r["from"], "to": r["to"],
                "a_m": float(a), "alpha_rad": float(alpha), "alpha_deg": float(np.degrees(alpha)),
                "d_m": float(d), "theta0_rad": float(theta0), "theta0_deg": float(np.degrees(theta0)),
                "joint_type": r["jtype"], "var": r["variable"],
                "recon_err": float(r["err_std"]),
                "rot_err_deg": float(r["err_std_ang"]), "trans_err_mm": float(r["err_std_mm"]),
            })
            i += 1
        return out

    def _print_markdown_table(self, side, table):
        print(f"\n### {side.upper()} arm — classic DH (exact, common-normal frames)\n"
              "| i | from | to | a [m] | alpha [rad] | alpha [deg] | d [m] | theta0 [rad] | theta0 [deg] | rot err [deg] | trans err [mm] |\n"
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in table:
            print(f"| {row['i']} | {row['from']} | {row['to']} | "
                  f"{row['a_m']:.6f} | {row['alpha_rad']:.6f} | {row['alpha_deg']:.3f} | "
                  f"{row['d_m']:.6f} | {row['theta0_rad']:.6f} | {row['theta0_deg']:.3f} | "
                  f"{row['rot_err_deg']:.3e} | {row['trans_err_mm']:.3e} |")

    def _write_csv(self, side, table, path=None):
        if path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base_dir, f"dh_table_{side}_exact.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "i","from","to","a_m","alpha_rad","alpha_deg","d_m","theta0_rad","theta0_deg",
                "joint_type","var","recon_err","rot_err_deg","trans_err_mm"
            ])
            w.writeheader()
            for row in table:
                w.writerow(row)
        print(f"[saved] {path}")

    def print_arm(self, side):
        assert side in ("left","right")
        end_joint = f"{side}_wrist_yaw_joint"
        chain = self._active_chain_to(end_joint, side_hint=(f"{side}_",))
        chain_names = [self.rmodel.names[j] for j in chain]

        print("\n" + "="*110)
        print(f"{side.upper()} ARM — active joints (proximal → distal):")
        for jn in chain_names:
            jmodel = self.rmodel.joints[self.rmodel.getJointId(jn)]
            print(f"  • {jn:28s}  [{joint_short_type(jmodel)}]")

        mats = self._pairwise_transforms_dual(chain, mode="exact")
        rows = self._dh_rows_dual(mats)

        # WORLD → first joint
        r0 = rows[0]
        print("\nWORLD → first active joint (URDF vs exact DH frame):")
        print("URDF frame:\n", r0["T_urdf"])
        print("Exact DH frame (canonical):\n", r0["T_std"])
        print(f"First-joint axis (URDF xyz): {r0['axis']}")
        print("Axis/Origin mapping A (URDF = A * canonical):\n", r0["A"])
        print(f"First joint type: {r0['jtype']}")

        # Per-link table
        print("\nPer-link DH (EXACT common-normal frames):")
        head = f"{'i':>2}  {'from':>26s}  {'to':>26s}  {'jtype':>10s}  |  {'a':>9s} {'alpha':>9s} {'d':>9s} {'theta0':>9s}   recon_frob   rot_deg   trans_mm"
        print(head); print("-"*len(head))
        for i, r in enumerate(rows[1:]):
            a, alpha, d, theta0 = r["dh_std"]
            print(f"{i:>2d}  {r['from']:>26s}  {r['to']:>26s}  {r['jtype']:>10s}  |  "
                  f"{a:9.6f} {alpha:9.6f} {d:9.6f} {theta0:9.6f}   "
                  f"{r['err_std']:10.2e} {r['err_std_ang']:8.3f} {r['err_std_mm']:10.3f}")

        # Ready-to-use table + CSV
        canon = self._canonical_table_from_rows(rows)
        self._print_markdown_table(side, canon)
        self._write_csv(side, canon)

    def _dh_fk(self, base_Tw, table_rows, q_vec):
        """FK with classic DH: base_Tw @ ∏ T_from_DH(a, α, d, q+θ0)."""
        T = base_Tw.copy()
        for i, r in enumerate(table_rows):
            a = r['a_m']; alpha = r['alpha_rad']; d = r['d_m']
            th0 = r['theta0_rad']; q = float(q_vec[i])
            T = T @ T_from_DH(a, alpha, d, q + th0)
        return T

    def validate_against_pinocchio(self, side, trials=5, jitter_deg=10.0):
        """
        Randomly jitters joint angles and checks DH FK vs Pinocchio for the last joint.
        Comparison is done in the same (canonical) frame.
        """
        end_joint = f"{side}_wrist_yaw_joint"
        chain = self._active_chain_to(end_joint, side_hint=(f"{side}_",))
        mats = self._pairwise_transforms_dual(chain, mode="exact")
        rows = self._dh_rows_dual(mats)
        base_Tw = rows[0]["T_std"]   # world→first canonical at q0
        canon = self._canonical_table_from_rows(rows)
        n = len(canon)

        # Map names to q indices
        q_idx = []
        cursor = 0
        name_set = {r['from'] for r in canon}
        for j in range(1, self.rmodel.njoints):
            nv = self.rmodel.joints[j].nv
            if nv > 0:
                name = self.rmodel.names[j]
                if name in name_set:
                    q_idx.append(cursor)
                cursor += nv
        if len(q_idx) != n:
            print("[warn] Could not align q indices cleanly; FK check skipped.")
            return

        # A_last converts URDF last joint to canonical (URDF = A * DH ⇒ ^wT_DH = ^wT_URDF · A)
        A_last = rows[-1]["A"]

        rng = np.random.default_rng(0)
        max_rot = 0.0; max_mm = 0.0
        for _ in range(trials):
            q = self.q0.copy()
            jit = (rng.uniform(-jitter_deg, jitter_deg, size=n) * np.pi/180.0)
            for k, idx in enumerate(q_idx):
                q[idx] += jit[k]

            pin.forwardKinematics(self.rmodel, self.rdata, q)
            pin.updateFramePlacements(self.rmodel, self.rdata)
            Tw_last_urdf = self.rdata.oMi[chain[-1]].homogeneous
            Tw_last_canon = Tw_last_urdf @ A_last  # world→last canonical

            T_dh = self._dh_fk(base_Tw, canon, jit)    # world→last canonical

            ef, ang, mm = _recon_error_breakdown(Tw_last_canon, T_dh)
            max_rot = max(max_rot, ang); max_mm = max(max_mm, mm)
        print(f"[DH vs Pinocchio] side={side}: max rotation err = {max_rot:.5f} deg, max translation err = {max_mm:.4f} mm")


if __name__ == "__main__":
    dh = G1_DH_Arms()
    dh.print_arm("left")
    dh.print_arm("right")
    dh.validate_against_pinocchio("left", trials=6, jitter_deg=8.0)
    dh.validate_against_pinocchio("right", trials=6, jitter_deg=8.0)
