"""
profiles.py
-----------
Task impedance + speed profiles for the G1 task-space demo.

Each profile bundles two decoupled axes:
  - COMPLIANCE : translational stiffness / damping (K_p, D_p)  -> the *visible* behavior
  - SPEED      : reference ramp caps (ref_vmax, ref_omega_max) -> how fast the arm travels

Kept FIXED across all profiles (edit here if you ever need to):
  - M_p                 (virtual translational mass)
  - K_r, D_r, M_r       (rotational impedance -- paper kept this fixed)

Design note on damping:
  D_p is chosen for a translational damping ratio zeta ~ 0.65 (with M_p = 1.5),
  using D = 2*zeta*sqrt(K*M). This keeps the stiff profile from overshooting
  and punching onto a fragile object. Recompute if you change M_p.

These three are meant to look OBVIOUSLY different to an observer:
  soft   -> yields a lot when pushed, moves slowly and gently
  medium -> standard pick-and-place feel
  stiff  -> holds pose against contact, snaps to target, moves briskly
"""

# --- Fixed across every profile -------------------------------------------
M_P_FIXED = (1.5, 1.5, 1.5)
K_R_FIXED = (3.0, 3.0, 2.0)
D_R_FIXED = (1.0, 1.0, 0.8)
M_R_FIXED = (0.2, 0.2, 0.2)

# --- The three demo profiles ----------------------------------------------
PROFILES = {
    "soft": {
        "label": "SOFT / delicate  (eggs, soft balls, cloth, plant)",
        "K_p": (2.0, 2.0, 2.0),
        "D_p": (2.5, 2.5, 2.5),
        "ref_vmax": 0.08,
        "ref_omega_max": 0.5,
        # "hand": "delicate_pinch",   # placeholder -> wire to BrainCo later
    },
    "medium": {
        "label": "MEDIUM / standard pick-place  (toys, generic rigid objects)",
        "K_p": (5.0, 5.0, 4.0),
        "D_p": (3.5, 3.5, 3.0),
        "ref_vmax": 0.15,
        "ref_omega_max": 0.8,
        # "hand": "power_grasp",
    },
    "stiff": {
        "label": "STIFF / firm contact  (bottle, surface-follow, press, poke)",
        "K_p": (10.0, 10.0, 9.0),
        "D_p": (5.0, 5.0, 4.5),
        "ref_vmax": 0.22,
        "ref_omega_max": 1.0,
        # "hand": "firm_grasp",
    },
}

DEFAULT_PROFILE = "medium"


def apply_profile(spring, name, speed_scale=1.0):
    """
    Apply a named profile to a SpringDamperCartesian instance IN PLACE.

    Mutates only the gains and ramp-speed caps. It does NOT touch the
    internal state (u_p, u_r) or the current target, so the arm keeps its
    pose and only its *behavior* changes -> no jump / lurch.

    Args:
        spring      : SpringDamperCartesian instance (must have set_gains + set_ref_speed)
        name        : one of PROFILES keys ("soft" / "medium" / "stiff")
        speed_scale : optional global multiplier on the ramp speed
                      (revival of the old 1.0 / 0.9 / 0.7 "intent" idea)

    Returns:
        The human-readable label of the applied profile.
    """
    if name not in PROFILES:
        raise KeyError(f"Unknown profile '{name}'. Options: {list(PROFILES)}")
    p = PROFILES[name]

    spring.set_gains(
        K_p=p["K_p"], D_p=p["D_p"], M_p=M_P_FIXED,
        K_r=K_R_FIXED, D_r=D_R_FIXED, M_r=M_R_FIXED,
    )
    spring.set_ref_speed(
        ref_vmax=p["ref_vmax"] * speed_scale,
        ref_omega_max=p["ref_omega_max"] * speed_scale,
    )
    return p["label"]


def describe(name):
    """Return a short one-line string describing a profile's key numbers."""
    p = PROFILES[name]
    return (f"{name:6s} | K_p={p['K_p']} D_p={p['D_p']} "
            f"| vmax={p['ref_vmax']} m/s  wmax={p['ref_omega_max']} rad/s")
