"""
run_simulations.py
=================
Full pipeline:
    1. Build occupancy grid (or load 'cspace_2000x2000.npy' if present).
    2. Run Informed RRT* to find a collision-free 2D path.
    3. Convert path waypoints to ENU metres and add cruise altitude.
    4. Run cascaded PID simulation tracking those waypoints.
    5. Save three plots automatically to the output directory.
    6. Print tracking metrics.

Output files
------------
    simulations_output/fig0_rrt_map.png
    simulations_output/fig1_3d_trajectory.png
    simulations_output/fig2_position_vs_time.png
    simulations_output/fig3_dashboard.png
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — safe for file-only output
import matplotlib.pyplot as plt
from pathlib import Path

# ── All paths relative to this script's location ─────────────────────────────
BASE_DIR = Path(__file__).parent
OUT_DIR  = BASE_DIR / "simulations_output"
OUT_DIR.mkdir(exist_ok=True)

# ── Import module from same folder ────────────────────────────────────────────
sys.path.insert(0, str(BASE_DIR))

from quad_utilss import (
    InformedRRTStar,
    WaypointTracker,
    run_rrt_pid,
    plot_results,
    compute_metrics,
    GRID_SCALE,
    GRID_SIZE,
)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — OCCUPANCY GRID
# ─────────────────────────────────────────────────────────────────────────────

GRID_NPY = BASE_DIR / "cspace_2000x2000.npy"

def build_synthetic_grid(size=GRID_SIZE, seed=42):
    """
    Generate a random occupancy grid with rectangular obstacles when the
    real cspace file is not available.

    Each cell:  0 = free,  1 = occupied.

    The start and goal regions are always kept clear.
    """
    rng  = np.random.default_rng(seed)
    grid = np.zeros((size, size), dtype=np.uint8)

    # Place 30 random rectangular obstacles
    for _ in range(30):
        r  = rng.integers(50,  300)
        c  = rng.integers(50,  size - 50)
        h  = rng.integers(20,  100)
        w  = rng.integers(20,  200)
        r0, r1 = max(0, r-h//2), min(size, r+h//2)
        c0, c1 = max(0, c-w//2), min(size, c+w//2)
        grid[r0:r1, c0:c1] = 1

    # Keep start (120,120) and goal (1200,1500) regions clear
    for cx, cy, radius in [(120, 120, 60), (1200, 1500, 60)]:
        rr, cc = np.ogrid[:size, :size]
        mask = (rr - cy)**2 + (cc - cx)**2 <= radius**2
        grid[mask] = 0

    return grid


if GRID_NPY.exists():
    print(f"[Grid] Loading {GRID_NPY}")
    grid = np.load(GRID_NPY)
else:
    print("[Grid] cspace_2000x2000.npy not found — generating synthetic grid")
    grid = build_synthetic_grid()
    np.save(OUT_DIR / "synthetic_grid.npy", grid)
    print(f"[Grid] Saved synthetic grid → {OUT_DIR / 'synthetic_grid.npy'}")

print(f"[Grid] Shape: {grid.shape}  occupied: {grid.sum()} cells  "
      f"({100*grid.mean():.1f}% fill)")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — INFORMED RRT* PATH PLANNING
# ─────────────────────────────────────────────────────────────────────────────

# All coordinates in grid cells (1 cell = GRID_SCALE metres = 1 cm)
START_GRID = np.array([120.0, 120.0])
GOAL_GRID  = np.array([1200.0, 1500.0])

RRT_ITER   = 800    # iteration budget
STEP_SIZE  = 200    # branch length [grid cells]  ≡ 2.0 m

print(f"\n[RRT*] Planning from {START_GRID} to {GOAL_GRID} (grid cells)")
print(f"       = ({START_GRID*GRID_SCALE}) m  →  ({GOAL_GRID*GRID_SCALE}) m")
print(f"       iterations={RRT_ITER}  step={STEP_SIZE} cells = {STEP_SIZE*GRID_SCALE} m")

planner = InformedRRTStar(
    start     = START_GRID,
    goal      = GOAL_GRID,
    grid      = grid,
    num_iter  = RRT_ITER,
    step_size = STEP_SIZE,
    scale     = GRID_SCALE,
)

waypoints_m = planner.plan(verbose=True)

if len(waypoints_m) == 0:
    # Fallback: straight-line path with 1 m spacing (no obstacle awareness)
    print("[RRT*] Fallback: using straight-line waypoints at 1 m spacing")
    start_m = START_GRID * GRID_SCALE
    goal_m  = GOAL_GRID  * GRID_SCALE
    n_pts   = max(5, int(np.linalg.norm(goal_m - start_m) / 1.0))
    waypoints_m = [start_m + (goal_m - start_m) * i / (n_pts - 1)
                   for i in range(n_pts)]

print(f"[RRT*] Waypoints: {len(waypoints_m)}")
for i, wp in enumerate(waypoints_m):
    print(f"        wp[{i:3d}] = ({wp[0]:.2f}, {wp[1]:.2f}) m")

# ── Save a 2D map of the planned path ────────────────────────────────────────
fig_map, ax_map = plt.subplots(figsize=(8, 8))
ax_map.imshow(grid, cmap="binary", origin="upper")
wp_arr = np.array(waypoints_m) / GRID_SCALE     # back to grid coords for plotting
ax_map.plot(wp_arr[:, 0], wp_arr[:, 1], "r-o", lw=1.5, ms=4, label="RRT* path")
ax_map.scatter(START_GRID[0], START_GRID[1], c="lime",   s=100, zorder=5, label="Start")
ax_map.scatter(GOAL_GRID[0],  GOAL_GRID[1],  c="orange", s=100, zorder=5, label="Goal")
ax_map.set_title("Informed RRT* Planned Path (grid units)")
ax_map.set_xlabel("X [grid cells]"); ax_map.set_ylabel("Y [grid cells]")
ax_map.legend(fontsize=9)
map_path = OUT_DIR / "fig0_rrt_map.png"
fig_map.savefig(str(map_path), dpi=150, bbox_inches="tight")
plt.close(fig_map)
print(f"[Plot] Saved → {map_path}")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — SIMULATION CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# ── Physical parameters ────────────────────────────────────
quadcopter = {
    "m"  : 0.468,        # mass [kg]
    "g"  : 9.81,         # gravity [m/s²]
    "l"  : 0.225,        # arm length [m]
    "kT" : 2.98e-6,      # thrust coefficient [N/(rad/s)²]
    "kQ" : 0.0382,       # torque coefficient [N.m/(rad/s)²]
    "Ixx": 4.856e-3,     # roll  inertia [kg.m²]
    "Iyy": 4.856e-3,     # pitch inertia [kg.m²]
    "Izz": 8.801e-3,     # yaw   inertia [kg.m²]
    "Ax" : 0.30,         # aero drag coefficient (x)
    "Ay" : 0.30,         # aero drag coefficient (y)
    "Az" : 0.25,         # aero drag coefficient (z)
    "Ar" : 0.20,         # rotational drag coefficient
    "IR" : 3.357e-5,     # rotor inertia [kg.m²]
}

# ── PID gains ─────────────────────────────────────────────────────────────────
controller = {
    # Inner loop — angular rate (200 Hz)
    "Kp_p": 0.80,  "Ki_p": 0.0001,  "Kd_p": 0.010,
    "Kp_q": 0.80,  "Ki_q": 0.0001,  "Kd_q": 0.010,
    "Kp_r": 0.60,  "Ki_r": 0.0001,  "Kd_r": 0.008,

    # Middle loop — attitude (100 Hz)
    "Kp_phi"  : 6.0,  "Ki_phi"  : 0.0001,  "Kd_phi"  : 0.80,  "AW_phi"  : 0.5,
    "Kp_theta": 6.0,  "Ki_theta": 0.0001,  "Kd_theta": 0.80,  "AW_theta": 0.5,
    "Kp_psi"  : 4.0,  "Ki_psi"  : 0.0000001,  "Kd_psi" : 0.40,  "AW_psi"  : 0.5,

    # Outer loop — position (50 Hz)
    "Kp_x": 0.90,  "Ki_x": 0.02,  "Kd_x": 1.20,  "AW_x": 0.5,
    "Kp_y": 1.00,  "Ki_y": 0.02,  "Kd_y": 1.40,  "AW_y": 0.5,
    "Kp_z": 4.00,  "Ki_z": 1.50,  "Kd_z": 2.00,  "AW_z": 3.0,

    # Velocity feedforward
    "Kff_x": 0.75,  "Kff_y": 0.80,

    # Limits
    "N_rate"        : 50,
    "omega_max"     : 1000.0,      # max rotor speed [rad/s]
    "T_max_factor"  : 4,
    "T_min"         : 0.0,
    "att_cmd_limit" : 0.40,        # ~23 deg
    "yaw_rate_limit": 1.0,
}

# ── Simulation and waypoint-tracking parameters ───────────────────────────────
path_m    = sum(np.linalg.norm(waypoints_m[i+1] - waypoints_m[i])
                for i in range(len(waypoints_m)-1))
avg_speed = 0.7      # conservative estimate [m/s] (controller limited)
t_cruise  = path_m / avg_speed
t_settle  = 5.0      # extra settling time at goal [s]
t_takeoff = 8.0
tf        = t_takeoff + t_cruise + t_settle

print(f"\n[Config] Path length = {path_m:.2f} m")
print(f"[Config] Estimated cruise time = {t_cruise:.1f} s")
print(f"[Config] tf = {tf:.1f} s")

simulation = {
    "dt"           : 5e-3,       # master timestep [s]
    "tf"           : tf,         # total sim time  [s]
    "z_ref"        : 1.5,        # cruise altitude [m]
    "switch_radius": 1.5,        # waypoint switch tolerance [m]
    "altitude_tol" : 0.10,       # takeoff completion tolerance [m]
    "t_takeoff"    : t_takeoff,  # min takeoff phase duration [s]
    "cruise_speed" : 0.7,        # feedforward speed [m/s]
}

config = {
    "quad"      : quadcopter,
    "controller": controller,
    "sim"       : simulation,
}

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — RUN SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  RUNNING CASCADED PID WAYPOINT-TRACKING SIMULATION")
print("="*60)

quad, tracker = run_rrt_pid(config, waypoints_m, verbose=True)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — SAVE PLOTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[Plot] Generating and saving figures ...")
p1, p2, p3 = plot_results(quad, tracker, save_dir=str(OUT_DIR))

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — METRICS
# ─────────────────────────────────────────────────────────────────────────────

t_hist = np.array(quad.ref_history["t"])
x_hist = np.array(quad.ref_history["xa"])
y_hist = np.array(quad.ref_history["ya"])
z_hist = np.array(quad.ref_history["za"])

metrics = compute_metrics(quad, tracker)

print("\n" + "="*50)
print("   CRUISE TRACKING LAG (Active Flight)")
print("="*50)
print(f"  (Window: t=8.0s → {metrics['t_arrival']:.1f}s)")
print(f"  RMSE X (Lag)   : {metrics['rmse_cruise_x']:.4f} m")
print(f"  RMSE Y (Lag)   : {metrics['rmse_cruise_y']:.4f} m")
print(f"  RMSE Z (Lag)   : {metrics['rmse_cruise_z']:.4f} m")
print(f"  RMSE 3D (Lag)  : {metrics['rmse_cruise_xyz']:.4f} m")

print("\n" + "="*50)
print("   HOVER STABILITY (After Arrival)")
print("="*50)
print(f"  (Window: t={metrics['t_arrival']:.1f}s → {t_hist[-1]:.1f}s)")
print(f"  RMSE 3D (Hover): {metrics['rmse_arrived_xyz']:.4f} m")
print(f"  Final 3D error : {metrics['final_error']:.4f} m")
print("="*50)

print(f"\n  Goal (m)  : ({tracker.waypoints[-1][0]:.2f}, "
      f"{tracker.waypoints[-1][1]:.2f}, {tracker.z_ref:.2f})")
print(f"  Final pos : ({x_hist[-1]:.2f}, {y_hist[-1]:.2f}, {z_hist[-1]:.2f})")

print(f"\n[Done] All outputs saved to: {OUT_DIR}")
