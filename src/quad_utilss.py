"""
quad_utilss.py
========================
Integrated module: quadcopter dynamics, cascaded PID controller, and
Informed RRT* path planner.

Integration design
------------------
Frames and units
  - RRT* plans on a 2D pixel grid (shape 2000×2000).
    Physical scale: 1 grid cell = 1 cm  →  scale = 0.01 m/cell.
  - Controller operates in inertial ENU metres.
  - Conversion: x_m = grid_x * GRID_SCALE,  y_m = grid_y * GRID_SCALE.
  - Altitude: RRT* is 2D.  Every waypoint is assigned z = z_ref.

Waypoint interface
  - InformedRRTStar.plan() returns a list of np.array([x_m, y_m]) in metres.
  - WaypointTracker wraps that list, adds z_ref, and feeds the triple
    (Xd, Yd, Zd) directly into OuterPositionPID.compute().
  - Switching criterion: Euclidean distance (3D) < tolerance_radius.

Timing
  - Master dt = 5 ms  (200 Hz inner rate loop).
  - Outer loop fires every 20 ms (50 Hz), middle every 10 ms (100 Hz).
"""

import numpy as np
import random
import c4dynamics as c4d
#from scipy.integrate import solve_ivp
from c4dynamics.rotmat import dcm321

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

GRID_SCALE = 0.01          # metres per grid cell  (1 cell ≡ 1 cm)
GRID_SIZE  = 2000          # grid is 2000 × 2000 cells


# ─────────────────────────────────────────────────────────────────────────────
#  DYNAMICS
# ─────────────────────────────────────────────────────────────────────────────

def dynamics(t, y, quad, rotor_speeds):
    """
    Full 12-state quadcopter dynamics in ENU inertial frame.

    State  X = [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]

    Body frame (right-handed):
      x → forward,  y → right,  z → down

    Inertial frame (ENU):
      x → east,  y → north,  z → up

    ZYX (yaw-pitch-roll) Euler rotation convention.

    Motor layout — X configuration:
      w1: front CCW (+)  w2: rear  CCW (+)
      w3: left  CW  (-)  w4: right CW  (-)

    Parameters
    ----------
    t            : float — current time [s]
    y            : array (12,) — current state
    quad         : c4d.rigidbody with physical parameters attached
    rotor_speeds : array [w1, w2, w3, w4] in rad/s

    Returns
    -------
    dX : array (12,)
    """
    x, y_, z, vx, vy, vz, phi, theta, psi, p, q, r = y
    w1, w2, w3, w4 = rotor_speeds

    m   = quad.m
    g   = quad.g
    L   = quad.l
    kT  = quad.kT
    kM  = quad.kQ
    IR  = quad.IR
    Ixx = quad.Ixx
    Iyy = quad.Iyy
    Izz = quad.Izz
    Ax  = quad.Ax
    Ay  = quad.Ay
    Az  = quad.Az
    Ar  = quad.Ar

    gamma = kM / kT

    F1 = kT * w1**2
    F2 = kT * w2**2
    F3 = kT * w3**2
    F4 = kT * w4**2

    T     =          F1 + F2 + F3 + F4
    tau_x =    L * (-F1 + F2 + F3 - F4)
    tau_y =     L * (F1 - F2 + F3 - F4)
    tau_z = gamma * (F1 + F2 - F3 - F4)

    Omega = w1 + w2 - w3 - w4       # net rotor speed for gyroscopic coupling

    # ── Rotational kinematics ─────────────────────────────────────────────────
    dphi   = p + np.sin(phi)*np.tan(theta)*q + np.cos(phi)*np.tan(theta)*r
    dtheta =                   np.cos(phi)*q -               np.sin(phi)*r
    dpsi   =   np.sin(phi)/np.cos(theta)*q  + np.cos(phi)/np.cos(theta)*r

    # ── Angular accelerations (Euler + aero drag + gyro coupling) ─────────────
    Mx = tau_x - Ar*p - IR*q*Omega
    My = tau_y - Ar*q + IR*p*Omega
    Mz = tau_z - Ar*r

    dp = (Mx - (Izz - Iyy)*q*r) / Ixx
    dq = (My - (Ixx - Izz)*p*r) / Iyy
    dr = (Mz - (Iyy - Ixx)*p*q) / Izz

    # ── Translational dynamics ────────────────────────────────────────────────
    # Rotation matrix: body ← inertial, with z-up body convention
    BI = dcm321(phi, theta, psi) @ dcm321(phi=np.pi)
    T  = -T           # thrust positive in z-up body frame

    # Velocity in body frame
    u, v, w = BI @ np.array([vx, vy, vz])

    # Thrust + aero drag in body frame
    Fb = np.array([-Ax*u, -Ay*v, T - Az*w])

    # Rotate forces back to inertial frame
    Fi = BI.T @ Fb

    dx  = vx
    dy_ = vy
    dz  = vz

    dvx, dvy, dvz = Fi / m
    dvz -= g      # gravity in inertial frame (z upward)

    return np.array([dx, dy_, dz, dvx, dvy, dvz, dphi, dtheta, dpsi, dp, dq, dr])

# ─────────────────────────────────────────────────────────────────────────────
#  RK4 INTEGRATOR 
# ─────────────────────────────────────────────────────────────────────────────

def rk4_step(t, y, quad, rotor_speeds, dt):
    """
    Fixed-step 4th-order Runge-Kutta integrator.
    
    Parameters
    ----------
    t            : float — current time [s]
    y            : array (12,) — current state
    quad         : c4d.rigidbody with physical parameters
    rotor_speeds : array [w1, w2, w3, w4] in rad/s
    dt           : float — timestep [s]
    
    Returns
    -------
    y_new : array (12,) — next state
    """
    k1 = dynamics(t,          y,                quad, rotor_speeds)
    k2 = dynamics(t + dt/2,   y + (dt/2) * k1,  quad, rotor_speeds)
    k3 = dynamics(t + dt/2,   y + (dt/2) * k2,  quad, rotor_speeds)
    k4 = dynamics(t + dt,     y + dt      * k3,  quad, rotor_speeds)
    return y + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


# ─────────────────────────────────────────────────────────────────────────────
#  CASCADE PID CONTROLLERS
# ─────────────────────────────────────────────────────────────────────────────

class OuterPositionPID:
    """
    Outer loop: position → desired angles + total thrust.
    Nominal rate: 50 Hz.
    """

    def __init__(self, params, m, g, kT):
        self.g = g
        self.m = m

        self.KP_Z = params["Kp_z"];  self.KI_Z = params["Ki_z"];  self.KD_Z = params["Kd_z"]
        self.KP_X = params["Kp_x"];  self.KI_X = params["Ki_x"];  self.KD_X = params["Kd_x"]
        self.KP_Y = params["Kp_y"];  self.KI_Y = params["Ki_y"];  self.KD_Y = params["Kd_y"]

        self.AW_Z = params["AW_z"];  self.AW_X = params["AW_x"];  self.AW_Y = params["AW_y"]

        self.T_max = params["T_max_factor"] * kT * params["omega_max"]**2
        self.T_min = params["T_min"]
        self.att_cmd_limit = params["att_cmd_limit"]

        self.FF_X = params["Kff_x"]
        self.FF_Y = params["Kff_y"]

        self.int_Z = self.int_X = self.int_Y = 0.0

    def compute(self, Xd, Yd, Zd, Vxd, Vyd, Psi_sp, quad, Ts):
        """
        Parameters
        ----------
        Xd, Yd, Zd : reference position in ENU metres
        Vxd, Vyd   : feedforward velocity in ENU m/s  (set to 0 for waypoints)
        Psi_sp      : desired yaw [rad]
        quad        : c4d.rigidbody
        Ts          : sample time [s]

        Returns
        -------
        T_cmd, phi_d, theta_d, psi_d
        """
        x, y, z   = quad.x, quad.y, quad.z
        vx, vy, vz = quad.vx, quad.vy, quad.vz
        phi, theta, psi = quad.phi, quad.theta, quad.psi

        BI = dcm321(phi, theta, psi) @ dcm321(phi=np.pi)
        HE = dcm321(psi=psi) @ dcm321(phi=np.pi)

        Tmin     = -self.T_max
        Tmax     = self.T_min
        Tfactor  = -1
        pitch_factor = -1
        phi_factor   =  1

        # Position errors (inertial frame)
        e_X = Xd - x
        e_Y = Yd - y
        e_Z = Zd - z

        # ── Altitude PID ──────────────────────────────────────────────────────
        self.int_Z = np.clip(self.int_Z + Ts*e_Z, -self.AW_Z, self.AW_Z)
        az_cmd     = self.KP_Z*e_Z + self.KI_Z*self.int_Z + self.KD_Z*(-vz)

        Tcmd_b = BI @ [0, 0, self.m*(self.g + az_cmd)]
        T_cmd  = Tfactor * np.clip(Tcmd_b[2], Tmin, Tmax)

        # ── Horizontal PID (errors rotated to body frame) ─────────────────────
        Xerr_b = HE @ [e_X, e_Y, 0]
        Vb     = HE @ [vx, vy, 0]

        self.int_X = np.clip(self.int_X + Ts*Xerr_b[0], -self.AW_X, self.AW_X)
        self.int_Y = np.clip(self.int_Y + Ts*Xerr_b[1], -self.AW_Y, self.AW_Y)

        Vff_b    = HE @ [Vxd, Vyd, 0]
        ff_theta = self.FF_X * Vff_b[0]
        ff_phi   = -self.FF_Y * Vff_b[1]

        theta_d = np.clip(
            pitch_factor * (self.KP_X*Xerr_b[0] - self.KP_X*Vb[0]
                            + self.KI_X*self.int_X
                            + self.KD_X*(Vff_b[0] - Vb[0])
                            + ff_theta),
            -self.att_cmd_limit, self.att_cmd_limit)

        phi_d = np.clip(
            phi_factor * (self.KP_Y*Xerr_b[1] - self.KP_Y*Vb[1]
                          + self.KI_Y*self.int_Y
                          + self.KD_Y*(Vff_b[1] - Vb[1])
                          + ff_phi),
            -self.att_cmd_limit, self.att_cmd_limit)

        return T_cmd, phi_d, theta_d, Psi_sp


class MiddleAttitudePID:
    """
    Middle loop: desired angles → desired body rates.
    Nominal rate: 100 Hz.
    """

    def __init__(self, params):
        self.KP_phi   = params["Kp_phi"];   self.KI_phi   = params["Ki_phi"];   self.KD_phi   = params["Kd_phi"]
        self.KP_theta = params["Kp_theta"]; self.KI_theta = params["Ki_theta"]; self.KD_theta = params["Kd_theta"]
        self.KP_psi   = params["Kp_psi"];   self.KI_psi   = params["Ki_psi"];   self.KD_psi   = params["Kd_psi"]

        self.AW_phi   = params["AW_phi"]
        self.AW_theta = params["AW_theta"]
        self.AW_psi   = params["AW_psi"]
        self.yaw_rate_limit = params["yaw_rate_limit"]

        self.int_phi = self.int_theta = self.int_psi = 0.0

    def compute(self, phi_d, theta_d, psi_d, quad, Ts):
        e_phi   = phi_d   - quad.phi
        e_theta = theta_d - quad.theta
        e_psi   = np.arctan2(np.sin(psi_d - quad.psi), np.cos(psi_d - quad.psi))

        self.int_phi   = np.clip(self.int_phi   + Ts*e_phi,
                                 -self.AW_phi   / self.KI_phi,
                                  self.AW_phi   / self.KI_phi)
        self.int_theta = np.clip(self.int_theta + Ts*e_theta,
                                 -self.AW_theta / self.KI_theta,
                                  self.AW_theta / self.KI_theta)
        self.int_psi = np.clip(self.int_psi + Ts*e_psi, -1.0, 1.0)

        rl  = self.yaw_rate_limit * 3
        p_d = np.clip(self.KP_phi*e_phi     + self.KI_phi*self.int_phi     - self.KD_phi*quad.p,   -rl, rl)
        q_d = np.clip(self.KP_theta*e_theta + self.KI_theta*self.int_theta - self.KD_theta*quad.q, -rl, rl)
        r_d = np.clip(self.KP_psi*e_psi     + self.KI_psi*self.int_psi     - self.KD_psi*quad.r,
                      -self.yaw_rate_limit, self.yaw_rate_limit)

        return p_d, q_d, r_d


class InnerRatePID:
    """
    Inner loop: desired body rates → torque commands.
    Nominal rate: 200 Hz (every master timestep).
    """

    def __init__(self, params, Ixx, Iyy, Izz, L, kT):
        self.KP_p = params["Kp_p"]; self.KI_p = params["Ki_p"]; self.KD_p = params["Kd_p"]
        self.KP_q = params["Kp_q"]; self.KI_q = params["Ki_q"]; self.KD_q = params["Kd_q"]
        self.KP_r = params["Kp_r"]; self.KI_r = params["Ki_r"]; self.KD_r = params["Kd_r"]

        self.N_rate = params["N_rate"]
        self.M_max  = L * kT * params["omega_max"]**2

        self.Ixx = Ixx; self.Iyy = Iyy; self.Izz = Izz

        self.int_p = self.int_q = self.int_r = 0.0
        self.ep_prev = self.eq_prev = self.er_prev = 0.0

    def compute(self, p_d, q_d, r_d, quad, Ts):
        ep = p_d - quad.p
        eq = q_d - quad.q
        er = r_d - quad.r

        # Tustin integrator with clamp
        self.int_p = np.clip(self.int_p + (Ts/2) * (ep + self.ep_prev), -0.5, 0.5)
        self.int_q = np.clip(self.int_q + (Ts/2) * (eq + self.eq_prev), -0.5, 0.5)
        self.int_r = np.clip(self.int_r + (Ts/2) * (er + self.er_prev), -0.5, 0.5)

        # Filtered derivative
        d  = 1 + self.N_rate * Ts
        dp = self.N_rate * (ep - self.ep_prev) / d
        dq = self.N_rate * (eq - self.eq_prev) / d
        dr = self.N_rate * (er - self.er_prev) / d

        tau_phi_raw   = self.Ixx * (self.KP_p*ep + self.KI_p*self.int_p + self.KD_p*dp)
        tau_theta_raw = self.Iyy * (self.KP_q*eq + self.KI_q*self.int_q + self.KD_q*dq)
        tau_psi_raw   = self.Izz * (self.KP_r*er + self.KI_r*self.int_r + self.KD_r*dr)

        tau_phi   = np.clip(tau_phi_raw,   -self.M_max, self.M_max)
        tau_theta = np.clip(tau_theta_raw, -self.M_max, self.M_max)
        tau_psi   = np.clip(tau_psi_raw,   -self.M_max, self.M_max)

        # Back-calculation anti-windup
        AW = 0.1
        self.int_p += AW * (tau_phi   - tau_phi_raw)   / (self.Ixx * self.KI_p + 1e-9)
        self.int_q += AW * (tau_theta - tau_theta_raw) / (self.Iyy * self.KI_q + 1e-9)
        self.int_r += AW * (tau_psi   - tau_psi_raw)   / (self.Izz * self.KI_r + 1e-9)

        self.ep_prev = ep; self.eq_prev = eq; self.er_prev = er

        return tau_phi, tau_theta, tau_psi


class ControlAllocator:
    """
    Thrust + torques → individual rotor speeds.

    X-configuration:
      w1 front CCW | w2 rear CCW | w3 left CW | w4 right CW
    """

    def __init__(self, kT, kQ, L, omega_max):
        self.kT     = kT
        self.kQ     = kQ
        self.L      = L
        self.sq_min = 0.0
        self.sq_max = omega_max**2

    def allocate(self, T_cmd, tau_phi, tau_theta, tau_psi):
        gamma = self.kQ / self.kT
        A1 = np.array([[1,-1, 1, 1],
                        [1, 1,-1, 1],
                        [1, 1, 1,-1],
                        [1,-1,-1,-1]]) / 4
        A2 = np.diag([1, 1/self.L, 1/self.L, 1/gamma])
        F  = A1 @ A2 @ np.array([T_cmd, tau_phi, tau_theta, tau_psi])

        w1 = np.sqrt(np.clip(F[0]/self.kT, self.sq_min, self.sq_max))
        w2 = np.sqrt(np.clip(F[1]/self.kT, self.sq_min, self.sq_max))
        w3 = np.sqrt(np.clip(F[2]/self.kT, self.sq_min, self.sq_max))
        w4 = np.sqrt(np.clip(F[3]/self.kT, self.sq_min, self.sq_max))

        return w1, w2, w3, w4


# ─────────────────────────────────────────────────────────────────────────────
#  INFORMED RRT*
# ─────────────────────────────────────────────────────────────────────────────

class _TreeNode:
    """Single node in the RRT* search tree."""
    __slots__ = ("locationX", "locationY", "children", "parent")

    def __init__(self, x, y):
        self.locationX = x
        self.locationY = y
        self.children  = []
        self.parent    = None


class InformedRRTStar:
    """
    Informed RRT* planner operating on a 2D occupancy grid.

    The planner works entirely in grid-cell units.
    Call plan() to run the search; the returned waypoints are converted
    to metric ENU metres before being returned.

    Parameters
    ----------
    start         : array [col, row] in grid cells
    goal          : array [col, row] in grid cells
    grid          : 2D np.ndarray  (1 = obstacle, 0 = free)
    num_iter      : maximum number of iterations
    step_size     : branch length in grid cells
    scale         : metres per grid cell  (default GRID_SCALE = 0.01)
    """

    def __init__(self, start, goal, grid, num_iter=800, step_size=200,
                 scale=GRID_SCALE):
        self.start    = np.asarray(start, dtype=float)
        self.goal     = np.asarray(goal,  dtype=float)
        self.grid     = grid
        self.rho      = step_size
        self.iters    = min(num_iter, 800)
        self.scale    = scale

        # Ellipse geometry for informed sampling
        self.c_min          = np.linalg.norm(self.goal - self.start)
        self.ellipse_angle  = np.arctan2(goal[1]-start[1], goal[0]-start[0])
        self.cx             = 0.5*(start[0] + goal[0])
        self.cy             = 0.5*(start[1] + goal[1])

        # Search radius for rewire neighbourhood
        self.search_radius = self.rho * 2

        # Tree root
        self._root          = _TreeNode(start[0], start[1])
        self._goal_node     = _TreeNode(goal[0],  goal[1])
        self._nearest       = None
        self._nearest_dist  = 1e9
        self._neighbours    = []
        self._goal_costs    = [1e9]  
        self._path_found    = False

        # Waypoints (grid units) — populated after first path
        self.waypoints_grid = []

    # ── private helpers ───────────────────────────────────────────────────────

    def _sample(self):
        """Uniform random sample within grid bounds."""
        return np.array([random.randint(1, self.grid.shape[1]-1),
                         random.randint(1, self.grid.shape[0]-1)], dtype=float)

    def _in_ellipse(self, px, py, c_best):
        """Return True if (px,py) lies strictly inside the current best ellipse."""
        rx = c_best / 2
        ry_sq = c_best**2 - self.c_min**2
        if ry_sq <= 0:
            return False
        ry = np.sqrt(ry_sq) / 2
        ca, sa = np.cos(-self.ellipse_angle), np.sin(-self.ellipse_angle)
        dx, dy = px - self.cx, py - self.cy
        return ((dx*ca + dy*sa)**2 / rx**2 + (-dx*sa + dy*ca)**2 / ry**2) < 1.0

    def _distance(self, node, point):
        return np.sqrt((node.locationX - point[0])**2 + (node.locationY - point[1])**2)

    def _unit_vector(self, node, point):
        v = np.array([point[0] - node.locationX, point[1] - node.locationY])
        n = np.linalg.norm(v)
        return v / max(n, 1.0)

    def _steer(self, from_node, to_point):
        """Step from_node toward to_point by rho cells, clamped to grid."""
        offset = self.rho * self._unit_vector(from_node, to_point)
        p = np.array([from_node.locationX + offset[0],
                      from_node.locationY + offset[1]])
        p[0] = np.clip(p[0], 0, self.grid.shape[1] - 1)
        p[1] = np.clip(p[1], 0, self.grid.shape[0] - 1)
        return p

    def _in_obstacle(self, node_a, point_b):
        """
        Walk along the segment node_a → point_b and return True if any
        intermediate grid cell is occupied.
        """
        u_hat = self._unit_vector(node_a, point_b)
        dist  = int(self._distance(node_a, point_b))
        for i in range(dist):
            cx = int(np.clip(round(node_a.locationX + i*u_hat[0]), 0, self.grid.shape[1]-1))
            cy = int(np.clip(round(node_a.locationY + i*u_hat[1]), 0, self.grid.shape[0]-1))
            if self.grid[cy, cx] == 1:
                return True
        return False

    def _find_nearest(self, root, point):
        """Recursive DFS to find the nearest tree node to point."""
        if root is None:
            return
        d = self._distance(root, point)
        if d <= self._nearest_dist and root.locationX != self._goal_node.locationX:
            self._nearest      = root
            self._nearest_dist = d
        for child in root.children:
            self._find_nearest(child, point)

    def _find_neighbours(self, root, point):
        """Recursive DFS collecting nodes within search_radius of point."""
        if root is None:
            return
        if self._distance(root, point) <= self.search_radius:
            self._neighbours.append(root)
        for child in root.children:
            self._find_neighbours(child, point)

    def _path_cost(self, node):
        """Cost (total branch length) from root to node."""
        cost = 0.0
        cur  = node
        while cur.locationX != self._root.locationX or cur.locationY != self._root.locationY:
            cost += self._distance(cur, np.array([cur.parent.locationX, cur.parent.locationY]))
            cur   = cur.parent
        return cost

    def _add_child(self, new_node):
        """Attach new_node (or goal) to the current nearest node."""
        if (new_node.locationX == self._goal_node.locationX and
                new_node.locationY == self._goal_node.locationY):
            self._nearest.children.append(self._goal_node)
            self._goal_node.parent = self._nearest
        else:
            self._nearest.children.append(new_node)
            new_node.parent = self._nearest

    def _retrace(self):
        """Walk from goal back to start, building self.waypoints_grid."""
        self.waypoints_grid = []
        node = self._goal_node
        cost = 0.0
        while node.locationX != self._root.locationX or node.locationY != self._root.locationY:
            self.waypoints_grid.insert(0, np.array([node.locationX, node.locationY]))
            cost += self._distance(node, np.array([node.parent.locationX, node.parent.locationY]))
            node  = node.parent
        self._goal_costs.append(cost)

    def _reset(self):
        self._nearest      = None
        self._nearest_dist = 1e9
        self._neighbours   = []

    # ── public interface ──────────────────────────────────────────────────────

    def plan(self, verbose=True):
        """
        Run the Informed RRT* search.

        Returns
        -------
        waypoints_m : list of np.array([x_m, y_m]) in ENU metres (XY only).
                      The start point is prepended; goal appended.
                      Returns [] if no path is found within the iteration budget.
        """
        for iteration in range(self.iters):
            self._reset()

            # ── Sampling: full grid or informed ellipse ───────────────────────
            if self._path_found:
                c_best = self._goal_costs[-1]
                point  = self._sample()
                while not self._in_ellipse(point[0], point[1], c_best):
                    point = self._sample()
            else:
                point = self._sample()

            # ── Find nearest and steer ────────────────────────────────────────
            self._find_nearest(self._root, point)
            if self._nearest is None:
                continue
            new_pt = self._steer(self._nearest, point)

            # ── Skip if new_pt is in collision ────────────────────────────────
            if self._in_obstacle(self._nearest, new_pt):
                continue

            # ── Find neighbourhood and select minimum-cost parent ─────────────
            self._find_neighbours(self._root, new_pt)
            min_node = self._nearest
            min_cost = self._path_cost(min_node) + self._distance(self._nearest, new_pt)

            for v in self._neighbours:
                c = self._path_cost(v) + self._distance(v, new_pt)
                if not self._in_obstacle(v, new_pt) and c < min_cost:
                    min_node = v
                    min_cost = c

            # ── Insert new node ───────────────────────────────────────────────
            self._nearest = min_node
            new_node = _TreeNode(new_pt[0], new_pt[1])
            self._add_child(new_node)

            # ── Rewire: offer shorter parent to neighbours via new_node ───────
            for v in self._neighbours:
                rewire_cost = min_cost + self._distance(v, new_pt)
                if (not self._in_obstacle(v, new_pt) and
                        rewire_cost < self._path_cost(v)):
                    v.parent = new_node

            # ── Goal check ───────────────────────────────────────────────────
            new_node_pt = np.array([new_node.locationX, new_node.locationY])
            if self._distance(self._goal_node, new_node_pt) <= self.rho:
                projected = (self._path_cost(new_node) +
                             self._distance(self._goal_node, new_node_pt))
                if projected < self._goal_costs[-1]:
                    self._nearest    = new_node
                    self._path_found = True
                    self._add_child(self._goal_node)
                    self._retrace()
                    if verbose:
                        print(f"  [RRT*] iter {iteration:4d}  cost = {self._goal_costs[-1]:.1f} cells")

        if not self._path_found:
            print("[RRT*] WARNING: no path found within iteration budget.")
            return []

        # ── Convert waypoints to metric ENU ──────────────────────────────────
        # Grid axes: col → ENU-x,  row → ENU-y
        waypoints_m = [np.array([pt[0] * self.scale, pt[1] * self.scale])
                       for pt in self.waypoints_grid]

        # Prepend start, ensure goal is at the end
        start_m = self.start * self.scale
        goal_m  = self.goal  * self.scale
        if len(waypoints_m) == 0 or not np.allclose(waypoints_m[0], start_m):
            waypoints_m.insert(0, start_m)
        if not np.allclose(waypoints_m[-1], goal_m):
            waypoints_m.append(goal_m)

        if verbose:
            print(f"[RRT*] Final path: {len(waypoints_m)} waypoints, "
                  f"best cost = {self._goal_costs[-1]:.1f} cells "
                  f"({self._goal_costs[-1]*self.scale:.2f} m)")

        return waypoints_m


# ─────────────────────────────────────────────────────────────────────────────
#  WAYPOINT TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class WaypointTracker:
    """
    Converts a sequence of 2D metric waypoints into 3D position references
    that feed directly into OuterPositionPID.compute().

    Behaviour
    ---------
    - Phase 1  (takeoff): hold (start_x, start_y) while climbing to z_ref.
      Takeoff completes when |z - z_ref| < altitude_tol.
    - Phase 2  (cruise) : advance through waypoints.
      Switch when 3D distance to current waypoint < switch_radius.
    - Phase 3  (arrived): hold final waypoint indefinitely.

    Parameters
    ----------
    waypoints_m    : list of np.array([x, y]) in ENU metres
    z_ref          : cruise altitude [m]
    switch_radius  : 3D waypoint-switching tolerance [m]  
    altitude_tol   : takeoff completion tolerance [m]     
    t_takeoff      : minimum takeoff time [s]             
    cruise_speed   : desired cruise speed for feedforward [m/s]
    """

    def __init__(self, waypoints_m, z_ref,
                 switch_radius=1.5, altitude_tol=0.10, t_takeoff=6.0,
                 cruise_speed=0.7):
        if len(waypoints_m) < 2:
            raise ValueError("Need at least 2 waypoints (start + goal).")

        self.waypoints     = waypoints_m   # list of [x_m, y_m]
        self.z_ref         = z_ref
        self.switch_radius = switch_radius
        self.altitude_tol  = altitude_tol
        self.t_takeoff     = t_takeoff
        self.cruise_speed  = cruise_speed

        self.wp_idx        = 0            # index of the current target waypoint
        self.phase         = "takeoff"    # "takeoff" | "cruise" | "arrived"
        self._elapsed      = 0.0

    @property
    def current_waypoint_3d(self):
        """Return (x_d, y_d, z_d) of the current active waypoint."""
        wp = self.waypoints[self.wp_idx]
        return wp[0], wp[1], self.z_ref

    def _compute_feedforward(self):
        """
        Compute feedforward velocity based on current and next waypoint.
        Returns (vx_ff, vy_ff) in m/s.
        """
        if self.phase != "cruise":
            return 0.0, 0.0
        
        # If we're at the last waypoint, no feedforward
        if self.wp_idx >= len(self.waypoints) - 1:
            return 0.0, 0.0
        
        current_wp = self.waypoints[self.wp_idx]
        next_wp    = self.waypoints[self.wp_idx + 1]
        
        direction = np.array([next_wp[0] - current_wp[0],
                              next_wp[1] - current_wp[1]])
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return 0.0, 0.0
        
        unit_dir = direction / distance
        # Feedforward velocity magnitude = cruise_speed
        vx_ff = unit_dir[0] * self.cruise_speed
        vy_ff = unit_dir[1] * self.cruise_speed
        
        return vx_ff, vy_ff

    def update(self, quad, dt):
        """
        Advance tracker state and return the current 3D setpoint and feedforward velocities.

        Parameters
        ----------
        quad : c4d.rigidbody — current state
        dt   : master timestep [s]

        Returns
        -------
        x_d, y_d, z_d : float — position setpoint for outer PID
        vx_ff, vy_ff  : float — feedforward velocities for outer PID
        """
        self._elapsed += dt
        pos = np.array([quad.x, quad.y, quad.z])

        if self.phase == "takeoff":
            # Hold XY at start, climb to z_ref
            x_d, y_d = self.waypoints[0]
            z_d       = self.z_ref
            # Transition when altitude within tolerance AND minimum time elapsed
            if (abs(quad.z - self.z_ref) < self.altitude_tol and
                    self._elapsed >= self.t_takeoff):
                self.phase  = "cruise"
                self.wp_idx = min(1, len(self.waypoints) - 1)
                print(f"[Tracker] Takeoff complete at t={self._elapsed:.1f}s  "
                      f"z={quad.z:.3f}m → entering cruise phase")
            vx_ff, vy_ff = 0.0, 0.0

        elif self.phase == "cruise":
            x_d, y_d, z_d = self.current_waypoint_3d
            target_3d     = np.array([x_d, y_d, z_d])
            dist          = np.linalg.norm(pos - target_3d)

            if dist < self.switch_radius:
                if self.wp_idx < len(self.waypoints) - 1:
                    self.wp_idx += 1
                    x_d, y_d, z_d = self.current_waypoint_3d
                    print(f"[Tracker] Switched to waypoint {self.wp_idx}/{len(self.waypoints)-1}  "
                          f"→ ({x_d:.2f}, {y_d:.2f}, {z_d:.2f}) m")
                else:
                    self.phase = "arrived"
                    print(f"[Tracker] Final waypoint reached at t={self._elapsed:.1f}s")
            
            # Compute feedforward velocities based on current segment
            vx_ff, vy_ff = self._compute_feedforward()

        else:  # arrived: hold final waypoint
            x_d, y_d, z_d = self.current_waypoint_3d
            vx_ff, vy_ff = 0.0, 0.0

        return x_d, y_d, z_d, vx_ff, vy_ff

    @property
    def done(self):
        return self.phase == "arrived"

    def waypoints_as_xyz(self, z_ref=None):
        """
        Return all waypoints as (N, 3) array at z_ref altitude.
        Useful for plotting the planned path.
        """
        z = z_ref if z_ref is not None else self.z_ref
        return np.array([[wp[0], wp[1], z] for wp in self.waypoints])


# ─────────────────────────────────────────────────────────────────────────────
#  INTEGRATED SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_rrt_pid(config, waypoints_m, verbose=True):
    """
    Full simulation: RRT*-planned waypoints tracked by cascaded PID.

    Parameters
    ----------
    config      : dict with keys 'quad', 'controller', 'sim'
    waypoints_m : list of np.array([x_m, y_m])  — output of InformedRRTStar.plan()
    verbose     : print progress every 10 s

    Returns
    -------
    quad    : c4d.rigidbody with full stored history
    tracker : WaypointTracker — for post-simulation analysis
    """
    # ── Build quadcopter object ───────────────────────────────────────────────
    quad = c4d.rigidbody()
    for k, v in config["quad"].items():
        setattr(quad, k, v)

    # Control inputs (stored alongside state)
    quad.F         = quad.m * quad.g   # hover thrust initialisation
    quad.tau_phi   = 0.0
    quad.tau_theta = 0.0
    quad.tau_psi   = 0.0

    # ── Build controllers ─────────────────────────────────────────────────────
    ctrl_p  = config["controller"]
    z_ref   = config["sim"].get("z_ref", 1.5)
    cruise_speed = config["sim"].get("cruise_speed", 0.5)

    outer_ctrl = OuterPositionPID(ctrl_p, quad.m, quad.g, quad.kT)
    mid_ctrl   = MiddleAttitudePID(ctrl_p)
    inner_ctrl = InnerRatePID(ctrl_p, quad.Ixx, quad.Iyy, quad.Izz, quad.l, quad.kT)
    allocator  = ControlAllocator(quad.kT, quad.kQ, quad.l, ctrl_p["omega_max"])

    # ── Waypoint tracker ──────────────────
    tracker = WaypointTracker(
        waypoints_m,
        z_ref         = z_ref,
        switch_radius = config["sim"].get("switch_radius", 2.0),
        altitude_tol  = config["sim"].get("altitude_tol",  0.15),
        t_takeoff     = config["sim"].get("t_takeoff",     6.0),
        cruise_speed  = cruise_speed,
    )

    # ── Loop timing ───────────────────────────────────────────────────────────
    dt         = config["sim"]["dt"]   # master timestep (200 Hz = 5 ms)
    tf         = config["sim"]["tf"]
    Ts_outer   = 1.0 / 50.0           # outer loop period [s]
    Ts_middle  = 1.0 / 100.0          # middle loop period [s]
    outer_time = middle_time = 0.0

    # ── Initial control setpoints ─────────────────────────────────────────────
    psi_d   = 0.0
    phi_d   = theta_d = 0.0
    p_d     = q_d = r_d = 0.0
    T_cmd   = quad.m * quad.g
    rotor_speeds = np.array([np.sqrt(T_cmd / (4 * quad.kT))] * 4)

    if not hasattr(quad, "ref_history"):
            quad.ref_history = {"x": [], "y": [], "z": [], "t": [],
                                "xa": [], "ya": [], "za": []}
    t_vec = np.arange(0.0, tf, dt)
    print(f"[Sim] Start  tf={tf}s  dt={dt}s  steps={len(t_vec)}")
    print(f"[Sim] Waypoints: {len(waypoints_m)}  z_ref={z_ref}m  cruise_speed={cruise_speed}m/s")

    for t in t_vec:
        if verbose and t % 10.0 < dt / 2:
            print(f"[Sim] t={t:6.1f}s  phase={tracker.phase}  "
                  f"wp={tracker.wp_idx}/{len(waypoints_m)-1}  "
                  f"pos=({quad.x:.2f},{quad.y:.2f},{quad.z:.2f})")

        # ── Log state and controls ────────────────────────────────────────────
        quad.store(t)
        quad.storeparams(["F", "tau_phi", "tau_theta", "tau_psi",
                          "x", "y", "z", "vx", "vy", "vz",
                          "phi", "theta", "psi"], t=t)

        # ── Compute setpoint and feedforward from waypoint tracker ────────────
        xd, yd, zd, vx_ff, vy_ff = tracker.update(quad, dt)

        # Store reference AND actual — guarantees identical length and timestamps
        quad.ref_history["x"].append(xd)
        quad.ref_history["y"].append(yd)
        quad.ref_history["z"].append(zd)
        quad.ref_history["t"].append(t)
        quad.ref_history["xa"].append(quad.x)
        quad.ref_history["ya"].append(quad.y)
        quad.ref_history["za"].append(quad.z)

        # ── Outer loop — Position  (50 Hz) ────────────────────────────────────
        outer_time += dt
        if outer_time >= Ts_outer:
            # Use feedforward velocities from tracker
            T_cmd, phi_d, theta_d, _ = outer_ctrl.compute(
                xd, yd, zd, vx_ff, vy_ff, 0.0, quad, Ts_outer
            )
            psi_d = 0.0
            quad.F = T_cmd
            outer_time = 0.0

        # ── Middle loop — Attitude  (100 Hz) ──────────────────────────────────
        middle_time += dt
        if middle_time >= Ts_middle:
            p_d, q_d, r_d = mid_ctrl.compute(phi_d, theta_d, psi_d, quad, Ts_middle)
            middle_time    = 0.0

        # ── Inner loop — Rate  (200 Hz, every step) ───────────────────────────
        quad.tau_phi, quad.tau_theta, quad.tau_psi = inner_ctrl.compute(
            p_d, q_d, r_d, quad, dt
        )

        # ── Control allocation ────────────────────────────────────────────────
        rotor_speeds = np.array(
            allocator.allocate(quad.F, quad.tau_phi, quad.tau_theta, quad.tau_psi)
        )

        # ── Integrate dynamics ────────────────────────────────────────────────
        quad.X = rk4_step(t, quad.X, quad, rotor_speeds, dt)
        
        # ── Wrap yaw to [-π, π] in BOTH state vector AND rigidbody property ──
        quad.X[8] = np.arctan2(np.sin(quad.X[8]), np.cos(quad.X[8]))
        quad.psi  = quad.X[8]

    print("[Sim] Complete.")
    return quad, tracker


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(quad, tracker, save_dir="."):
    """
    Generate and save the three result figures.

    Figure 1 — 3D trajectory (planned path + actual flight)
    Figure 2 — Position components vs time + tracking error
    Figure 3 — Dashboard: XY plane | altitude | attitude | error

    Parameters
    ----------
    quad      : c4d.rigidbody
    tracker   : WaypointTracker
    save_dir  : directory for saved PNGs
    """
    import os
    from mpl_toolkits.mplot3d import Axes3D  
    from matplotlib import pyplot as plt

    t_hist = quad.data("x")[0]
    x_hist = quad.data("x")[1]
    y_hist = quad.data("y")[1]
    z_hist = quad.data("z")[1]

    phi_hist   = np.degrees(np.unwrap(quad.data("phi")[1]))
    theta_hist = np.degrees(np.unwrap(quad.data("theta")[1]))
    psi_hist   = np.degrees(np.unwrap(quad.data("psi")[1]))

    # Planned 3D path (RRT* waypoints at z_ref)
    planned = tracker.waypoints_as_xyz()
    p_x, p_y, p_z = planned[:, 0], planned[:, 1], planned[:, 2]

    # Reference setpoint history (rebuilt from tracker waypoints — step-wise)
    # For error computation, build nearest-waypoint reference over time
    # (simple: replicate the setpoint the tracker would have given at each step)
    err = np.sqrt((x_hist - p_x[np.clip(
                   np.searchsorted(np.linspace(0,1,len(p_x)),
                                   np.linspace(0,1,len(t_hist))),
                   0, len(p_x)-1)])**2 +
                  (y_hist - p_y[np.clip(
                   np.searchsorted(np.linspace(0,1,len(p_y)),
                                   np.linspace(0,1,len(t_hist))),
                   0, len(p_y)-1)])**2 +
                  (z_hist - tracker.z_ref)**2)

    lw = 1.5

    # ════════════════════════════════════════════════════════════════════════
    #  FIGURE 1 — 3D trajectory
    # ════════════════════════════════════════════════════════════════════════
    fig1 = plt.figure("3D Trajectory", figsize=(10, 8))
    ax   = fig1.add_subplot(111, projection="3d")
    ax.plot(x_hist, y_hist, z_hist, "b-", lw=lw, label="Actual flight")
    ax.plot(p_x, p_y, p_z, "r--o", lw=lw, ms=5, label="RRT* plan")
    ax.scatter(p_x[0],  p_y[0],  p_z[0],  c="green",  s=80, zorder=5, label="Start")
    ax.scatter(p_x[-1], p_y[-1], p_z[-1], c="orange", s=80, zorder=5, label="Goal")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("3D Trajectory — Informed RRT* + Cascade PID")
    ax.legend(fontsize=9); ax.grid(True)
    plt.tight_layout()
    path1 = os.path.join(save_dir, "fig1_3d_trajectory.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {path1}")

    # ════════════════════════════════════════════════════════════════════════
    #  FIGURE 2 — Position over time + tracking error
    # ════════════════════════════════════════════════════════════════════════
    fig2, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig2.suptitle("Position Tracking over Time", fontsize=14, fontweight="bold")

    axes[0].plot(t_hist, x_hist, "b-",  lw=lw, label="X actual")
    axes[0].axhline(p_x[-1], color="r", ls="--", lw=1, label="X goal")
    axes[0].set_ylabel("X (m)"); axes[0].legend(fontsize=8); axes[0].grid(True)

    axes[1].plot(t_hist, y_hist, "g-",  lw=lw, label="Y actual")
    axes[1].axhline(p_y[-1], color="r", ls="--", lw=1, label="Y goal")
    axes[1].set_ylabel("Y (m)"); axes[1].legend(fontsize=8); axes[1].grid(True)

    axes[2].plot(t_hist, z_hist, "m-",  lw=lw, label="Z actual")
    axes[2].axhline(tracker.z_ref, color="r", ls="--", lw=1, label="z_ref")
    axes[2].set_ylabel("Z (m)"); axes[2].legend(fontsize=8); axes[2].grid(True)

    axes[3].plot(t_hist, err, "r-", lw=lw)
    axes[3].set_ylabel("3D Error (m)"); axes[3].set_xlabel("Time (s)"); axes[3].grid(True)
    axes[3].set_title("3D Error to Final Goal (includes transit)")

    plt.tight_layout()
    path2 = os.path.join(save_dir, "fig2_position_vs_time.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {path2}")

    # ════════════════════════════════════════════════════════════════════════
    #  FIGURE 3 — Dashboard (2 × 3)
    # ════════════════════════════════════════════════════════════════════════
    fig3 = plt.figure("Dashboard", figsize=(16, 10))
    fig3.suptitle("Cascade PID + Informed RRT* — Simulation Dashboard",
                  fontsize=14, fontweight="bold")

    # 3D subplot
    ax3d = fig3.add_subplot(2, 3, 1, projection="3d")
    ax3d.plot(x_hist, y_hist, z_hist, "b-", lw=lw, label="Actual")
    ax3d.plot(p_x, p_y, p_z, "r--", lw=1,  label="Planned")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_title("3D Trajectory"); ax3d.legend(fontsize=8); ax3d.grid(True)

    # XY plane
    ax = fig3.add_subplot(2, 3, 2)
    ax.plot(x_hist, y_hist, "b-", lw=lw, label="Actual")
    ax.plot(p_x, p_y, "r--o", lw=1, ms=4, label="Planned")
    ax.scatter(p_x[0], p_y[0], c="green", s=60, zorder=5)
    ax.scatter(p_x[-1], p_y[-1], c="orange", s=60, zorder=5)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("XY Plane"); ax.legend(fontsize=8); ax.grid(True); ax.axis("equal")

    # X and Y vs time
    ax = fig3.add_subplot(2, 3, 3)
    ax.plot(t_hist, x_hist, "b-", lw=lw, label="X actual")
    ax.plot(t_hist, y_hist, "g-", lw=lw, label="Y actual")
    ax.axhline(p_x[-1], color="b", ls="--", lw=1, label="X goal")
    ax.axhline(p_y[-1], color="g", ls="--", lw=1, label="Y goal")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Position (m)")
    ax.set_title("Horizontal Position"); ax.legend(fontsize=8); ax.grid(True)

    # Altitude
    ax = fig3.add_subplot(2, 3, 4)
    ax.plot(t_hist, z_hist, "b-", lw=lw, label="Z actual")
    ax.axhline(tracker.z_ref, color="r", ls="--", lw=1, label=f"z_ref={tracker.z_ref}m")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Altitude (m)")
    ax.set_title("Altitude Tracking"); ax.legend(fontsize=8); ax.grid(True)

    # Tracking error
    ax = fig3.add_subplot(2, 3, 5)
    ax.plot(t_hist, err, "r-", lw=lw)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (m)")
    ax.set_title("3D Tracking Error"); ax.grid(True)

    # Attitude angles
    ax = fig3.add_subplot(2, 3, 6)
    ax.plot(t_hist, phi_hist,   "b-", lw=lw, label="Roll φ")
    ax.plot(t_hist, theta_hist, "g-", lw=lw, label="Pitch θ")
    ax.plot(t_hist, psi_hist,   "r-", lw=lw, label="Yaw ψ")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (deg)")
    ax.set_title("Attitude Angles"); ax.legend(fontsize=8); ax.grid(True)

    plt.tight_layout()
    path3 = os.path.join(save_dir, "fig3_dashboard.png")
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {path3}")

    plt.close("all")
    return path1, path2, path3


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(quad, tracker):
    """
    Compute BOTH cruise tracking lag AND hover stability.
    Uses the reference history logged during the simulation.
 
    Reads actual positions from quad.ref_history["xa/ya/za"] and
    reference positions from quad.ref_history["x/y/z"] — both arrays
    are guaranteed to be the same length and share the same timestamps.
    """
    import numpy as np
 
    t_hist = np.array(quad.ref_history["t"])
    x_hist = np.array(quad.ref_history["xa"])
    y_hist = np.array(quad.ref_history["ya"])
    z_hist = np.array(quad.ref_history["za"])
 
    ref_x  = np.array(quad.ref_history["x"])
    ref_y  = np.array(quad.ref_history["y"])
    ref_z  = np.array(quad.ref_history["z"])
 
    goal_x = tracker.waypoints[-1][0]
    goal_y = tracker.waypoints[-1][1]
    goal_z = tracker.z_ref
 
    # --- 1. Find when the drone arrived at the goal ---
    dist_to_goal = np.sqrt(
        (x_hist - goal_x)**2 +
        (y_hist - goal_y)**2 +
        (z_hist - goal_z)**2
    )
    arrival_indices = np.where(dist_to_goal < 0.5)[0]
    t_arrival = t_hist[arrival_indices[0]] if len(arrival_indices) > 0 else t_hist[-1]
 
    # --- 2. CRUISE TRACKING LAG (t=8.0 s to t_arrival) ---
    t_takeoff   = 8.0
    cruise_mask = (t_hist >= t_takeoff) & (t_hist <= t_arrival)
 
    if cruise_mask.sum() > 0:
        ex = x_hist[cruise_mask] - ref_x[cruise_mask]
        ey = y_hist[cruise_mask] - ref_y[cruise_mask]
        ez = z_hist[cruise_mask] - ref_z[cruise_mask]
        rmse_cruise_x   = np.sqrt(np.mean(ex**2))
        rmse_cruise_y   = np.sqrt(np.mean(ey**2))
        rmse_cruise_z   = np.sqrt(np.mean(ez**2))
        rmse_cruise_xyz = np.sqrt(np.mean(ex**2 + ey**2 + ez**2))
    else:
        rmse_cruise_x = rmse_cruise_y = rmse_cruise_z = rmse_cruise_xyz = 0.0
 
    # --- 3. HOVER STABILITY (t_arrival to end) ---
    arrived_mask    = t_hist >= t_arrival
    ex_arr          = x_hist[arrived_mask] - goal_x
    ey_arr          = y_hist[arrived_mask] - goal_y
    ez_arr          = z_hist[arrived_mask] - goal_z
    rmse_arrived_xyz = np.sqrt(np.mean(ex_arr**2 + ey_arr**2 + ez_arr**2))
 
    final_3d = np.sqrt(
        (x_hist[-1] - goal_x)**2 +
        (y_hist[-1] - goal_y)**2 +
        (z_hist[-1] - goal_z)**2
    )
 
    return {
        "rmse_cruise_x"   : rmse_cruise_x,
        "rmse_cruise_y"   : rmse_cruise_y,
        "rmse_cruise_z"   : rmse_cruise_z,
        "rmse_cruise_xyz" : rmse_cruise_xyz,
        "rmse_arrived_xyz": rmse_arrived_xyz,
        "final_error"     : final_3d,
        "t_arrival"       : float(t_arrival),
        # Expose arrays for external plotting / debugging
        "_t"      : t_hist,
        "_x"      : x_hist,
        "_y"      : y_hist,
        "_z"      : z_hist,
    }