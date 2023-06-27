import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
# from numba import jit
# @jit(nopython=False, forceobj=True)

# Functions needed
def rel_crtbp(t, x, mu=0.012150583925359):
    """
    Circular Restricted Three-Body Problem Dynamicsnopython=True

    :param t: Time, scalar
    :param x: State, vector 6x1
    :param mu: Gravitational constant, scalar
    :return: State Derivative, vector 6x1
    """

    # Initialize ODE
    dxdt = np.zeros((12,))
    # Initialize Target State
    xt = x[0]
    yt = x[1]
    zt = x[2]
    xtdot = x[3]
    ytdot = x[4]
    ztdot = x[5]
    # Initialize Relative State
    xr = x[6]
    yr = x[7]
    zr = x[8]
    xrdot = x[9]
    yrdot = x[10]
    zrdot = x[11]

    # Relative CRTBP Dynamics
    r1t = [xt + mu, yt, zt]
    r2t = [xt + mu - 1, yt, zt]
    r1t_norm = (
        np.sqrt((xt + mu) ** 2 + yt**2 + zt**2) + t * 0
    )  # JUST TO REMOVE ERROR IN T
    r2t_norm = np.sqrt((xt + mu - 1) ** 2 + yt**2 + zt**2)
    rho = [xr, yr, zr]

    dxdt[0:3] = [xtdot, ytdot, ztdot]
    dxdt[3:6] = [
        2 * ytdot
        + xt
        - (1 - mu) * (xt + mu) / r1t_norm**3
        - mu * (xt + mu - 1) / r2t_norm**3,
        -2 * xtdot + yt - (1 - mu) * yt / r1t_norm**3 - mu * yt / r2t_norm**3,
        -(1 - mu) * zt / r1t_norm**3 - mu * zt / r2t_norm**3,
    ]

    dxdt[6:9] = [xrdot, yrdot, zrdot]
    dxdt[9:12] = [
        2 * yrdot
        + xr
        + (1 - mu)
        * (
            (xt + mu) / r1t_norm**3
            - (xt + xr + mu) / np.linalg.norm(np.add(r1t, rho)) ** 3
        )
        + mu
        * (
            (xt + mu - 1) / r2t_norm**3
            - (xt + xr + mu - 1) / np.linalg.norm(np.add(r2t, rho)) ** 3
        ),
        -2 * xrdot
        + yr
        + (1 - mu)
        * (yt / r1t_norm**3 - (yt + yr) / np.linalg.norm(np.add(r1t, rho)) ** 3)
        + mu * (yt / r2t_norm**3 - (yt + yr) / np.linalg.norm(np.add(r2t, rho)) ** 3),
        (1 - mu)
        * (zt / r1t_norm**3 - (zt + zr) / np.linalg.norm(np.add(r1t, rho)) ** 3)
        + mu * (zt / r2t_norm**3 - (zt + zr) / np.linalg.norm(np.add(r2t, rho)) ** 3),
    ]
    return dxdt


def rel_crtbp2(t, x, mu=0.012150583925359):
    """
    Circular Restricted Three-Body Problem Dynamics

    :param t: Time, scalar
    :param x: State, vector 6x1
    :param mu: Gravitational constant, scalar
    :return: State Derivative, vector 12x1
    """

    # Initialize ODE
    dxdt = np.zeros((12,))
    # Initialize Target State
    xt = x[0]
    yt = x[1]
    zt = x[2]
    xtdot = x[3]
    ytdot = x[4]
    ztdot = x[5]
    # Initialize Chaser State
    xc = x[6] + xt
    yc = x[7] + yt
    zc = x[8] + zt
    xcdot = x[9] + xtdot
    ycdot = x[10] + ytdot
    zcdot = x[11] + ztdot

    # Crtbp relative dynamics
    r1t_norm = (
        np.sqrt((xt + mu) ** 2 + yt**2 + zt**2) + t * 0
    )  # JUST TO REMOVE ERROR IN T
    r2t_norm = np.sqrt((xt + mu - 1) ** 2 + yt**2 + zt**2)
    r1c_norm = np.sqrt((xc + mu) ** 2 + yc**2 + zc**2)
    r2c_norm = np.sqrt((xc + mu - 1) ** 2 + yc**2 + zc**2)

    dxdt[0:3] = [xtdot, ytdot, ztdot]
    dxdt[3:6] = [
        2 * ytdot
        + xt
        - (1 - mu) * (xt + mu) / r1t_norm**3
        - mu * (xt + mu - 1) / r2t_norm**3,
        -2 * xtdot + yt - (1 - mu) * yt / r1t_norm**3 - mu * yt / r2t_norm**3,
        -(1 - mu) * zt / r1t_norm**3 - mu * zt / r2t_norm**3,
    ]

    dxdt[6:9] = np.subtract([xcdot, ycdot, zcdot], dxdt[0:3])
    dxdt[9:12] = np.subtract(
        [
            2 * ycdot
            + xc
            - (1 - mu) * (xc + mu) / r1c_norm**3
            - mu * (xc + mu - 1) / r2c_norm**3,
            -2 * xcdot + yc - (1 - mu) * yc / r1c_norm**3 - mu * yc / r2c_norm**3,
            -(1 - mu) * zc / r1c_norm**3 - mu * zc / r2c_norm**3,
        ],
        dxdt[3:6],
    )

    return dxdt


# Data
mu = 0.012150583925359
m_star = 6.0458 * 1e24  # Kilograms
l_star = 3.844 * 1e8  # Meters
t_star = 375200  # Seconds

# Initialization
# x0t_state = np.array([1.0221, 0, -0.1821, 0, -0.1033, 0])  # 9:2 NRO, already corrected
x0t_state = np.array(
    [
        1.02206694e00,
        -5.20646335e-07,
        -1.82100000e-01,
        -6.66065965e-07,
        -1.03353155e-01,
        2.53475103e-06,
    ]
)
x0r_state = np.array(  # 600ish meters
    [
        1.62951874e-11,
        1.62263261e-06,
        -6.20702933e-11,
        2.07476683e-06,
        -5.50569035e-11,
        -7.90303573e-06,
    ]
)
x0r_state_small = np.array(  # 200 meters
    [
        1.67732495e-12,
        5.20646335e-07,
        -6.38444853e-12,
        6.66065965e-07,
        -5.66750813e-12,
        -2.53475103e-06,
    ]
)
x0 = np.concatenate((x0t_state, x0r_state_small))

# Integration
sol = solve_ivp(
    rel_crtbp,
    (0, np.pi),
    x0,
    "RK45",
    rtol=2.220446049250313e-14,
    atol=2.220446049250313e-14,
)
x_sol = np.transpose(sol.y)
t_vec = sol.t
xt_sol = x_sol[:, 0:6]
xr_sol = x_sol[:, 6:12]

# Approach Corridor
rad_kso = 200
ang_corr = np.deg2rad(10)
rad_entry = np.tan(ang_corr) * rad_kso
x_cone, y_cone = np.mgrid[-rad_entry:rad_entry:1000j, -rad_entry:rad_entry:1000j]
x_cone = x_cone / l_star
y_cone = y_cone / l_star
z_cone = np.sqrt((x_cone**2 + y_cone**2) / np.square(np.tan(ang_corr)))
z_cone = np.where(z_cone > (rad_kso / l_star), np.nan, z_cone)

# Keep-Out Sphere
x_sph, y_sph = np.mgrid[-rad_kso:rad_kso:1000j, -rad_kso:rad_kso:1000j]
x_sph = x_sph / l_star
y_sph = y_sph / l_star
z_sph1_sq = (rad_kso / l_star) ** 2 - x_sph**2 - y_sph**2
z_sph1_sq = np.where(z_sph1_sq < 0, np.nan, z_sph1_sq)
z_sph1 = np.sqrt(z_sph1_sq)
z_sph2 = -z_sph1

# Plot Target
plt.figure(1)
ax = plt.axes(projection="3d")
ax.plot3D(xt_sol[:, 0], xt_sol[:, 1], xt_sol[:, 2], "b", linewidth=2)
ax.plot3D(1 - mu, 0, 0, "ko", markersize=5)
ax.plot3D(xt_sol[0, 0], xt_sol[0, 1], xt_sol[0, 2], "go", markersize=5)
ax.plot3D(
    xt_sol[0, 0] + xr_sol[0, 0],
    xt_sol[0, 1] + xr_sol[0, 1],
    xt_sol[0, 2] + xr_sol[0, 2],
    "ro",
    markersize=5,
)
ax.legend(["Trajectory", "Moon", "Target", "Chaser"], loc="upper right")
ax.set_xlabel("x [DU]")
ax.set_ylabel("y [DU]")
ax.zaxis.set_rotate_label(False)
ax.set_zlabel("z [DU]", rotation=90, labelpad=20)
plt.locator_params(axis="x", nbins=4)
plt.locator_params(axis="y", nbins=4)
plt.locator_params(axis="z", nbins=4)
plt.tick_params(axis="z", which="major", pad=10)
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
ax.set_aspect("equal", "box")
ax.set_xticks([1.00])
ax.xaxis.pane.set_edgecolor("black")
ax.yaxis.pane.set_edgecolor("black")
ax.zaxis.pane.set_edgecolor("black")
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.view_init(elev=20, azim=40)
plt.show(block=False)
plt.savefig("Halo92.pdf", format="pdf")

# Plot Chaser Relative
plt.figure(2)
ax = plt.axes(projection="3d")
ax.plot3D(xr_sol[:, 0], xr_sol[:, 1], xr_sol[:, 2], "b", linewidth=2)
ax.plot3D(0, 0, 0, "ko", markersize=5)
ax.plot3D(xr_sol[0, 0], xr_sol[0, 1], xr_sol[0, 2], "go", markersize=5)
ax.plot3D(xr_sol[-1, 0], xr_sol[-1, 1], xr_sol[-1, 2], "ro", markersize=5)
ax.legend(["Trajectory", "Target", "Initial State", "Final State"], loc="upper right")
ax.plot_surface(x_cone, z_cone, y_cone, color="r")
ax.plot_surface(x_sph, y_sph, z_sph1, color="y", alpha=0.2)
ax.plot_surface(x_sph, y_sph, z_sph2, color="y", alpha=0.2)
ax.set_xlabel("$\delta x$ [DU]")
ax.set_ylabel("$\delta y$ [DU]")
ax.zaxis.set_rotate_label(False)
ax.set_zlabel("$\delta z$ [DU]", rotation=90, labelpad=20)
ax.set_xticks([0])
plt.locator_params(axis="x", nbins=4)
plt.locator_params(axis="y", nbins=4)
plt.locator_params(axis="z", nbins=4)
ax.set_aspect("equal", "box")
ax.xaxis.pane.set_edgecolor("black")
ax.yaxis.pane.set_edgecolor("black")
ax.zaxis.pane.set_edgecolor("black")
plt.tick_params(axis="z", which="major", pad=10)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
ax.zaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.view_init(elev=10, azim=30)
plt.savefig("Halorelci.pdf", format="pdf")
plt.show()

# Plot Chaser Relative Velocity
rhodot = np.zeros(len(t_vec))
for k in range(len(t_vec)):
    rhodot[k] = np.linalg.norm(xr_sol[k, 3:6]) * l_star / t_star
plt.figure(3)
plt.plot(t_vec * t_star, rhodot, "b", linewidth=2)
plt.legend(["Relative Velocity Chaser"], loc="upper right")
plt.xlabel("$t$ [s]")
plt.ylabel("$\delta v$ [m/s]")

# Plot Target Position
r = np.zeros(len(t_vec))
for k in range(len(t_vec)):
    r[k] = np.linalg.norm(xt_sol[k, 0:2]) * l_star
    print(r[k])
plt.figure(4)
# plt.plot(t_vec * t_star, r, "b", linewidth=2)
plt.plot(t_vec * t_star, xt_sol[:, 0] * l_star, "b", linewidth=2)
plt.plot(t_vec * t_star, xt_sol[:, 1] * l_star, "r", linewidth=2)
plt.plot(t_vec * t_star, xt_sol[:, 2] * l_star, "g", linewidth=2)
plt.legend(["x", "y", "z"], loc="upper right")
plt.xlabel("$t$ [s]")
plt.ylabel("$x$ [m]")
plt.show()

# Plot Target Velocity
v = np.zeros(len(t_vec))
for k in range(len(t_vec)):
    v[k] = np.linalg.norm(xt_sol[k, 3:6]) * l_star / t_star
    print(v[k])
plt.figure(5)
# plt.plot(t_vec * t_star, v, "b", linewidth=2)
plt.plot(t_vec * t_star, xt_sol[:, 3] * l_star / t_star, "b", linewidth=2)
plt.plot(t_vec * t_star, xt_sol[:, 4] * l_star / t_star, "r", linewidth=2)
plt.plot(t_vec * t_star, xt_sol[:, 5] * l_star / t_star, "g", linewidth=2)
plt.legend(["xdot", "ydot", "zdot"], loc="upper right")
plt.xlabel("$t$ [s]")
plt.ylabel("$v$ [m/s]")
plt.show()

# Jacobi integral
Cj = np.zeros(len(t_vec))
mu = 0.012150583925359
x = xt_sol[:, 0]
y = xt_sol[:, 1]
z = xt_sol[:, 2]
xdot = xt_sol[:, 3]
ydot = xt_sol[:, 4]
zdot = xt_sol[:, 5]
for k in range(len(t_vec)):
    r1 = np.sqrt((x[k] + mu) ** 2 + y[k] ** 2 + z[k] ** 2)
    r2 = np.sqrt((x[k] + mu - 1) ** 2 + y[k] ** 2 + z[k] ** 2)
    Cj[k] = (
        (x[k] ** 2 + y[k] ** 2)
        + 2 * (1 - mu) / r1
        + 2 * mu / r2
        - (xdot[k] ** 2 + ydot[k] ** 2 + zdot[k] ** 2)
    )
plt.figure(6)
plt.plot(t_vec * t_star, Cj, "b", linewidth=2)
plt.legend(["Jacobi Integral"], loc="upper right")
plt.xlabel("$t$ [s]")
plt.ylabel("$C$ [-]")
plt.show()
