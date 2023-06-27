import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Functions needed


def stm_crtbp(t, xm, mu=0.012150583925359):
    """
    Circular Restricted Three-Body Problem Dynamics

    :param t: Time, scalar
    :param xm: State, vector 6x1
    :param mu: Gravitational constant, scalar
    :return: State Derivative, vector 6x1
    """

    # Initialize
    dxmdt = np.zeros((42,))
    x = xm[0]
    y = xm[1]
    z = xm[2]
    xdot = xm[3]
    ydot = xm[4]
    zdot = xm[5]
    m = np.reshape(xm[6:42], (6, 6))  # From equations to STM

    # Crtbp dynamics
    r1_norm = (
        np.sqrt((x + mu) ** 2 + y**2 + z**2) + t * 0
    )  # JUST TO REMOVE ERROR IN T
    r2_norm = np.sqrt((x + mu - 1) ** 2 + y**2 + z**2)
    dxmdt[0:3] = [xdot, ydot, zdot]
    dxmdt[3:6] = [
        2 * ydot
        + x
        - (1 - mu) * (x + mu) / r1_norm**3
        - mu * (x + mu - 1) / r2_norm**3,
        -2 * xdot + y - (1 - mu) * y / r1_norm**3 - mu * y / r2_norm**3,
        -(1 - mu) * z / r1_norm**3 - mu * z / r2_norm**3,
    ]

    # Variational equations
    df4dx = (
        1
        - (1 - mu) / r1_norm**3
        + 3 * (1 - mu) * (x + mu) ** 2 / r1_norm**5
        - mu / r2_norm**3
        + 3 * mu * (x + mu - 1) ** 2 / r2_norm**5
    )
    df4dy = (
        3 * (1 - mu) * (x + mu) * y / r1_norm**5
        + 3 * mu * (x + mu - 1) * y / r2_norm**5
    )
    df4dz = (
        3 * (1 - mu) * (x + mu) * z / r1_norm**5
        + 3 * mu * (x + mu - 1) * z / r2_norm**5
    )
    df5dy = (
        1
        - (1 - mu) / r1_norm**3
        + 3 * (1 - mu) * y**2 / r1_norm**5
        - mu / r2_norm**3
        + 3 * mu * y**2 / r2_norm**5
    )
    df5dz = 3 * (1 - mu) * y * z / r1_norm**5 + 3 * mu * y * z / r2_norm**5
    df6dz = (
        -(1 - mu) / r1_norm**3
        + 3 * (1 - mu) * z**2 / r1_norm**5
        - mu / r2_norm**3
        + 3 * mu * z**2 / r2_norm**5
    )

    a = np.array(
        [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [df4dx, df4dy, df4dz, 0, 2, 0],
            [df4dy, df5dy, df5dz, -2, 0, 0],
            [df4dz, df5dz, df6dz, 0, 0, 0],
        ]
    )
    dmdt = np.matmul(a, m)

    dxmdt[6:12] = dmdt[0, :]
    dxmdt[12:18] = dmdt[1, :]
    dxmdt[18:24] = dmdt[2, :]
    dxmdt[24:30] = dmdt[3, :]
    dxmdt[30:36] = dmdt[4, :]
    dxmdt[36:42] = dmdt[5, :]

    return dxmdt


def event_corr1(t, y):
    return y[1] + 1e-100 + t * 0  # Not trigger even_corr at the initial condition


event_corr1.terminal = True  # Stop integration at the first event


def event_corr2(t, y):
    return y[1] - 1e-100 + t * 0  # Not trigger even_corr at the initial condition


event_corr2.terminal = True  # Stop integration at the first event

# Continuation Method - Part 1

# Initialization
mu = 0.012150583925359
x0_state = np.array(
    [1.120745458770128, 0, 0.011371823619780, 0, 0.175237781529976, 0]
)  # Generic Halo orbit around L2
x0_stm = np.reshape(np.eye(6), (36,))
x0 = np.concatenate((x0_state, x0_stm))
dx0 = np.zeros((6,))
i = 0

plt.figure(1)
ax = plt.axes(projection="3d")

for z0 in np.arange(x0_state[2], 0.075, 0.005):
    # Initialization
    i += 1
    x0_state[2] = z0

    # Differential correction
    xxm = np.ones((6,))
    dx0 = np.zeros((6,))

    while np.absolute(xxm[3]) > 1e-4 or np.absolute(xxm[5]) > 1e-4:
        # Integration
        x0 = np.concatenate((np.add(x0_state, dx0), x0_stm))
        sol = solve_ivp(
            stm_crtbp,
            (0, 2 * np.pi),
            x0,
            method="LSODA",
            events=event_corr1,
            rtol=2.5 * 1e-14,
            atol=2.5 * 1e-14,
        )
        xxm = np.transpose(sol.y[:, -1])

        # Reshape
        stmf = np.reshape(xxm[6:42], (6, 6))

        # Correction
        a_corr = np.matrix([[stmf[3, 0], stmf[3, 4]], [stmf[5, 0], stmf[5, 4]]])
        b_corr = np.array([-xxm[3], -xxm[5]])
        corr = np.linalg.solve(a_corr, b_corr)
        dx0 = np.add(dx0, np.array([corr[0], 0, 0, 0, corr[1], 0]))

    # Continuation
    x0_state = np.add(x0_state, dx0)

    # Final integration
    x0_cont = np.concatenate((x0_state, x0_stm))
    sol = solve_ivp(
        stm_crtbp,
        (0, 1.1 * np.pi),
        x0_cont,
        method="LSODA",
        rtol=1e-13,
        atol=1e-13,
        dense_output=True,
    )
    xx_corr = np.transpose(sol.y)

    # Plot with LAST continuation
    ax.plot3D(xx_corr[:, 0], xx_corr[:, 1], xx_corr[:, 2], "b", linewidth=2)

# Continuation Method - Part 2

# Initialization
mu = 0.012150583925359
x0_state = np.array([1.0221, 0, -0.1821, 0, -0.1033, 0])  # 9:2 NRO, already corrected
x0_stm = np.reshape(np.eye(6), (36,))
x0 = np.concatenate((x0_state, x0_stm))
dx0 = np.zeros((6,))
i = 0
k = 0

for x0 in np.arange(x0_state[0], 1.155, 0.001):
    # Initialization
    i += 1
    k += 1
    x0_state[0] = x0

    # Differential correction
    xxm = np.ones((6,))
    dx0 = np.zeros((6,))

    while np.absolute(xxm[3]) > 1e-8 or np.absolute(xxm[5]) > 1e-8:
        # Integration
        x0 = np.concatenate((np.add(x0_state, dx0), x0_stm))
        sol = solve_ivp(
            stm_crtbp,
            (0, 2 * np.pi),
            x0,
            method="LSODA",
            events=event_corr2,
            rtol=2.5 * 1e-14,
            atol=2.5 * 1e-14,
        )
        xxm = np.transpose(sol.y[:, -1])

        # Reshape
        stmf = np.reshape(xxm[6:42], (6, 6))

        # Correction
        a_corr = np.matrix([[stmf[3, 2], stmf[3, 4]], [stmf[5, 2], stmf[5, 4]]])
        b_corr = np.array([-xxm[3], -xxm[5]])
        corr = np.linalg.solve(a_corr, b_corr)
        dx0 = np.add(dx0, np.array([0, 0, corr[0], 0, corr[1], 0]))

    # Continuation
    x0_state = np.add(x0_state, dx0)

    if k == 5:
        # Final integration
        x0_cont = np.concatenate((x0_state, x0_stm))
        sol = solve_ivp(
            stm_crtbp,
            (0, 1.1 * np.pi),
            x0_cont,
            method="LSODA",
            rtol=1e-13,
            atol=1e-13,
            dense_output=True,
        )
        xx_corr = np.transpose(sol.y)

        # Plot with LAST continuation
        ax.plot3D(xx_corr[:, 0], xx_corr[:, 1], xx_corr[:, 2], "b", linewidth=2)

        k = 0

ax.plot3D(1 - mu, 0, 0, "ko", markersize=5)
ax.text(1 - mu, 0, 0.01, "Moon")
ax.plot3D(1.1556821603, 0, 0, "ko", markersize=5)
ax.text(1.1556821603, 0, 0.01, "L2")
ax.set_xlabel("x [DU]", labelpad=20)
ax.set_ylabel("y [DU]", labelpad=20)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel("z [DU]", rotation=90, labelpad=30)
ax.set_aspect("equal", "box")
ax.xaxis.pane.set_edgecolor("black")
ax.yaxis.pane.set_edgecolor("black")
ax.zaxis.pane.set_edgecolor("black")
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.view_init(elev=15, azim=-140)
plt.savefig("Halofamily.pdf", format="pdf")
plt.show()
