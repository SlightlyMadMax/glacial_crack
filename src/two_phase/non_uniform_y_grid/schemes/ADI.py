from parameters import *
from linear_algebra.tdma import tdma
from src.two_phase.non_uniform_y_grid.grid_generation import get_node_coord
import numpy as np


def find_rhs(T, F_new, F_old, j: int, i: int):
    inv_F_new = 1.0 / F_new[i]
    inv_FH_new = 1.0 / (H - F_new[i])
    inv_dx = 1.0 / dx
    inv_dt = 1.0 / dt
    j_int = int(0.5 * (N_Y - 1))
    chi = 1.0 if j < j_int else c_ice*rho_ice*k_w/(c_w*rho_w*k_ice)

    # Координата y
    if j == 0:
        y = 0.0
    elif j == N_Y - 1:
        y = 2.0
    else:
        y = get_node_coord(2 * j / (N_Y - 1))

    # Производная F по x
    df_dx = F_new[i + 1] - F_new[i - 1]

    # Вторая производная T по x
    d_xx = T[j, i + 1] - 2.0 * T[j, i] + T[j, i - 1]

    # Первая и смешанная производная T по y
    if j == 0:
        h = get_node_coord(2.0 / (N_Y - 1))
        h_1 = get_node_coord(4.0 / (N_Y - 1)) - h

        d_y = ((T[1, i] - T[0, i]) * (h + h_1) * (h + h_1) - (T[2, i] - T[0, i]) * h * h) / (h * h_1 * (h + h_1))

        d_xy = ((T[1, i+1]-T[0, i+1]-T[1, i-1]+T[0, i-1])*(h+h_1)*(h+h_1) +
                (T[2, i-1]-T[0, i-1]-T[2, i+1]+T[0, i+1])*h*h)/(h*h_1*(h+h_1)*dx)

    elif j == N_Y - 1:
        h = 2.0 - get_node_coord(2 * (N_Y - 2) / (N_Y - 1))
        h_0 = get_node_coord(2 * (N_Y - 2) / (N_Y - 1)) - get_node_coord(2 * (N_Y - 3) / (N_Y - 1))

        d_y = ((T[N_Y-1, i] - T[N_Y-2, i])*(h_0 + h)*(h_0 + h) + (T[N_Y-3, i] - T[N_Y-1, i])*h*h)/(h*h_0*(h+h_0))

        d_xy = ((T[N_Y-1, i+1]-T[N_Y-2, i+1]-T[N_Y-1, i-1]+T[N_Y-2, i-1])*(h_0+h)*(h_0+h) +
                (T[N_Y-3, i+1] - T[N_Y-1, i+1] - T[N_Y-3, i-1] + T[N_Y-1, i-1])*h*h)/(h*h_0*(h+h_0)*dx)
    else:
        h = get_node_coord(2 * (j + 1) / (N_Y - 1)) - y
        h_0 = y - get_node_coord(2 * (j - 1) / (N_Y - 1))
        h_2 = get_node_coord(2 * (j + 2) / (N_Y - 1)) - get_node_coord(2 * (j + 1) / (N_Y - 1))

        d_y = (T[j + 1, i] * h_0 * h_0 - T[j - 1, i] * h * h + T[j, i] * (h * h - h_0 * h_0))/(h * h_0 * (h + h_0))

        # TODO: second order for d_xy
        d_xy = (T[j + 1, i + 1] - T[j + 1, i - 1] - T[j - 1, i + 1] + T[j - 1, i - 1]) / (dx * (h_0 + h_2))

    if j <= j_int:
        kappa = y * inv_F_new * (inv_dt * (F_new[i] - F_old[i]) + 0.5 * inv_F_new * inv_dx * inv_dx * df_dx * df_dx -
                                inv_dx * inv_dx * (F_new[i + 1] - 2 * F_new[i] + F_new[i - 1]))
        zeta = - y * inv_F_new * df_dx * inv_dx
    else:
        kappa = inv_FH_new * (inv_dt * (2.0 - y) * (F_new[i] - F_old[i]) +
                              0.5 * inv_FH_new * (y - 1.0) * inv_dx * inv_dx * df_dx * df_dx -
                              (y - 1.0) * inv_dx * inv_dx * (F_new[i + 1] - 2 * F_new[i] + F_new[i - 1]))
        zeta = - (y - 1.0) * inv_FH_new * df_dx * inv_dx

    return T[j, i] + \
        dt * (d_xx * inv_dx * inv_dx +
              0.5 * zeta * d_xy +
              0.5 * kappa * d_y) / chi


def solve(T, F_new, F_old):
    temp_T = np.copy(T)
    new_T = np.copy(T)
    F_new = np.copy(F_new)
    F_old = np.copy(F_old)
    j_int = int(0.5 * (N_Y - 1))
    inv_dx = 1.0 / dx

    a_y = np.empty((N_Y - 1), )
    b_y = np.empty((N_Y - 1), )
    c_y = np.empty((N_Y - 1), )

    rhs = np.empty(N_Y,)
    for i in range(1, N_X - 1):
        inv_F_new = 1.0 / F_new[i]
        inv_FH_new = 1.0 / (H - F_new[i])
        df_dx = F_new[i + 1] - F_new[i - 1]
        for j in range(0, N_Y - 1):
            chi = 1.0 if j < j_int else c_ice * rho_ice * k_w / (c_w * rho_w * k_ice)
            if j == 0:
                y = 0.0
                h = get_node_coord(2.0 / (N_Y - 1))
                h_0 = 1.0  # не играет роли
            else:
                y = get_node_coord(2 * j / (N_Y - 1))
                h = get_node_coord(2 * (j + 1) / (N_Y - 1)) - y
                h_0 = y - get_node_coord(2 * (j - 1) / (N_Y - 1))

            if j <= j_int:
                sigma_j = W * W * inv_F_new * inv_F_new + \
                          (0.25 * y * inv_dx * inv_F_new * df_dx * y * inv_dx * inv_F_new * df_dx)
            else:
                sigma_j = W * W * inv_FH_new * inv_FH_new + \
                          (0.25 * (1.0 - y) * (1.0 - y) * inv_dx * inv_FH_new * df_dx * inv_dx * inv_FH_new * df_dx)

            a_y[j] = -2.0 * dt * sigma_j / (chi * (h * (h + h_0)))
            c_y[j] = -2.0 * dt * sigma_j / (chi * (h_0 * (h + h_0)))
            b_y[j] = 1.0 + 2.0 * dt * sigma_j / (chi * (h * h_0))

            rhs[j] = find_rhs(T, F_new, F_old, j, i)

        rhs[N_Y-1] = find_rhs(T, F_new, F_old, N_Y-1, i)

        # ПРОГОНКА
        temp_T[:, i] = tdma(
            alpha_0=0.0,  # Из левого граничного условия по y (1-го рода)
            beta_0=T_ice / T_0,  # Из левого граничного условия по y
            condition_type=1,
            phi=T_w / T_0,
            a=a_y,
            b=b_y,
            c=c_y,
            f=rhs
        )
        temp_T[j_int, i] = T_0 / T_0

    rhs = np.empty(N_X,)
    for j in range(1, N_Y - 1):
        chi = 1.0 if j < j_int else c_ice * rho_ice * k_w / (c_w * rho_w * k_ice)
        a = c = np.ones((N_X - 1,)) * inv_dx * inv_dx * -dt / chi
        b = np.ones((N_X - 1,)) * (1.0 + 2.0 * dt * inv_dx * inv_dx / chi)
        for i in range(0, N_X):
            if i == 0:
                d_xx = T[j, 2] - 2.0 * T[j, 1] + T[j, 0]
            elif i == N_X - 1:
                d_xx = T[j, N_X - 1] - 2.0 * T[j, N_X - 2] + T[j, N_X - 3]
            else:
                d_xx = T[j, i + 1] - 2.0 * T[j, i] + T[j, i - 1]

            rhs[i] = temp_T[j, i] - dt * inv_dx * inv_dx * d_xx / chi

        new_T[j, :] = tdma(
            alpha_0=1.0,  # Из левого граничного условия по x (2-го рода)
            beta_0=0.0,  # Из левого граничного условия по x
            condition_type=2,
            phi=0.0,
            a=a,
            b=b,
            c=c,
            f=rhs
        )

    return new_T
