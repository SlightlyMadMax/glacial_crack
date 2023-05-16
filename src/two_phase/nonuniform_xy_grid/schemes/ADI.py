from parameters import *
from linear_algebra.tdma import tdma
from src.two_phase.nonuniform_xy_grid.grid_generation import get_x_node_coord, get_y_node_coord
import numpy as np
from src.two_phase.nonuniform_xy_grid.temperature import air_temperature
import numba
from src.two_phase.nonuniform_xy_grid.plotting import plot_non_transformed


@numba.jit
def find_rhs(T, F_new, F_old, j: int, i: int, chi: float):
    inv_F = 1.0 / F_new[i]
    inv_FH = 1.0 / (H - F_new[i])
    inv_dt = 1.0 / dt
    j_int = int(0.5 * (N_Y - 1))

    # Координата y
    y = get_y_node_coord(j, j_int)

    # Координата x
    x = get_x_node_coord(i)

    h_x = get_x_node_coord(i + 1) - x
    h_x_0 = x - get_x_node_coord(i - 1)
    h_x_01 = get_x_node_coord(i + 1) - get_x_node_coord(i - 1)

    df_dx = (F_new[i + 1] * h_x_0 * h_x_0 - F_new[i - 1] * h_x * h_x + F_new[i] * (h_x * h_x - h_x_0 * h_x_0)) / (h_x * h_x_0 * (h_x + h_x_0))

    d2f_dx2 = (F_new[i + 1] * h_x_0 + F_new[i - 1] * h_x - F_new[i] * h_x_01) / (0.5 * h_x_01 * h_x * h_x_0)

    # Вторая производная T по x
    d_xx = (T[j, i + 1] * h_x_0 + T[j, i - 1] * h_x - T[j, i] * h_x_01) / (0.5 * h_x_01 * h_x * h_x_0)

    # Первая и смешанная производная T по y
    if j == 0:
        h_y = get_y_node_coord(1, j_int)
        h_y_1 = get_y_node_coord(2, j_int) - h_y

        d_y = ((T[1, i] - T[0, i]) * (h_y + h_y_1) * (h_y + h_y_1) - (T[2, i] - T[0, i]) * h_y * h_y) / (h_y * h_y_1 * (h_y + h_y_1))
        d_xy = (T[1, i + 1] - T[1, i - 1] - T[0, i + 1] + T[0, i - 1]) / (h_x_01 * h_y)
    elif j == j_int and chi != 1.0:
        h_y = get_y_node_coord(j_int + 1, j_int) - 1.0
        h_y_1 = get_y_node_coord(j_int + 2, j_int) - get_y_node_coord(j_int + 1, j_int)

        d_y = ((T[1, i] - T[0, i]) * (h_y + h_y_1) * (h_y + h_y_1) - (T[2, i] - T[0, i]) * h_y * h_y) / (h_y * h_y_1 * (h_y + h_y_1))
        d_xy = (T[j_int, i + 1] - T[j_int, i - 1] - T[j_int - 1, i + 1] + T[j_int - 1, i - 1]) / (h_x_01 * h_y)
    elif j == N_Y - 1:
        h_y = 2.0 - get_y_node_coord(N_Y - 2, j_int)
        h_y_0 = get_y_node_coord(N_Y - 2, j_int) - get_y_node_coord(N_Y - 3, j_int)

        d_y = ((T[N_Y-1, i] - T[N_Y-2, i])*(h_y_0 + h_y)*(h_y_0 + h_y) + (T[N_Y-3, i] - T[N_Y-1, i])*h_y*h_y)/(h_y*h_y_0*(h_y+h_y_0))
        d_xy = (T[N_Y - 1, i + 1] - T[N_Y - 1, i - 1] - T[N_Y - 2, i + 1] + T[N_Y - 2, i - 1]) / (h_x_01 * h_y)
    elif j == j_int and chi == 1.0:
        h_y = 1.0 - get_y_node_coord(j_int - 1, j_int)
        h_y_0 = get_y_node_coord(j_int - 1, j_int) - get_y_node_coord(j_int - 2, j_int)

        d_y = ((T[N_Y-1, i] - T[N_Y-2, i])*(h_y_0 + h_y)*(h_y_0 + h_y) + (T[N_Y-3, i] - T[N_Y-1, i])*h_y*h_y)/(h_y*h_y_0*(h_y+h_y_0))
        d_xy = (T[j_int, i + 1] - T[j_int, i - 1] - T[j_int-1, i + 1] + T[j_int-1, i - 1]) / (h_x_01 * h_y)
    else:
        h_y = get_y_node_coord(j + 1, j_int) - y
        h_y_0 = y - get_y_node_coord(j - 1, j_int)
        h_y_01 = get_y_node_coord(j + 1, j_int) - get_y_node_coord(j - 1, j_int)

        d_y = (T[j + 1, i] * h_y_0 * h_y_0 - T[j - 1, i] * h_y * h_y + T[j, i] * (h_y * h_y - h_y_0 * h_y_0))/(h_y * h_y_0 * (h_y + h_y_0))
        d_xy = (T[j + 1, i + 1] - T[j + 1, i - 1] - T[j - 1, i + 1] + T[j - 1, i - 1]) / (h_x_01 * h_y_01)

    if j <= j_int:
        kappa = y * inv_F * (inv_dt * (F_new[i] - F_old[i]) +
                             2.0 * inv_F * df_dx * df_dx -
                             d2f_dx2)
        zeta = - 2.0 * y * inv_F * df_dx
    else:
        kappa = inv_FH * (y - 2.0) * (inv_dt * (F_new[i] - F_old[i]) +
                                      2.0 * inv_FH * df_dx * df_dx / chi -
                                      d2f_dx2 / chi)
        zeta = 2 * (y - 2.0) * inv_FH * df_dx / chi

    return T[j, i] + \
        dt * (d_xx / chi +
              0.5 * zeta * d_xy +
              0.5 * kappa * d_y)


# @numba.jit
def solve(T, F_new, F_old, time: float):
    temp_T = np.copy(T)
    new_T = np.copy(T)
    # F_new = np.copy(F_new)
    # F_old = np.copy(F_old)
    j_int = int(0.5 * (N_Y - 1))

    a_y = np.empty((N_Y - 1), )
    b_y = np.empty((N_Y - 1), )
    c_y = np.empty((N_Y - 1), )

    a_x = np.empty((N_X - 1), )
    b_x = np.empty((N_X - 1), )
    c_x = np.empty((N_X - 1), )

    T_air_t = air_temperature(time)

    rhs = np.empty(N_Y,)
    for i in range(1, N_X - 1):
        inv_F = 1.0 / F_new[i]
        inv_FH = 1.0 / (H - F_new[i])

        x = get_x_node_coord(i)

        h_x = get_x_node_coord(i + 1) - x
        h_x_0 = x - get_x_node_coord(i - 1)

        df_dx = (F_new[i + 1] * h_x_0 * h_x_0 - F_new[i - 1] * h_x * h_x + F_new[i] * (h_x * h_x - h_x_0 * h_x_0)) / (
                    h_x * h_x_0 * (h_x + h_x_0))

        for j in range(0, N_Y - 1):
            chi = 1.0 if j <= j_int else c_w * rho_w * k_ice / (c_ice * rho_ice * k_w)
            if j == 0:
                y = 0.0
                h = get_y_node_coord(1, j_int)
                h_0 = 1.0  # не играет роли
            else:
                y = get_y_node_coord(j, j_int)
                h = get_y_node_coord(j + 1, j_int) - y
                h_0 = y - get_y_node_coord(j - 1, j_int)

            if j <= j_int:
                sigma_j = W * W * inv_F * inv_F + \
                          (y * y * inv_F * inv_F * df_dx * df_dx)
            else:
                sigma_j = W * W * inv_FH * inv_FH + \
                          (2.0 - y) * (2.0 - y) * inv_FH * df_dx * inv_FH * df_dx

            a_y[j] = -2.0 * dt * sigma_j / (chi * (h * (h + h_0)))
            c_y[j] = -2.0 * dt * sigma_j / (chi * (h_0 * (h + h_0)))
            b_y[j] = 1.0 + 2.0 * dt * sigma_j / (chi * (h * h_0))

            rhs[j] = find_rhs(T, F_new, F_old, j, i, chi)

        chi = c_w * rho_w * k_ice / (c_ice * rho_ice * k_w)
        rhs[N_Y-1] = find_rhs(T, F_new, F_old, N_Y-1, i, chi)

        # ПРОГОНКА ДЛЯ ЛЬДА (первый шаг метода переменных направлений)
        temp_T[0:j_int + 1, i] = tdma(
            alpha_0=0.0,  # Из левого граничного условия по y (1-го рода)
            beta_0=T_ice / T_0,  # Из левого граничного условия по y (1-го рода)
            condition_type=1,
            phi=T_0 / T_0,  # правое граничное условие
            a=a_y[0:j_int],
            b=b_y[0:j_int],
            c=c_y[0:j_int],
            f=rhs[0:j_int + 1]
        )

        h = get_y_node_coord(j_int + 1, j_int) - 1.0
        h_0 = 1.0
        sigma_j = W * W * inv_FH * inv_FH + inv_FH * df_dx * inv_FH * df_dx
        a_y[j_int] = -2.0 * dt * sigma_j / (chi * (h * (h + h_0)))
        c_y[j_int] = -2.0 * dt * sigma_j / (chi * (h_0 * (h + h_0)))
        b_y[j_int] = 1.0 + 2.0 * dt * sigma_j / (chi * (h * h_0))

        rhs[j_int] = find_rhs(T, F_new, F_old, j_int, i, chi)

        # ПРОГОНКА ДЛЯ ВОДЫ (первый шаг метода переменных направлений)
        temp_T[j_int:N_Y, i] = tdma(
            alpha_0=0.0,  # Из левого граничного условия по y (1-го рода)
            beta_0=T_0 / T_0,  # Из левого граничного условия по y (1-го рода)
            condition_type=1,
            phi=T_air_t / T_0,  # правое граничное условие
            a=a_y[j_int:N_Y-1],
            b=b_y[j_int:N_Y-1],
            c=c_y[j_int:N_Y-1],
            f=rhs[j_int:N_Y]
        )

    plot_non_transformed(
        T=temp_T,
        F=F_new,
        time=1,
        graph_id=1
    )

    rhs = np.empty(N_X,)
    for j in range(1, N_Y - 1):
        chi = 1.0 if j <= j_int else c_w * rho_w * k_ice / (c_ice * rho_ice * k_w)
        for i in range(0, N_X - 1):
            x = get_x_node_coord(i)
            h_x = get_x_node_coord(i + 1) - x
            h_x_0 = x - get_x_node_coord(i - 1)
            h_x_01 = get_x_node_coord(i + 1) - get_x_node_coord(i - 1)

            a_x[i] = -2.0 * dt / (chi * (h_x * (h_x + h_x_0)))
            c_x[i] = -2.0 * dt / (chi * (h_x_0 * (h_x + h_x_0)))
            b_x[i] = 1.0 + 2.0 * dt / (chi * (h_x * h_x_0))

            if i != N_X-1:
                d_xx = (T[j, i + 1] * h_x_0 + T[j, i - 1] * h_x - T[j, i] * h_x_01) / (0.5 * h_x_01 * h_x * h_x_0)
            else:
                d_xx = 0

            rhs[i] = temp_T[j, i] - dt * d_xx / chi

        new_T[j, :] = tdma(
            alpha_0=1.0,  # Из левого граничного условия по x (2-го рода)
            beta_0=0.0,  # Из левого граничного условия по x
            condition_type=2,
            phi=0.0,
            a=a_x,
            b=b_x,
            c=c_x,
            f=rhs
        )
    plot_non_transformed(
        T=new_T,
        F=F_new,
        time=2,
        graph_id=2
    )
    return new_T
