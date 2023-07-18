from parameters import *
from linear_algebra.tdma import tdma
from two_phase.nonuniform_y_grid.temperature import air_temperature
import numpy as np
import numba
from two_phase.nonuniform_y_grid.plotting import plot_non_transformed


@numba.jit
def find_rhs(T, F_new, F_old, Y, j: int, i: int, chi: float):
    inv_F = 1.0 / F_new[i]
    inv_FH = 1.0 / (H - F_new[i])
    inv_dx = 1.0 / dx
    inv_dt = 1.0 / dt
    j_int = int(0.5 * (N_Y - 1))

    # Координата y
    y = Y[j]

    # Производная F по x
    df_dx = F_new[i + 1] - F_new[i - 1]

    # Вторая производная T по x
    d_xx = T[j, i + 1] - 2.0 * T[j, i] + T[j, i - 1]

    # Первая и смешанная производная T по y
    if j == 0:
        h = Y[1]
        h_1 = Y[2] - h

        d_y = ((T[1, i] - T[0, i]) * (h + h_1) * (h + h_1) - (T[2, i] - T[0, i]) * h * h) / (h * h_1 * (h + h_1))

        d_xy = ((T[1, i+1]-T[0, i+1]-T[1, i-1]+T[0, i-1])*(h+h_1)*(h+h_1) +
                (T[2, i-1]-T[0, i-1]-T[2, i+1]+T[0, i+1])*h*h)/(h*h_1*(h+h_1)*dx)
    elif j == j_int and chi != 1.0:
        h = Y[j_int+1] - 1.0
        h_1 = Y[j_int+2] - Y[j_int+1]

        d_y = ((T[j_int+1, i] - T[j_int, i]) * (h + h_1) * (h + h_1) - (T[j_int+2, i] - T[j_int, i]) * h * h) / (h * h_1 * (h + h_1))

        d_xy = ((T[j_int+1, i+1]-T[j_int, i+1]-T[j_int+1, i-1]+T[j_int, i-1])*(h+h_1)*(h+h_1) +
                (T[j_int+2, i-1]-T[0, i-1]-T[j_int+2, i+1]+T[j_int, i+1])*h*h)/(h*h_1*(h+h_1)*dx)
    elif j == N_Y - 1:
        h = 2.0 - Y[N_Y-2]
        h_0 = Y[N_Y-2] - Y[N_Y-3]

        d_y = ((T[N_Y-1, i] - T[N_Y-2, i])*(h_0 + h)*(h_0 + h) + (T[N_Y-3, i] - T[N_Y-1, i])*h*h)/(h*h_0*(h+h_0))

        d_xy = ((T[N_Y-1, i+1]-T[N_Y-2, i+1]-T[N_Y-1, i-1]+T[N_Y-2, i-1])*(h_0+h)*(h_0+h) +
                (T[N_Y-3, i+1] - T[N_Y-1, i+1] - T[N_Y-3, i-1] + T[N_Y-1, i-1])*h*h)/(h*h_0*(h+h_0)*dx)
    elif j == j_int and chi == 1.0:
        h = 1.0 - Y[j_int-1]
        h_0 = Y[j_int-1] - Y[j_int-2]

        d_y = ((T[j_int, i] - T[j_int-1, i])*(h_0 + h)*(h_0 + h) + (T[j_int-2, i] - T[j_int, i])*h*h)/(h*h_0*(h+h_0))

        d_xy = ((T[j_int, i+1]-T[j_int-1, i+1]-T[j_int, i-1]+T[j_int-1, i-1])*(h_0+h)*(h_0+h) +
                (T[j_int-2, i+1] - T[j_int, i+1] - T[j_int-2, i-1] + T[j_int, i-1])*h*h)/(h*h_0*(h+h_0)*dx)
    else:
        h = Y[j+1] - y
        h_0 = y - Y[j-1]
        h_2 = Y[j+1] - Y[j-1]

        d_y = (T[j + 1, i] * h_0 * h_0 - T[j - 1, i] * h * h + T[j, i] * (h * h - h_0 * h_0))/(h * h_0 * (h + h_0))

        # TODO: second order for d_xy
        d_xy = (T[j + 1, i + 1] - T[j + 1, i - 1] - T[j - 1, i + 1] + T[j - 1, i - 1]) / (dx * h_2)

    if j <= j_int:
        kappa = y * inv_F * (inv_dt * (F_new[i] - F_old[i]) +
                             0.5 * inv_F * inv_dx * inv_dx * df_dx * df_dx -
                             inv_dx * inv_dx * (F_new[i + 1] - 2 * F_new[i] + F_new[i - 1]))
        zeta = -y * inv_F * df_dx * inv_dx
    else:
        kappa = inv_FH * (y - 2.0) * (inv_dt * (F_new[i] - F_old[i]) +
                                      0.5 * inv_FH * inv_dx * inv_dx * df_dx * df_dx / chi -
                                      inv_dx * inv_dx * (F_new[i + 1] - 2 * F_new[i] + F_new[i - 1]) / chi)
        zeta = (y - 2.0) * inv_FH * df_dx * inv_dx / chi

    return T[j, i] + \
        dt * (d_xx * inv_dx * inv_dx / chi +
              0.25 * zeta * d_xy +
              0.5 * kappa * d_y)


@numba.jit
def solve(T, F_new, F_old, Y, time: float):
    temp_T = np.copy(T)
    new_T = np.copy(T)
    j_int = int(0.5 * (N_Y - 1))
    inv_dx = 1.0 / dx

    a_y = np.empty((N_Y - 1), )
    b_y = np.empty((N_Y - 1), )
    c_y = np.empty((N_Y - 1), )

    T_air_t = air_temperature(time)

    rhs = np.empty(N_Y,)
    for i in range(1, N_X - 1):
        inv_F = 1.0 / F_new[i]
        inv_FH = 1.0 / (H - F_new[i])
        df_dx = F_new[i + 1] - F_new[i - 1]
        for j in range(0, N_Y - 1):
            chi = 1.0 if j <= j_int else (c_w * rho_w * k_ice) / (c_ice * rho_ice * k_w)
            y = Y[j]
            h = Y[j+1] - y
            h_0 = y - Y[j-1]
            if j <= j_int:
                sigma_j = W * W * inv_F * inv_F + \
                          (0.25 * y * y * inv_F * inv_F * inv_dx * df_dx * inv_dx * df_dx)
            else:
                sigma_j = W * W * inv_FH * inv_FH + \
                          0.25 * (2.0 - y) * (2.0 - y) * inv_FH * inv_FH * inv_dx * df_dx * inv_dx * df_dx
            a_y[j] = -2.0 * dt * sigma_j / (chi * h * (h + h_0))
            c_y[j] = -2.0 * dt * sigma_j / (chi * h_0 * (h + h_0))
            b_y[j] = 1.0 + 2.0 * dt * sigma_j / (chi * h * h_0)

            rhs[j] = find_rhs(T, F_new, F_old, Y, j, i, chi)

        chi = (c_w * rho_w * k_ice) / (c_ice * rho_ice * k_w)
        rhs[N_Y-1] = find_rhs(T, F_new, F_old, Y, N_Y-1, i, chi)

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

        # h = Y[j_int+1] - 1.0
        # h_0 = 1.0
        # sigma_j = W * W * inv_FH * inv_FH + 0.25 * inv_FH * inv_dx * df_dx * inv_FH * inv_dx * df_dx
        # a_y[j_int] = -2.0 * dt * sigma_j / (chi * (h * (h + h_0)))
        # c_y[j_int] = -2.0 * dt * sigma_j / (chi * (h_0 * (h + h_0)))
        # b_y[j_int] = 1.0 + 2.0 * dt * sigma_j / (chi * (h * h_0))

        rhs[j_int] = find_rhs(T, F_new, F_old, Y, j_int, i, chi)

        # ПРОГОНКА ДЛЯ ВОДЫ (первый шаг метода переменных направлений)
        temp_T[j_int:N_Y, i] = tdma(
            alpha_0=0.0,  # Из левого граничного условия по y (1-го рода)
            beta_0=T_0 / T_0,  # Из левого граничного условия по y (1-го рода)
            condition_type=3,
            phi=-conv_coef,  # правое граничное условие
            psi=conv_coef * T_air_t / T_0,
            h=(Y[N_Y-1]-Y[N_Y-2]),
            a=a_y[j_int:N_Y-1],
            b=b_y[j_int:N_Y-1],
            c=c_y[j_int:N_Y-1],
            f=rhs[j_int:N_Y]
        )

    # plot_non_transformed(
    #     T=temp_T,
    #     F=F_new,
    #     time=1,
    #     graph_id=1
    # )

    rhs = np.empty(N_X,)
    for j in range(1, N_Y - 1):
        inv_chi = 1.0 if j <= j_int else (c_ice * rho_ice * k_w) / (c_w * rho_w * k_ice)
        a = c = np.ones((N_X - 1,)) * inv_dx * inv_dx * -dt * inv_chi
        b = np.ones((N_X - 1,)) * (1.0 + 2.0 * dt * inv_dx * inv_dx * inv_chi)
        for i in range(0, N_X):
            if i == 0:
                d_xx = T[j, 2] - 2.0 * T[j, 1] + T[j, 0]
            elif i == N_X - 1:
                d_xx = T[j, N_X - 1] - 2.0 * T[j, N_X - 2] + T[j, N_X - 3]
            else:
                d_xx = T[j, i + 1] - 2.0 * T[j, i] + T[j, i - 1]

            rhs[i] = temp_T[j, i] - dt * inv_dx * inv_dx * d_xx * inv_chi

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

    # plot_non_transformed(
    #     T=new_T,
    #     F=F_new,
    #     time=2,
    #     graph_id=2
    # )

    return new_T
