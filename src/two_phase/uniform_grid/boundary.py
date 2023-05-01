import numpy as np
import math
from parameters import *


def init_f_vector(n_x):
    """
    Инициализация положения границы фазового перехода
    :param n_x: размерность (число шагов по X)
    :return: массив с координатами границы ф.п.
    """
    F = np.empty(n_x)

    # Гладкая ступенька
    F[:] = [0.4 + 0.7 / (1.0 + math.exp(-20.0 * (i * dx - 0.5))) for i in range(0, n_x)]

    # Плоскость
    # F[:] = a

    # Парабола
    # for i in range(n_x):
    #      F[i] = a + 2*(i*dx - W/2)*(i*dx - W/2)  # Парабола f(x, t=0) = 3*(x - W/2)^2 + a

    # Внешний угол
    # F[:] = [0.2 + 0.6*i*dx if i*dx < 0.5 else 0.5 for i in range(n_x)]

    # Угол
    # F[:] = [0.8 - i*dx if i*dx < 0.5 else i*dx - 0.2 for i in range(n_x)]

    # Трещина-гауссиана
    # F[:] = [1.5 - 0.7*math.exp(-(i*dx - 0.5)**2/0.01) for i in range(n_x)]

    # Параболическая трещина
    # for i in range(n_x):
    #     x = i*dx
    #     if 0.4 < x < 0.6:
    #         F[i] = 0.2 + 80*(i*dx - W/2)*(i*dx - W/2)
    #     else:
    #         F[i] = 1.0

    # Угольная трещина
    # for i in range(n_x):
    #     x = i * dx
    #     if x < 0.4 or x > 0.6:
    #         F[i] = 1.0
    #     else:
    #         F[i] = 3.8 - 7.0 * x if x< 0.5 else 7.0 * x - 3.2
    return F


def recalculate_boundary(F, T):
    """
    Пересчёт положения границы фазового перехода в соответствии с условием Стефана в новых координатах.
    :param F: одномерный массив координат границы фазового перехода.
    :param T: матрица значений температуры на сетке в НОВЫХ координатах.
    :return: массив с координатами границы ф.п. на новом временном шаге
    """

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_W = 1.0 / W
    inv_gamma = 1.0 / gamma  # безразмерный параметр в условии Стефана
    j_int = int(0.5 * (N_Y - 1))  # координата границы фазового перехода в новых координатах

    F_new = np.copy(F)

    for i in range(1, N_X - 1):
        df_dx = 0.5 * inv_W * inv_dx * (F[i + 1] - F[i - 1])

        dT_dy_i = 0.5 * inv_dy * (3.0 * T[j_int, i] - 4.0 * T[j_int-1, i] + T[j_int-2, i])

        dT_dy_w = 0.5 * inv_dy * (4.0 * T[j_int+1, i] - 3.0 * T[j_int, i] - T[j_int+2, i])

        F_new[i] = F[i] + \
                   dt * inv_gamma * (1.0 + df_dx * df_dx) * (dT_dy_i / F[i] - k_w * dT_dy_w / (k_ice * (H - F[i])))

    # При x = 0 и x = W рассчитываем отдельно, т.к. нужны другие конечно-разностные аппроксимации для df_dx
    df_dx_0 = 0.5 * inv_W * inv_dx * (4.0 * F[1] - 3.0 * F[0] - F[2])

    dT_dy_i_0 = 0.5 * inv_dy * (3.0 * T[j_int, 0] - 4.0 * T[j_int-1, 0] + T[j_int-2, 0])

    dT_dy_w_0 = 0.5 * inv_dy * (4.0 * T[j_int+1, 0] - 3.0 * T[j_int, 0] - T[j_int+2, 0])

    F_new[0] = F[0] + dt * inv_gamma * (1.0 + df_dx_0 * df_dx_0) * \
               (dT_dy_i_0 / F[0] - k_w * dT_dy_w_0 / (k_ice * (H - F[0])))

    df_dx_n = 0.5 * inv_W * inv_dx * (3.0 * F[N_X - 1] - 4.0 * F[N_X - 2] + F[N_X - 3])

    dT_dy_i_n = 0.5 * inv_dy * (3.0 * T[j_int, N_X-1] - 4.0 * T[j_int-1, N_X-1] + T[j_int-2, N_X-1])

    dT_dy_w_n = 0.5 * inv_dy * (4.0 * T[j_int+1, N_X-1] - 3.0 * T[j_int, N_X-1] - T[j_int+2, N_X-1])

    F_new[N_X - 1] = F[N_X-1] + dt * inv_gamma * (1.0 + df_dx_n * df_dx_n) * \
                     (dT_dy_i_n / F[N_X-1] - k_w * dT_dy_w_n / (k_ice * (H - F[N_X-1])))

    return F_new
