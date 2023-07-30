import math
import numpy as np
from parameters import *
from two_phase.nonuniform_y_grid.grid_generation import get_node_coord


def init_f_vector(n_x):
    """
    Инициализация положения границы фазового перехода
    :param n_x: размерность (число шагов по X)
    :return: массив с координатами границы ф.п.
    """
    F = np.empty(n_x)

    # Гладкая ступенька
    # F[:] = [0.2 + 0.5 / (1.0 + math.exp(-20.0 * (i * dx - 0.5))) for i in range(0, n_x)]

    # Плоскость
    # F[:] = a

    # Парабола
    # for i in range(n_x):
    #      F[i] = a + 2*(i*dx - W/2)*(i*dx - W/2)  # Парабола f(x, t=0) = 3*(x - W/2)^2 + a

    # Внешний угол
    # F[:] = [0.1 + i*dx if i*dx < 0.5 else 1.1 - i*dx for i in range(n_x)]

    # Угол
    # F[:] = [0.8 - i*dx if i*dx < 0.5 else i*dx - 0.2 for i in range(n_x)]

    # Для теста
    # F[:] = [3.0 - 1.5 * math.exp(-(i * dx - 0.5) ** 2 / 0.01) for i in range(n_x)]

    # Трещина-гауссиана
    F[:] = [10.0 - 5.0 * math.exp(-(i * dx - 0.5) ** 2 / 0.001) for i in range(n_x)]

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


def recalculate_boundary(F, F_2, T):
    """
    Пересчёт положения границы фазового перехода в соответствии с условием Стефана в новых координатах.
    :param F: вектор значений положения границы фазового перехода
    :param T: матрица значений температуры на сетке в НОВЫХ координатах
    :return: вектор с координатами границы фазового перехода
    """
    F_new = np.copy(F)
    inv_dx = 1.0 / dx
    inv_W = 1.0 / W
    inv_gamma = 1.0 / gamma

    j_int = int(0.5 * (N_Y - 1))  # координата границы фазового перехода в новых координатах

    h_i = 1.0 - get_node_coord(j_int - 1, j_int)
    h_0_i = get_node_coord(j_int - 1, j_int) - get_node_coord(j_int - 2, j_int)

    h_w = get_node_coord(j_int + 1, j_int) - 1.0
    h_1_w = get_node_coord(j_int + 2, j_int) - get_node_coord(j_int + 1, j_int)

    for i in range(1, N_X - 1):
        df_dx = 0.5 * inv_W * inv_dx * (F_2[i + 1] - F_2[i - 1])

        dT_dy_i = ((T[j_int, i] - T[j_int - 1, i]) * (h_i + h_0_i) * (h_i + h_0_i) + (
                    T[j_int - 2, i] - T[j_int, i]) * h_i * h_i) / (h_0_i * h_i * (h_0_i + h_i))

        dT_dy_w = ((T[j_int + 1, i] - T[j_int, i]) *
                   (h_w + h_1_w) * (h_w + h_1_w) - (T[j_int + 2, i] - T[j_int, i]) * h_w * h_w) / \
                  (h_w * h_1_w * (h_w + h_1_w))

        F_new[i] = F[i] + \
                   dt * inv_gamma * (1.0 + df_dx * df_dx) * (dT_dy_i / F_2[i] - k_w * dT_dy_w / (k_ice * (H - F_2[i])))

    df_dx_0 = 0.5 * inv_W * inv_dx * (4.0 * F_2[1] - 3.0 * F_2[0] - F_2[2])

    dT_dy_i_0 = ((T[j_int, 0] - T[j_int - 1, 0]) * (h_i + h_0_i) *
                 (h_i + h_0_i) + (T[j_int - 2, 0] - T[j_int, 0]) * h_i * h_i) / (h_0_i * h_i * (h_0_i + h_i))

    dT_dy_w_0 = ((T[j_int + 1, 0] - T[j_int, 0]) *
                 (h_w + h_1_w) * (h_w + h_1_w) - (T[j_int + 2, 0] - T[j_int, 0]) * h_w * h_w) / \
                (h_w * h_1_w * (h_w + h_1_w))

    F_new[0] = F[0] + dt * inv_gamma * (1.0 + df_dx_0 * df_dx_0) * \
               (dT_dy_i_0 / F_2[0] - k_w * dT_dy_w_0 / (k_ice * (H - F_2[0])))

    df_dx_n = 0.5 * inv_W * inv_dx * (3.0 * F_2[N_X - 1] - 4.0 * F_2[N_X - 2] + F_2[N_X - 3])

    dT_dy_i_n = ((T[j_int, N_X-1] - T[j_int - 1, N_X-1]) * (h_i + h_0_i) *
                 (h_i + h_0_i) + (T[j_int - 2, N_X-1] - T[j_int, N_X-1]) * h_i * h_i) / (h_0_i * h_i * (h_0_i + h_i))

    dT_dy_w_n = ((T[j_int + 1, N_X-1] - T[j_int, N_X-1]) *
                 (h_w + h_1_w) * (h_w + h_1_w) - (T[j_int + 2, N_X-1] - T[j_int, N_X-1]) * h_w * h_w) / \
                (h_w * h_1_w * (h_w + h_1_w))

    F_new[N_X - 1] = F[N_X-1] + dt * inv_gamma * (1.0 + df_dx_n * df_dx_n) * \
                     (dT_dy_i_n / F_2[N_X-1] - k_w * dT_dy_w_n / (k_ice * (H - F_2[N_X-1])))

    return F_new
