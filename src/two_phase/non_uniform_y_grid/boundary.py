import numpy as np
from parameters import *
from src.two_phase.non_uniform_y_grid.grid_generation import get_node_coord


def init_f_vector(n_x):
    """
    Задание начального положения границы фазового перехода.
    """
    F = np.empty(n_x)
    for i in range(n_x):
        if i * dx < 0.4 or i * dx > 0.6:  # Подобие трещины
            F[i] = 1.0
        else:
            F[i] = 3.8 - 7.0 * i * dx if i * dx < 0.5 else 7.0 * i * dx - 3.2
    return F


def recalculate_boundary(F, T):
    """
    Пересчёт положения границы фазового перехода в соответствии с условием Стефана в новых координатах.
    F – вектор значений положения границы фазового перехода.
    T – матрица значений температуры на сетке в НОВЫХ координатах.
    dx, dy – шаги по x и y на сетке в НОВЫХ координатах.
    """

    F_new = np.copy(F)
    inv_dx = 1.0 / dx
    inv_W = 1.0 / W
    inv_gamma = 1.0 / gamma

    j_int = int(0.5 * (N_Y - 1))  # координата границы фазового перехода в новых координатах

    h_i = get_node_coord(2 * j_int / (N_Y - 1)) - get_node_coord(2 * (j_int - 1) / (N_Y - 1))
    h_0_i = get_node_coord(2 * (j_int - 1) / (N_Y - 1)) - get_node_coord(2 * (j_int - 2) / (N_Y - 1))

    h_w = get_node_coord(2 * (j_int + 1) / (N_Y - 1)) - get_node_coord(2 * j_int / (N_Y - 1))
    h_1_w = get_node_coord(2 * (j_int + 2) / (N_Y - 1)) - get_node_coord(2 * (j_int + 1) / (N_Y - 1))

    for i in range(1, N_X - 1):
        df_dx = 0.5 * inv_W * inv_dx * (F[i + 1] - F[i - 1])

        dT_dy_i = ((T[j_int, i] - T[j_int - 1, i]) * (h_i + h_0_i) * (h_i + h_0_i) + (
                    T[j_int - 2, i] - T[j_int, i]) * h_i * h_i) / (h_0_i * h_i * (h_0_i + h_i))

        dT_dy_w = ((T[j_int + 1, i] - T[j_int, i]) *
                   (h_w + h_1_w) * (h_w + h_1_w) - (T[j_int + 2, i] - T[j_int, i]) * h_w * h_w) / \
                  (h_w * h_1_w * (h_w + h_1_w))

        F_new[i] = F[i] + \
                   dt * inv_gamma * (1.0 + df_dx * df_dx) * (dT_dy_i / F[i] - k_w * dT_dy_w / (k_ice * (H - F[i])))

    df_dx_0 = 0.5 * inv_W * inv_dx * (4.0 * F[1] - 3.0 * F[0] - F[2])

    dT_dy_i_0 = ((T[j_int, 0] - T[j_int - 1, 0]) * (h_i + h_0_i) *
                 (h_i + h_0_i) + (T[j_int - 2, 0] - T[j_int, 0]) * h_i * h_i) / (h_0_i * h_i * (h_0_i + h_i))

    dT_dy_w_0 = ((T[j_int + 1, 0] - T[j_int, 0]) *
                 (h_w + h_1_w) * (h_w + h_1_w) - (T[j_int + 2, 0] - T[j_int, 0]) * h_w * h_w) / \
                (h_w * h_1_w * (h_w + h_1_w))

    F_new[0] = F[0] + dt * inv_gamma * (1.0 + df_dx_0 * df_dx_0) * \
               (dT_dy_i_0 / F[0] - k_w * dT_dy_w_0 / (k_ice * (H - F[0])))

    df_dx_n = 0.5 * inv_W * inv_dx * (3.0 * F[N_X - 1] - 4.0 * F[N_X - 2] + F[N_X - 3])

    dT_dy_i_n = ((T[j_int, N_X-1] - T[j_int - 1, N_X-1]) * (h_i + h_0_i) *
                 (h_i + h_0_i) + (T[j_int - 2, N_X-1] - T[j_int, N_X-1]) * h_i * h_i) / (h_0_i * h_i * (h_0_i + h_i))

    dT_dy_w_n = ((T[j_int + 1, N_X-1] - T[j_int, N_X-1]) *
                 (h_w + h_1_w) * (h_w + h_1_w) - (T[j_int + 2, N_X-1] - T[j_int, N_X-1]) * h_w * h_w) / \
                (h_w * h_1_w * (h_w + h_1_w))

    F_new[N_X - 1] = F[N_X-1] + dt * inv_gamma * (1.0 + df_dx_n * df_dx_n) * \
                     (dT_dy_i_n / F[N_X-1] - k_w * dT_dy_w_n / (k_ice * (H - F[N_X-1])))

    return F_new
