import numpy as np
from parameters import *


def init_temperature(F):
    """
    Задает начальное распределение температуры на сетке в НОВЫХ координатах.
    :return: двумерный массив со значениями температуры в начальный момент времени
    """
    T = np.empty((N_Y, N_X))
    T[:, :] = T_ice/T_0  # Температура льда
    T[N_Y-1, :] = 1.0  # Температура фазового перехода

    # Линейное уменьшение температуры с глубиной
    # f_min = np.amin(F)
    # for i in range(N_X):
    #     for j in range(N_Y - 1, -1, -1):
    #         if F[i] * (N_Y - 1 - j) / (f_min * (N_Y - 1)) < 1.0:
    #             T[j, i] = T_0/T_0 - (F[i] * (N_Y - 1 - j) * (T_0 - T_ice)) / (f_min * T_0 * (N_Y - 1))
    #         else:
    #             T[j, i] = T_ice / T_0
    return T


def reverse_transform(T, F):
    """
    Пересчёт температуры в исходные координаты.
    :param T: двумерный массив со значениями температуры в НОВЫХ координатах
    :param F: вектор с координатами границы фазового перехода
    :return: двумерный массив со значениями температуры в ИСХОДНЫХ координатах
    """
    new_H = int(H*N_Y)
    new_W = int(W*N_X)

    T_new = np.zeros((new_H, new_W))

    for i in range(N_X):
        new_x = int(round(i*W))
        for j in range(N_Y):
            new_y = int(round(j * F[i]))
            if new_y < new_H and new_x < int(W*N_X):
                T_new[new_y, new_x] = T_0*T[j, i] - T_0  # Пересчитываем в ИСХОДНЫЕ координаты и переходим к °С
    return T_new
