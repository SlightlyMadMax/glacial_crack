import numpy as np
from parameters import *


def init_temperature():
    """
    Задание начального распределения температуры на сетке в НОВЫХ координатах.
    """
    T = np.empty((N_Y, N_X))
    T[:, :] = T_ice/T_0  # Температура льда
    T[N_Y-1, :] = 1.0  # Температура фазового перехода
    return T


def reverse_transform(T, F):
    """
    Пересчёт температуры в исходные координаты.
    T – матрица значений температуры на сетке в НОВЫХ координатах.
    F – вектор значений положения границы фазового перехода.
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

