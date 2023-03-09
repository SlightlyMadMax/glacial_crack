import numpy as np
from parameters import *


def init_f_vector(dx, n_x):
    """
    Задание начального положения границы фазового перехода.
    """
    F = np.empty(n_x)
    for i in range(n_x):
        # F[i] = h
        F[i] = h + 3*(i*dx - W/2)*(i*dx - W/2)  # Парабола f(x, t=0) = 3*(x - W/2)^2 + h
    return F


def init_temperature():
    """
    Задание начального распределения температуры на сетке в НОВЫХ координатах.
    """
    T = np.empty((N_Y, N_X))
    T[:, :] = T_ice/T_0  # Температура льда
    T[N_Y-1, :] = T_0/T_0  # Температура фазового перехода
    return T


def recalculate_boundary(F, T, dx, dy):
    """
    Пересчёт положения границы фазового перехода в соответствии с условием Стефана в новых координатах.
    F – вектор значений положения границы фазового перехода.
    T – матрица значений температуры на сетке в НОВЫХ координатах.
    dx, dy – шаги по x и y на сетке в НОВЫХ координатах.
    """
    F_new = np.copy(F)
    inv_dy = 1.0/dy
    inv_dx = 1.0/dx
    inv_W = 1.0/W
    inv_gamma = 1.0/gamma

    for i in range(1, N_X-1):
        inv_F = 1.0/F[i]
        F_new[i] = F[i] + dt*inv_gamma*inv_F*(1.0 + (0.5*inv_W*inv_dx*(F[i+1]-F[i-1]))**2) *\
                   (0.5*inv_dy*(3.0 * T[N_Y - 1, i] - 4.0 * T[N_Y - 2, i] + T[N_Y - 3, i]))

    F_new[0] = F[0] + dt*inv_gamma*(1.0 + (0.5*inv_W*inv_dx*(4.0*F[1]-3.0*F[0]-F[2]))**2) *\
               (0.5*inv_dy*(3.0 * T[N_Y-1, 0] - 4.0 * T[N_Y-2, 0] + T[N_Y-3, 0]))/F[0]
    F_new[N_X-1] = F[N_X-1] + dt*inv_gamma*(1.0 + (0.5*inv_W*inv_dx*(3.0*F[N_X-1]-4.0*F[N_X-2]+F[N_X-3]))**2) *\
                (0.5*inv_dy*(3.0 * T[N_Y-1, N_X-1] - 4.0 * T[N_Y-2, N_X-1] + T[N_Y-3, N_X-1]))/F[N_X-1]
    return F_new


def reverse_transform(T, F):
    """
    Пересчёт температуры в исходные координаты.
    T – матрица значений температуры на сетке в НОВЫХ координатах.
    F – вектор значений положения границы фазового перехода.
    """
    T_new = np.zeros((int(H*N_Y), int(W*N_X)))
    for j in range(N_Y):
        for i in range(N_X):
            if j*F[int(i*W)] < H*N_Y:
                T_new[int(j*F[int(i*W)]), int(W*i)] = T_0*(T[j, i] - 1.0)  # Пересчитываем в ИСХОДНЫЕ координаты и переходим к °С
    return T_new

