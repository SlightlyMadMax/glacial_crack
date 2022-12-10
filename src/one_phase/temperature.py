import numpy as np
from parameters import *
from one_phase.fd_scheme import predict_correct


def init_f_vector(x):
    n_x = len(x)
    F = np.empty(n_x)
    for i in range(0, n_x):
        F[i] = h if x[i] <= s else x[i] + h - s
        # F[i] = h + x[i]*x[i]
    return F


def init_temperature():
    T = np.empty((N_Y, N_X))
    T[:, :] = T_ice/T_0
    T[N_Y-1, :] = T_0/T_0
    return T


def recalculate_boundary(F, T, dy):
    # print((3.0*T[N_Y-1, :] - 4*T[N_Y-2, :] + T[N_Y-3, :]))
    F[:] = F[:] + k_ice*dt*(3.0*T[N_Y-1, :] - 4*T[N_Y-2, :] + T[N_Y-3, :])/(2*dy*gamma*F[:])
    # print(k_ice*(T_0/T_0 - T[N_Y-2, :])*dt/(2*dy*gamma*F[:]))
    return F


def recalculate_temperature(T, F, dx, dy):
    F_old = np.copy(F)
    F_new = recalculate_boundary(F=np.copy(F_old), T=T, dy=dy)
    T_new = predict_correct(T, F_new, F_old, dx=dx, dy=dy)

    return T_new, F_new
