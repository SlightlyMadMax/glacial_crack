from parameters import N_X, N_Y


def d_xx(T, j: int, i: int):
    if i == 0:
        return T[j, 2] - 2.0 * T[j, 1] + T[j, 0]
    elif i == N_X - 1:
        return T[j, N_X - 1] - 2.0 * T[j, N_X - 2] + T[j, N_X - 3]
    else:
        return T[j, i + 1] - 2.0 * T[j, i] + T[j, i - 1]


def d_yy(T, j: int, i: int):
    if j == 0:
        return T[2, i] - 2.0 * T[1, i] + T[0, i]
    elif j == N_Y - 1:
        return T[N_Y - 1, i] - 2.0 * T[N_Y - 2, i] + T[N_Y - 3, i]
    else:
        return T[j + 1, i] - 2.0 * T[j, i] + T[j - 1, i]


def d_xy(T, j: int, i: int):
    return d_yy(T, j, i + 1) - d_yy(T, j, i - 1)


def d_x(F, i: int):
    if i == 0:
        return 4.0 * F[1] - 3.0 * F[0] - F[2]
    elif i == N_X - 1:
        return 3.0 * F[N_X - 1] - 4.0 * F[N_X - 2] + F[N_X - 3]
    else:
        return F[i + 1] - F[i - 1]


def d_y(T, j: int, i: int):
    if j == 0:
        return 4.0 * T[1, i] - 3.0 * T[0, i] - T[2, i]
    elif j == N_Y - 1:
        return 3.0 * T[N_Y - 1, i] - 4.0 * T[N_Y - 2, i] + T[N_Y - 3, i]
    else:
        return T[j + 1, i] - T[j - 1, i]
