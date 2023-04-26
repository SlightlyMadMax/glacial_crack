import math
import numpy as np
import matplotlib.pyplot as plt

N_X = 15
N_Y = 15
W = 1.0
dx = 1.0/(N_X-1.0)


def f(x: float):
    # return 0.5 - 0.8 * x if x < W / 2 else 0.8 * x - 0.3
    # return 0.5 + 0.25*math.sin(2*math.pi*x)
    return 0.1 + 2.0*(x - W/2.0)*(x - W/2.0)


def calculate_x_mesh(F):
    dF_dx = np.empty(N_X)
    for i in range(1, N_X - 1):
        dF_dx[i] = 0.5 * abs(F[i + 1] - F[i - 1]) / dx
    dF_dx[0] = 0.5 * abs(4.0 * F[1] - 3.0 * F[0] - F[2]) / dx
    dF_dx[N_X - 1] = 0.5 * abs(3.0 * F[N_X - 1] - 4.0 * F[N_X - 2] + F[N_X - 3]) / dx

    P = np.zeros(N_X)
    for i in range(1, N_X):
        P[i] = P[i - 1] + 0.5 * dx * (dF_dx[i - 1] + dF_dx[i])

    eq_grid = np.empty(N_X)
    eq_grid[:] = [P[N_X - 1] * i / (N_X - 1.0) for i in range(0, N_X)][:]

    return np.interp(eq_grid, P, [i * dx for i in range(0, N_X)])


F = np.empty(N_X)
F[:] = [f(dx*i) for i in range(0, N_X)][:]

result = calculate_x_mesh(F)

temp = 0.0

plt.plot([dx*i for i in range(0, N_X)], F, lw=0.5)

for i in range(0, N_Y+1):
    temp += 2*i/(N_Y*(N_Y+1))
    y = [(1.0 - temp)*f(result[j]) for j in range(0, N_Y)]
    plt.plot(result, y, 'o', color='k', markersize=3,)

plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()
