import math
import numpy as np
import matplotlib.pyplot as plt

N_X = 100
N_Y = 20
W = 1.0
dx = 1.0/(N_X-1.0)


def f(x: float):
    # return 0.5 - 0.8 * x if x < W / 2 else 0.8 * x - 0.3
    # return 0.5 + 0.25*math.sin(2*math.pi*x)
    # return 0.5
    return 3.0 - 2.0 * math.exp(-(x - 0.5) ** 2 / 0.005)


# def calculate_x_mesh(F):
#     dF_dx = np.empty(N_X)
#     for i in range(1, N_X - 1):
#         dF_dx[i] = 0.5 * abs(F[i + 1] - F[i - 1]) / dx
#     dF_dx[0] = 0.5 * abs(4.0 * F[1] - 3.0 * F[0] - F[2]) / dx
#     dF_dx[N_X - 1] = 0.5 * abs(3.0 * F[N_X - 1] - 4.0 * F[N_X - 2] + F[N_X - 3]) / dx
#
#     P = np.zeros(N_X)
#     for i in range(1, N_X):
#         P[i] = P[i - 1] + 0.5 * dx * (dF_dx[i - 1] + dF_dx[i])
#
#     eq_grid = np.empty(N_X)
#     eq_grid[:] = [P[N_X - 1] * i * dx for i in range(0, N_X)][:]
#
#     return np.interp(eq_grid, P, [i * dx for i in range(0, N_X)])
#
#
F = np.empty(N_X)
F[:] = [f(dx*i) for i in range(0, N_X)][:]

# result = calculate_x_mesh(F)

result = np.empty(N_X)

b = 10

for i in range(0, N_X):
    t = i * dx

    if i == 0:
        result[i] = 0.0
    elif i == N_X - 1:
        result[i] = 1.0
    else:
        result[i] = 0.5 - math.log(1.0/t - 1.0)/b

print(len(result))

y = [dx*i for i in range(0, N_Y)]

plt.plot([dx*i for i in range(0, N_X)], F, lw=0.5)

temp = []
for j in range(1, N_Y-1):
    temp = [y[j] for i in range(0, N_X)]
    plt.plot(result, temp, 'o', color='k', markersize=3,)

plt.show()
