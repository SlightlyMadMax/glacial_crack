import numpy as np
import matplotlib.pyplot as plt
import math

N_Y = 200
N_X = 200
W = 1.0
H = 1.0
T_0 = 273.15
s_ice = s_w = 8.0


def get_node_coord(j: int, j_int: int) -> float:
    """
    Функция для генерации неравномерной сетки. Сгущает узлы ближе к фазовой границе.
    :param t: координата на равномерной сетке на интервале (0, 1).
    :return: координата на неравномерной сетке.
    """
    if j == 0:
        return 0.0
    elif j == j_int:
        return 1.0
    elif j == N_Y - 1:
        return 2.0
    elif j < j_int:
        t = 2 * j / (N_Y - 2)
        return 1.0 - math.exp(-s_ice * t) + math.exp(-s_ice)
    else:
        t = (2 * N_Y - 4 - 2 * j) / (N_Y - 2)
        return 1.0 + math.exp(-s_w * t) - math.exp(-s_w)


plt.rcParams.update({'font.size': 12})

loaded_0 = np.load('../../data/f_and_temp_at_0.npz')
loaded_1 = np.load('../../data/f_and_temp_at_720.npz')
loaded_2 = np.load('../../data/f_and_temp_at_9360.npz')

x = np.linspace(0, 1.0, 200)
y = np.empty(N_Y)
j_int = int(0.5 * (N_Y - 1))

for j in range(N_Y):
    y[j] = get_node_coord(j, j_int)

X, Y = np.meshgrid(x, y)
X = X * W

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 4))

ax1.set_title('a')

ax1.set_xlim(0.0, 1.0)
ax1.set_ylim(0.0, 1.0)

F = loaded_0['F']
T = loaded_0['T']
for i in range(N_X):
    for j in range(N_Y):
        if j <= j_int:
            Y[j, i] = Y[j, i] * F[i]
        else:
            Y[j, i] = (Y[j, i] - 1.0) * (H - F[i]) + F[i]
ax1.plot(X[j_int, :], Y[j_int, :], linewidth=1, color='r')  # граница ф.п.
ax1.contourf(X, Y, T_0 * T - T_0, 100, cmap="viridis")

ax1.set_aspect("equal")
ax1.set_ylabel('y, м')
ax1.set_xlabel('x, м')


ax2.set_title('б')

ax2.set_xlim(0.0, 1.0)
ax2.set_ylim(0.0, 1.0)

X, Y = np.meshgrid(x, y)
X = X * W

F = loaded_1['F']
T = loaded_1['T']
for i in range(N_X):
    for j in range(N_Y):
        if j <= j_int:
            Y[j, i] = Y[j, i] * F[i]
        else:
            Y[j, i] = (Y[j, i] - 1.0) * (H - F[i]) + F[i]
ax2.plot(X[j_int, :], Y[j_int, :], linewidth=1, color='r')  # граница ф.п.
ax2.contourf(X, Y, T_0 * T - T_0, 100, cmap="viridis")

ax2.set_aspect("equal")
ax2.set_ylabel('y, м')
ax2.set_xlabel('x, м')


ax3.set_title('в')

ax3.set_xlim(0.0, 1.0)
ax3.set_ylim(0.0, 1.0)

X, Y = np.meshgrid(x, y)
X = X * W

F = loaded_2['F']
T = loaded_2['T']
for i in range(N_X):
    for j in range(N_Y):
        if j <= j_int:
            Y[j, i] = Y[j, i] * F[i]
        else:
            Y[j, i] = (Y[j, i] - 1.0) * (H - F[i]) + F[i]
ax3.plot(X[j_int, :], Y[j_int, :], linewidth=1, color='r')  # граница ф.п.
fuck = ax3.contourf(X, Y, T_0 * T - T_0, 100, cmap="viridis")

ax3.set_aspect("equal")
ax3.set_ylabel('y, м')
ax3.set_xlabel('x, м')

fig.colorbar(fuck, ax=ax3, orientation='horizontal')

plt.show()
# plt.savefig('../../graphs/comparison/two_phase/1d_comp.png')
# plt.savefig('../../graphs/comparison/two_phase/1d_comp.eps', format="eps")
plt.savefig('../../graphs/comparison/two_phase/1d_comp_2.png')
