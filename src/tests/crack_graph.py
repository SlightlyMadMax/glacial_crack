import numpy as np
import matplotlib.pyplot as plt
import math
N_X = 1000
N_Y = 200
s_w = 4.5
s_ice = 8.0
T_0 = 273.15
W = 1.0
H=10.05
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


plt.rcParams.update({'font.size': 14})

loaded_data = np.load(f'../../data/crack/f_and_temp_at_{295200}.npz')
F = loaded_data['F']
T = loaded_data['T']

x = np.linspace(0, 1.0, N_X)
y = np.empty(N_Y)
j_int = int(0.5 * (N_Y - 1))

for j in range(N_Y):
    y[j] = get_node_coord(j, j_int)

X, Y = np.meshgrid(x, y)

X = X * W
for i in range(N_X):
    for j in range(N_Y):
        if j <= j_int:
            Y[j, i] = Y[j, i] * F[i]
        else:
            Y[j, i] = (Y[j, i] - 1.0) * (H - F[i]) + F[i]


plt.rcParams.update({'font.size': 12})
fig = plt.figure()
ax = plt.axes()

ax.set_aspect("equal")
plt.plot(X[j_int, :], Y[j_int, :], linewidth=1, color='r', label='Граница ф.п.')  # граница ф.п.
plt.contourf(X, Y, T_0*T - T_0, 100, cmap="viridis")

plt.ylim((9.9, 10.05))
plt.xlim((0.3, 0.7))

plt.colorbar(orientation='horizontal')


plt.ylabel('x, м')
plt.xlabel('y, м')
# plt.savefig('../../graphs/comparison/two_phase/crack_boundary.png')
# plt.savefig('../../graphs/comparison/two_phase/crack_boundary5.eps', format='eps')
plt.savefig('../../graphs/t1.png')
