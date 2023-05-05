from parameters import W, H, N_X, N_Y, dt, t_0, T_0
import matplotlib.pyplot as plt
import numpy as np
from src.two_phase.non_uniform_y_grid.grid_generation import get_node_coord
from src.two_phase.non_uniform_y_grid.temperature import air_temperature


def plot_non_transformed(T, F, time: float, graph_id: int, non_uniform: bool = True):
    """
    Строит график по двумерному массиву температуры, заданному в НОВЫХ координатах.
    Осуществляет перевод в исходные координаты, построение и сохранение графика.
    :param T: двумерный массив температуры
    :param F: вектор с координатами границы фазового перехода
    :param time: время в часах
    :param graph_id: идентификатор или порядковый номер графика
    :param non_uniform: температура задана на однородной или неоднородной сетке
    :return: None
    """
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

    fig = plt.figure()
    ax = plt.axes()
    # plt.plot(X[j_int, :], Y[j_int, :], linewidth=1, color='r')  # граница ф.п.
    plt.plot(X, Y, marker=".", markersize=0.5, color='k', linestyle='none')  # сетка
    # plt.contourf(X, Y, T_0*T - T_0, 100, cmap="viridis")
    # plt.colorbar()

    if non_uniform:
        title = f"time = {round(time, 2)} h, T_air = {round(air_temperature(graph_id * dt * t_0) - 273.15, 2)} C" \
                f"\n non-uniform grid, dt = {round(dt * t_0 / 3600.0, 2)} h"
    else:
        title = f"time = {round(time, 2)} h\n dx = 1/{N_X} m, dy = 1/{N_Y} m, dt = {round(dt * t_0 / 3600.0, 2)} h"

    ax.set_title(title)
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    plt.savefig(f"graphs/temperature/T_{graph_id}.png")
    plt.show()
    # plt.close()
