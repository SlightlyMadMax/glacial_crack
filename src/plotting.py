from parameters import W, H, N_X, N_Y, dt, t_0, T_0
import matplotlib.pyplot as plt
import numpy as np


def plot_non_transformed(T, F, time: float, graph_id: int, non_uniform: bool = True):
    factor = 1.0 / (N_Y * (N_Y + 1.0))

    x = np.linspace(0, 1.0, N_X)
    y = [(j + 1.0) * (2.0 * N_Y - j) * factor for j in range(N_Y)]

    X, Y = np.meshgrid(x, y)

    X = X * W
    Y = Y * F

    fig = plt.figure()
    ax = plt.axes()
    plt.contourf(X, Y, T_0*T - T_0, 20, cmap="viridis")
    plt.colorbar()

    if non_uniform:
        title = f"time = {str(time)} h\n non-uniform grid, dt = {str(round(dt * t_0 / 3600.0, 2))} h"
    else:
        title = f"time = {str(time)} h\n dx = 1/{str(N_X)} m, dy = 1/{str(N_Y)} m, dt = {str(round(dt * t_0 / 3600.0, 2))} h"

    ax.set_title(title)
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    plt.savefig(f"../graphs/temperature/T_{str(graph_id)}.png")
    plt.close()
    # plt.show()


def plot_temperature(T, time: float, graph_id: int):
    """
    Построение графика температуры в исходных координатах
    T – матрица со значениями температуры на двумерной сетке в ИСХОДНЫХ координатах
    time – время
    graph_id – id графика
    """

    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes()
    plt.imshow(T
               , extent=[0, W, 0, H]
               , origin='lower'
               , cmap='winter'
               , interpolation='none'
               , vmin=-10
               , vmax=0
               )
    plt.colorbar()
    ax.set_title(
        'time = ' + str(time) + ' h\n' +
        'dx = 1/' + str(N_X) + ' m, dy = 1/' + str(N_Y) +
        ' m, dt = ' + str(round(dt * t_0 / 3600.0, 2)) + ' h'
    )
    ax.set_xlabel('x, m')
    ax.set_ylabel('y, m')
    plt.savefig('../graphs/temperature/T_' + str(graph_id) + '.png')
    plt.show()
