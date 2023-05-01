from parameters import W, H, N_X, N_Y, dt, t_0, T_0
import matplotlib.pyplot as plt
import numpy as np


def plot_non_transformed(T, F, time: float, graph_id: int):
    x = np.linspace(0, 1.0, N_X)
    y = np.linspace(0, 1.0, N_Y)

    X, Y = np.meshgrid(x, y)

    X = X * W
    Y[:, :] = Y[:, :] * F[:]

    fig = plt.figure()
    ax = plt.axes()
    # plt.plot(X, Y[j_int, :], marker=".", markersize=0.1, color='red', linestyle='none')  # сетка
    plt.contourf(X, Y, T_0 * T - T_0, 100, cmap="viridis")
    plt.colorbar()

    title = f"time = {str(time)} h\n dx = 1/{str(N_X)} m, dy = 1/{str(N_Y)} m, dt = {str(round(dt * t_0 / 3600.0, 2))} h"

    ax.set_title(title)
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    plt.savefig(f"graphs/temperature/T_{str(graph_id)}.png")
    plt.show()
    # plt.close()
