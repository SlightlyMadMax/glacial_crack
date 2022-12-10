from parameters import W, H, N_Y, N_X, T_0
import numpy as np


def plot_temperature(T_matrix, F, t: float, graph_id: int):
    T = np.zeros((int(0.1*N_Y), N_X))
    for j in range(0, N_Y):
        for i in range(0, N_X):
            T[int(j*F[i]), i] = T_matrix[j, i] - T_0

    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes()
    # print('For printing:')
    # print(T.round(3))
    plt.imshow(T
               , extent=[0, W, 0, H]
               , origin='lower'
               , cmap='winter'
               , interpolation='none'
               , vmin=-20
               , vmax=0
               )
    plt.colorbar()
    ax.set_title('time = ' + str(t) + 's')
    ax.set_xlabel('x, m')
    ax.set_ylabel('y, m')
    plt.savefig('../graphs/T_' + str(graph_id) + '.png')
    plt.show()


