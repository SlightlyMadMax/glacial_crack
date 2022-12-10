from one_phase.temperature import init_temperature, init_f_vector, recalculate_temperature
import numpy as np
from parameters import *
from plotting import plot_temperature
import time


# Entry point of a script
if __name__ == '__main__':
    X = np.linspace(0, W, N_X, endpoint=True)
    x = np.linspace(0, 1, N_X, endpoint=True)
    y = np.linspace(0, 1, N_Y, endpoint=True)
    #
    # print('X = ' + str(X.round(3)))
    # print('x = ' + str(x.round(3)))
    # print('y = ' + str(y.round(3)))

    dx = W/(N_X - 1)
    dy = 1/(N_Y - 1)
    #
    # print('dx = ' + str(dx))
    # print('dy = ' + str(dy))

    F = init_f_vector(x=X)

    # print('F = ' + str(F.round(3)))

    T = init_temperature()

    # print('Initial temperature:')
    # print(T.round(3))

    plot_temperature(T_matrix=T * T_0, F=F, t=0, graph_id=0)

    t_step = 1
    T_new = np.copy(T)
    F_new = np.copy(F)

    while t_step < N_t:
        print("### CALCULATING NEW T, STEP = " + str(t_step) + " ###")
        T_new, F_new = recalculate_temperature(T=T_new, F=F_new, dx=dx, dy=dy)
        # print('T at ' + str(t_step) + ' step:')
        # print(T_new.round(3))
        # print('F at ' + str(t_step) + ' step:')
        # print(F_new.round(3))
        # print("### NEW TEMPERATURE CALCULATED ###")
        # print("### SAVING THE GRAPH ###")
        plot_temperature(T_matrix=T_new*T_0, F=F_new, t=round(t_step*(dt*t_0), 2), graph_id=t_step)
        t_step = t_step + 1
        # print('Waiting for 5 sec')
        # time.sleep(5)
    print("### DONE ###")
