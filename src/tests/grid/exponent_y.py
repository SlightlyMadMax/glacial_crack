import math
import numpy as np
import matplotlib.pyplot as plt


N_Y = 10
N_X = 10
alpha = 4
dx = 1.0/(N_X - 1.0)


def get_node_coord(t: float):
    return 1.0 - math.exp(-alpha*t) + math.exp(-alpha)


z = [i / (N_Y - 1) for i in range(1, N_Y - 1)]

y = [get_node_coord(z[j]) for j in range(0, N_Y - 2)]

print(y)

x = [dx*i for i in range(0, N_X - 2)]

X, Y = np.meshgrid(x, y)

plt.plot(X, Y, marker=".", color='k', linestyle='none')
plt.ylim(0, 1)
plt.show()
