import math
import numpy as np
import matplotlib.pyplot as plt


N_Y = 100
N_X = 10

alpha = 5.0

dx = 1.0/(N_X - 1.0)

j_int = int(0.5*(N_Y - 1))  # координата границы фазового перехода в новых координатах


def get_node_coord(t: float):
    return 1.0 - math.log(2.0/t - 1.0)/alpha


z = [2*i / (N_Y - 1) for i in range(1, N_Y-1)]

# print(z)

y = [get_node_coord(z[j]) for j in range(0, N_Y-2)]

print(get_node_coord(2*j_int/(N_Y-1)))

x = [dx*i for i in range(0, N_X-2)]

X, Y = np.meshgrid(x, y)

plt.plot(X, Y, marker=".", color='k', linestyle='none')
plt.ylim(0, 2)
plt.show()
