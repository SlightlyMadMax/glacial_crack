import math
import numpy as np
import matplotlib.pyplot as plt


N_Y = 10
N_X = 10
alpha = 3
dx = 1.0/(N_X - 1.0)


def sigma(t: float):
    return 1.0 / (1.0 + math.exp(-alpha * (t - 0.5)))


z = [2 * i / (N_Y - 2) for i in range(1, N_Y)]

y = [sigma(z[j]) for j in range(0, N_Y-1)]

print(y)

x = [dx*i for i in range(1, N_X)]

X, Y = np.meshgrid(x, y)

plt.plot(X, Y, marker=".", color='k', linestyle='none')

plt.show()
