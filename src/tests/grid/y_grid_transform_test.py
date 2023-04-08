import matplotlib.pyplot as plt
import numpy as np

N_X = 10
N_Y = 10
dx = 1.0/(N_X - 1.0)


def f(x):
    return 0.1 + 2*(x - 0.5)*(x - 0.5)


x = [dx*i for i in range(0, N_X)]
y = []
temp = 0

# plt.plot(x, f(x), lw=0.5)

for i in range(0, N_Y + 1):
    temp += 2*i/(N_Y*(N_Y+1))
    # y = [1.0 - temp for j in range(0, N + 1)]
    y = [(1.0 - temp)*f(x[j]) for j in range(0, N_Y)]
    plt.plot(x, y, 'o', color='k', markersize=3,)

# plt.ylim(0, 1)
# plt.xlim(0, 1)
plt.show()
