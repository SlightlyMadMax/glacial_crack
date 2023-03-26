import matplotlib.pyplot as plt
import numpy as np


N = 10


def f(x):
    return 0.1 + 2*(x - 0.5)*(x - 0.5)


x = np.arange(0.0, 1.0 + 1.0/N, 1.0/N)
y = []
temp = 0

# plt.plot(x, f(x), lw=0.5)

for i in range(0, N + 1):
    temp += 2*i/(N*(N+1))
    # y = [1.0 - temp for j in range(0, N + 1)]
    y = [(1.0 - temp)*f(x[j]) for j in range(0, N+1)]
    plt.plot(x, y, 'o', color='k', markersize=3,)

# plt.ylim(0, 1)
# plt.xlim(0, 1)
plt.show()
