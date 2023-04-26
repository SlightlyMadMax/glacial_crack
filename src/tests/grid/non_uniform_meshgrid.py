import matplotlib.pyplot as plt
import numpy as np

# Высота
H = 1.2
# Ширина
W = 1.0

N_X = 10  # Число узлов сетки по оси X
N_Y = 10  # Число узлов сетки по оси Y


def z2(x, y):
    return np.sqrt(x ** 2 + y ** 2)


f = lambda x: 0.1 + 2*(x - W/2)*(x - W/2)

x = np.linspace(0, W, N_X)
y = np.linspace(0, H, N_Y)


X, Y = np.meshgrid(x, y)

Y = Y*f(x)

Z = z2(X, Y)

# cетка
plt.plot(X, Y, marker=".", color='k', linestyle='none')

# на обычном  контурном графике
# plt.contour(X, Y, Z, colors='black')

plt.contourf(X, Y, Z, 20, cmap='viridis')
plt.colorbar()

plt.show()
