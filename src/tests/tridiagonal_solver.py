import numpy as np
import scipy as sp
import time
from linear_algebra.tdma import tdma


def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)


def test_tridiagonal_solvers(N: int):
    dt = 0.1
    dx = 1.0/1000.0
    inv_dx = 1/dx

    a = c = np.ones((N - 1,)) * 0.5 * inv_dx * inv_dx * -dt
    b = np.ones(N) * (1 + dt * inv_dx * inv_dx)

    A_sparse = sp.sparse.diags([a, b, c], [-1, 0, 1], format='csr')

    f = np.random.rand(N)*20 + 250

    # Calculate solving time for scipy

    t_2 = time.process_time()

    x_2 = sp.sparse.linalg.spsolve(A_sparse, f)

    r_2 = time.process_time() - t_2

    # Calculate solving time for my tdma
    b = b[1:N]

    t_3 = time.process_time()

    x_3 = tdma(
        alpha_0=0,
        beta_0=x_2[0],
        u_r=x_2[N-1],
        a=a,
        b=b,
        c=c,
        f=f
    )

    r_3 = time.process_time() - t_3

    return [r_2, r_3]


sum_1 = sum_2 = 0

for i in range(100):
    tmp = test_tridiagonal_solvers(1000)
    sum_1 += tmp[0]
    sum_2 += tmp[1]

print("Scipy avg: " + str(sum_1/100) + "\n" + "TDMA avg: " + str(sum_2/100) + "\n")


