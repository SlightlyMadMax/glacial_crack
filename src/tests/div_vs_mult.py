import time


def div_vs_mult():
    dx = 1.0/1000.0
    inv_dx = 1.0/dx

    a = 0
    t_start = time.process_time()

    for i in range(1000000):
       a = i + i/dx

    print("Division. Elapsed time: " + str(time.process_time() - t_start))

    a = 0
    t_start = time.process_time()

    for i in range(1000000):
        a = i + i * inv_dx

    print("Multiplication. Elapsed time: " + str(time.process_time() - t_start))


div_vs_mult()
