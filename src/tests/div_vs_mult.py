import time


def div_vs_mult():
    dx = 1.0/1000.0
    inv_dx = 1.0/dx

    a = 0
    t_start = time.process_time()

    for i in range(100000):
       a = i + i/dx

    r_1 = time.process_time() - t_start

    a = 0
    t_start = time.process_time()

    for i in range(100000):
        a = i + i * inv_dx

    r_2 = time.process_time() - t_start

    return r_1, r_2


sum_1 = sum_2 = 0

for i in range(100):
    r_1, r_2 = div_vs_mult()
    sum_1 += r_1
    sum_2 += r_2

print("Division avg time: " + str(sum_1/100) + "\nMultiplication avg time: " + str(sum_2/100))
