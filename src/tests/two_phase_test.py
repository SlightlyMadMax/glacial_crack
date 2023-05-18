import numpy as np
import matplotlib.pyplot as plt

loaded_0 = np.load('../../data/1/f_and_temp_at_0.npz')
loaded_1 = np.load('../../data/1/f_and_temp_at_1200.npz')
loaded_2 = np.load('../../data/1/f_and_temp_at_3600.npz')
loaded_3 = np.load('../../data/1/f_and_temp_at_9600.npz')

loaded_2_0 = np.load('../../data/2/f_and_temp_at_0.npz')
loaded_2_1 = np.load('../../data/2/f_and_temp_at_1200.npz')
loaded_2_2 = np.load('../../data/2/f_and_temp_at_3600.npz')
loaded_2_3 = np.load('../../data/2/f_and_temp_at_9600.npz')

x = np.linspace(0, 1.0, 200)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.set_title('a')

ax1.set_xlim(0.0, 1.0)
ax1.set_ylim(0.0, 1.0)

ax1.plot(x, loaded_0['F'], 'blue', lw=0.5, label="t = 0")
ax1.plot(x, loaded_1['F'], 'orange', lw=0.5, label="t = 25 days")
ax1.plot(x, loaded_2['F'], 'violet', lw=0.5, label="t = 75 days")
ax1.plot(x, loaded_3['F'], 'green', lw=0.5, label="t = 200 days")

ax1.legend()
ax1.set_ylabel('y, м')
ax1.set_xlabel('x, м')


ax2.set_title('б')

ax2.set_xlim(0.0, 1.0)
ax2.set_ylim(0.0, 1.0)

ax2.plot(x, loaded_2_0['F'], 'blue', lw=0.5, label="t = 0")
ax2.plot(x, loaded_2_1['F'], 'orange', lw=0.5, label="t = 25 days")
ax2.plot(x, loaded_2_2['F'], 'violet', lw=0.5, label="t = 75 days")
ax2.plot(x, loaded_2_3['F'], 'green', lw=0.5, label="t = 200 days")

ax2.legend()
ax2.set_ylabel('y, м')
ax2.set_xlabel('x, м')

plt.savefig('../../graphs/comparison/two_phase/1d_comp.png')
plt.savefig('../../graphs/comparison/two_phase/1d_comp.eps', format="eps")
