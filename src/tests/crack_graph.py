import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

x = np.linspace(0, 1.0, 1000)

fig = plt.figure(figsize=(12, 5))
ax = plt.axes()

for i in range(42):
    loaded_data = np.load(f'../../data/crack/f_and_temp_at_{i*7200}.npz')
    plt.plot(10.05 - loaded_data['F'], x, 'k', lw=0.5)

plt.xlim(0, 5.5)
plt.ylim(0.4, 0.6)

plt.ylabel('x, м')
plt.xlabel('y, м')
# plt.savefig('../../graphs/comparison/two_phase/crack_boundary.png')
# plt.savefig('../../graphs/comparison/two_phase/crack_boundary5.eps', format='eps')
plt.savefig('../../graphs/comparison/two_phase/crack_boundary_hr.tiff', dpi=300, format='tiff')
