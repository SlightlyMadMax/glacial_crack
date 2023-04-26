import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

spd = 60.*60.*24.  # Seconds per day
Lf = 334000.0  # Latent heat of fusion (J/kg)
rho_ice = 918.9
rho_w = 999.84
Ks = spd*2.1  # Conductivity of ice (J/mKs)
cs = 2097.  # Heat capacity of ice (J/kgK) - ** Van der Veen uses 2097 but see Tr and Aschwanden 2012)
ks = Ks*2/((rho_ice+rho_w)*cs)  # Cold ice diffusivity (m2/sec)


# Problem Constants
s0 = 9.0
t0 = 0.
Tm = 0.0
T_ = -10.0


ts = np.arange(0, 36, 1)


def boundary(t):
    return s0 + 2*cs*(Tm - T_)*(ks*t)**.5/(np.pi**.5*Lf)


num = [9.0, 9.021406806224608, 9.030356884300677, 9.037211173851972, 9.042984747797844, 9.04806884590136, 9.052663661467943, 9.056887957049538, 9.060819042661226, 9.064510582872142, 9.068001631188771, 9.07132165842397, 9.074493553348116, 9.077535513628419, 9.080462290964084, 9.08328604123409, 9.08601692292409, 9.088663529481536, 9.091233208804125, 9.093732304010903, 9.096166338043476, 9.0985401573576, 9.100858045265646, 9.103123812386258, 9.105340869564332, 9.107512287185859, 9.10964084380623, 9.111729066294624, 9.113779263180076, 9.11579355250564, 9.117773885214305, 9.11972206487731, 9.121639764412317, 9.123528540312833, 9.125389844811123, 9.127225036319247]


error = []
for i in range(1, 36):
    error.append(100*abs(num[i]-boundary(ts[i]))/(boundary(ts[i])-9.0))

plt.plot(ts, num, 'r', lw=0.5, label='Numerical')
plt.plot(ts, boundary(ts), 'k', lw=0.5, label='Analytical')
plt.legend()
plt.ylim(s0, s0+0.15)
plt.xlim(0, 36)
plt.ylabel('m')
plt.xlabel('days')
plt.savefig('../../graphs/comparison/1d.png')

plt.clf()
plt.plot(ts[1:36], error, 'k', lw=2, label='Relative error, %')
plt.legend()

plt.ylim(0, 10)
plt.xlim(0, 36)

plt.ylabel('%')
plt.xlabel('days')
plt.savefig('../../graphs/error/1d.png')
