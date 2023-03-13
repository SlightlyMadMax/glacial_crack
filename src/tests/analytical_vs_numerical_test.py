import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve

spy = 60.*60.*24.  # Seconds per day
Lf = 334000.0  # Latent heat of fusion (J/kg)
rho = 999.84  # Bulk density of water (kg/m3), density changes are ignored
Ks = spy*2.39  # Conductivity of ice (J/mKs)
cs = 1943.  # Heat capacity of ice (J/kgK) - ** Van der Veen uses 2097 but see Tr and Aschwanden 2012)
ks = Ks/(919.4*cs)  # Cold ice diffusivity (m2/sec)


# Problem Constants
s0 = 0.2
t0 = 0.
Tm = 0.0
T_ = -20.0


# position of the phase boundary
def MeltLoc(lm, t):
    return s0 + 2*lm*(t-t0)**.5


def Bs(lm):
    return (Tm-T_)/(erf(lm*ks**(-.5)))


def equation_for_lambda(lm):
    lhs = rho*Lf*lm
    rhs = -Ks*Bs(lm)*np.pi**(-.5)*np.exp(-lm**2*ks**(-1))*ks**(-.5)
    return lhs-rhs


lam = fsolve(equation_for_lambda, 1.)[0]

ts = np.arange(0, 36, 1)

num = [0.2, 0.25884774861572457, 0.3016078559176033, 0.3389919954384831, 0.3726304431873688, 0.4034653136595022, 0.4320994789965059, 0.4589463887412189, 0.4843042226542184, 0.5083964284869843, 0.531395584203004, 0.5534382738868037, 0.5746347973372544, 0.5950757506697477, 0.6148366287304986, 0.633981131011398, 0.652563591369823, 0.6706307997224468, 0.6882233919588481, 0.7053769269458907, 0.7221227326489171, 0.7384885791325418, 0.7544992198633753, 0.7701768315093169, 0.7855413745738962, 0.8006108916164385, 0.8154017557743516, 0.8299288793506097, 0.8442058900407686, 0.8582452807321619, 0.8720585375638911, 0.8856562499837001, 0.8990482058019319, 0.9122434736690268, 0.9252504749526341, 0.9380770466333911]
plt.plot(ts, MeltLoc(lam, ts), 'k', lw=2)
plt.plot(ts, num, 'r', lw=2)

plt.ylim(0.2, 1.2)
plt.xlim(0, 36)

plt.ylabel('meters')
plt.xlabel('days')
plt.savefig('comparison')
