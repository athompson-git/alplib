#import sys
#sys.path.append("alplib/dark_arc")

from flux_DarkARC import XeResponse

import numpy as np
from numpy import pi, sqrt, log10

import matplotlib.pyplot as plt

me = 0.511
Gf = 1.16637e-5
gae = 0.5
gve = 2*0.23 + 0.5

# Set the range
Enu0 = 0.4  # 20 keV
Enu1 = 0.5  # 0.5 MeV

# Set the weight function = differential cross section
def dSigmadq_nu_edm(q, Enu=Enu0):
    T = q**2 / (2*me)
    return (q/me) * (Gf**2 * me)/(2*pi) * ((gae + gve)**2 + (T*me)/Enu**2 * (gae**2 - gve**2) + (gae - gve)**2 * (1 - T/Enu)**2)

def dSigmadT_nu_edm(T, Enu=Enu0):
    return (Gf**2 * me)/(2*pi) * ((gae + gve)**2 + (T*me)/Enu**2 * (gae**2 - gve**2) + (gae - gve)**2 * (1 - T/Enu)**2)



# Get the cross section shape integrated over q
genDA = XeResponse(shell="total", keV=0.001)

T_list = np.logspace(-6, 0, 100)
diffxs_ion = np.array([genDA.QIntegrate(T, dSigmadq_nu_edm, 0.00001, sqrt(2*me*Enu0), nsamples=1000) for T in T_list])
diffxs_free = np.array([54*dSigmadT_nu_edm(T) for T in T_list])


plt.plot(T_list, diffxs_free, label="free")
plt.plot(T_list, diffxs_ion, label="PI")
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r"$E_e$ [MeV]")
plt.ylabel(r"$\dfrac{d\sigma}{dE_e}$ [MeV$^{-3}$]")
plt.title(r"$E_\nu = 400$ keV", loc='right')
plt.legend()
plt.show()
plt.close()
