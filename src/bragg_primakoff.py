import numpy as np
from numpy import sqrt, pi, log10, exp, cos, sin, heaviside
from scipy.special import erf
from scipy.integrate import quad

import matplotlib.pyplot as plt

from itertools import product

from constants import *

# Global Constants in keV angstroms
kALPHA = 1/137
kME = 511
kHBarC = 1.97

# Crystal constants
zGe = 32
r0Ge = 0.53
vCrys = 1e12  # 1 cubic micron in cubic angstroms
va = 181
a = 5.66


class BraggPrimakoff:
    def __init__(self):
        # Primitive basis vectors
        self.a0 = np.array([0,0,0])
        self.a1 = (a/4) * np.array([1,1,1])

        # Bravais lattice
        self.b1 = (2*pi/a) * np.array([-1, 1, 1])
        self.b2 = (2*pi/a) * np.array([1, -1, 1])
        self.b3 = (2*pi/a) * np.array([-1, 1, -1])

    # Reciprocal Lattice
    def vecG(self, mList):
        return mList[0] * self.b1 + mList[1] * self.b2 + mList[2] * self.b3

    # Atomic form factor squared
    def FA(self, q2, k):
        return (zGe * sqrt(4*pi*kALPHA) * k**2 / ( 1 / (r0Ge**2) + q2))**2

    # Structure function squared
    def S2(self, mList):
        return 4 * (cos(np.dot(self.a1, self.vecG(mList)) / 2))**2

    def S2Expanded(self, mList):
        h = mList[0]
        k = mList[1]
        l = mList[2]
        return 4 * cos(pi/4 * (h+k+l))**2 * (cos(pi*h)*(cos(pi*k)+cos(pi*l)) + cos(pi*k)*cos(pi*l) + 1)

    # Energy resolution function
    def FW(self, Ea, E1, E2):
        return 0.5 * ( erf((Ea - E1)/(sqrt(2)/2.35)) - erf((Ea - E2)/(sqrt(2)/2.35)) )

    # Incident photon vector
    def vecU(self, theta_z, phi):
        return np.array([-sin(theta_z)*cos(phi), -sin(theta_z)*sin(phi), -cos(theta_z)])

    # Bragg condition
    def Ea(self, theta_z, phi, mList):
        return kHBarC * np.dot(self.vecG(mList), self.vecG(mList)) / (2 * np.dot(self.vecU(theta_z, phi), self.vecG(mList)))

    def Ea2(self, theta, mList):
        return kHBarC * np.sqrt(np.dot(self.vecG(mList), self.vecG(mList))) / (2 * sin(theta/2))

    # Solar ALP flux
    def SolarFlux(self, Ea, gagamma):
        return (gagamma * 1e8)**2 * (5.95e14 / 1.103) * (Ea / 1.103)**3 / (exp(Ea / 1.103) - 1)


    # Getter for the list of reciprocal vectors
    def GetReciprocalLattice(self, nmax=5):
        g = []
        for mList in product(np.arange(1,nmax),repeat=3):
            if (np.sum(mList) == 0):
                continue
            if (np.all(np.array(mList) % 2 == 1) or (np.all(np.array(mList) % 2 == 0) and np.sum(np.array(mList)) % 4 == 0)):
                g.append(mList)
        return np.array(g)

    print(GetReciprocalLattice())
    for m in GetReciprocalLattice():
        print(m)


    # Bragg-Primakoff event rate
    def BraggPrimakoff(self, theta_z, phi, E1=2.0, E2=2.5, gagamma=1e-10):
        rate = 0.0
        prefactor = (gagamma / 1e6)**2 * kHBarC**2 * (vCrys / va**2) / 4  # 1e6 to convert to keV^-1
        for mList in GetReciprocalLattice():
            sineThetaBy2 = np.dot(vecU(theta_z, phi), vecG(mList)) / np.dot(vecG(mList),vecG(mList))
            sineSquaredTheta = 4 * sineThetaBy2**2 * (1 - sineThetaBy2**2)
            formFactorSquared = FA(sqrt(np.dot(vecG(mList), vecG(mList))), Ea(theta_z, phi, mList))
            rate += heaviside(Ea(theta_z, phi, mList), 0.0) * SolarFlux(Ea(theta_z, phi, mList), gagamma) * sineSquaredTheta * formFactorSquared * S2(mList) \
                * FW(Ea(theta_z, phi, mList), E1, E2) * (1 / np.dot(vecG(mList), vecG(mList)))
        
        return prefactor * rate




    def BraggPrimakoffAvgPhi(self, theta_z, E1=2.0, E2=2.5, gagamma=1e-10):
        prefactor = (gagamma / 1e6)**2 * kHBarC**2 * (vCrys / va**2) / 4  # 1e6 to convert to keV^-1
        mList = GetReciprocalLattice()
        def Rate(phi):
            rate = 0.0
            for m in mList:
                sineThetaBy2 = np.dot(vecU(theta_z, phi), vecG(m)) / np.dot(vecG(m),vecG(m))
                sineSquaredTheta = 4 * sineThetaBy2**2 * (1 - sineThetaBy2**2)
                formFactorSquared = FA(sqrt(np.dot(vecG(m), vecG(m))), Ea(theta_z, phi, m))
                rate += np.sum(heaviside(Ea(theta_z, phi, m), 0.0) * SolarFlux(Ea(theta_z, phi, m), gagamma) * sineSquaredTheta * formFactorSquared * S2Expanded(m) \
                        * FW(Ea(theta_z, phi, m), E1, E2) * (1 / np.dot(vecG(m), vecG(m))))
            return rate
        
        return prefactor * quad(Rate, 0.0, 2*pi)[0]



    # Bragg-Primakoff event rate
    def BraggPrimakoffScatteringPlane(self, theta, E1=2.0, E2=2.5, gagamma=1e-10):
        rate = 0.0
        prefactor = (gagamma / 1e6)**2 * kHBarC**2 * (vCrys / va**2) / 4  # 1e6 to convert to keV^-1
        for mList in GetReciprocalLattice():
            if (np.all(np.array(mList) % 2 == 1) or (np.all(np.array(mList) % 2 == 0) and np.sum(np.array(mList)) % 4 == 0)):
                sineSquaredTheta = sin(theta)**3
                formFactorSquared = FA(sqrt(np.dot(vecG(mList), vecG(mList))), Ea2(theta, mList))
                rate += heaviside(Ea2(theta, mList), 0.0) * SolarFlux(Ea2(theta, mList), gagamma) * sineSquaredTheta * formFactorSquared * S2(mList) \
                    * FW(Ea2(theta, mList), E1, E2) * (1 / np.dot(vecG(mList), vecG(mList)))
        
        return prefactor * rate


    def AtomicPrimakoffDifferentialRate(self, Ea, theta, gagamma=1e-10):
        return SolarFlux(Ea, gagamma) * zGe**2 * kALPHA * (gagamma / 1e6)**2 * Ea**4 * sin(theta)**3 / (2 * Ea**2 * (cos(theta)-1)**2) / 4

    def AtomicPrimakoffRate(self, theta, gagamma=1e-10):
        return quad(AtomicPrimakoffDifferentialRate, 0.001, 20.0, args=(theta,gagamma,))[0]




"""
# Plot the event rate as a function of theta_z for a given phi.
thetaList = np.linspace(0.0, 2*pi, 100)

# Compare with the atomic primakoff rate
atomic_primakoff_rates = np.array([AtomicPrimakoffRate(theta) for theta in thetaList])
bragg_primakoff_rates_avg_phi = np.array([BraggPrimakoffAvgPhi(theta) for theta in thetaList])


# Plot the phi-averaged rates in the 2.0 to 2.5 keV window
plt.plot(thetaList, bragg_primakoff_rates_avg_phi)
plt.xlabel(r"$\theta_z$")
plt.ylabel("Rate (a.u.)")
plt.show()
plt.close()
"""


