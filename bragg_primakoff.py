# Bragg-Primakoff effect classes
from itertools import product

from .constants import *
from .fmath import *
from .det_xs import iprimakoff_sigma
from .materials import Material
from .crystal import Crystal
from .borrmann import Borrmann

# Global Constants in keV angstroms
M_E_KeV = 1e3 * M_E
HBARC_KEV_ANG = 1.97

# Crystal constants
zGe = 32
r0Ge = 0.53
vCrys = 1e12  # 1 cubic micron in cubic angstroms

class BraggPrimakoff:
    def __init__(self, crys: Crystal):
        # Lattice params
        self.a = crys.a
        self.z = crys.z
        self.r0 = crys.r0
        self.va = crys.cell_volume
        self.ntargets = crys.ntargets
        self.volume = crys.volume*1e24  # convert cm3 to A3

        self.borrmann = Borrmann(Material(crys.mat_name))

        # Primitive basis vectors
        self.a0 = np.array([0,0,0])
        self.a1 = (self.a/4) * np.array([1,1,1])

        # Bravais lattice
        self.b1 = (2*pi/self.a) * np.array([-1, 1, 1])
        self.b2 = (2*pi/self.a) * np.array([1, -1, 1])
        self.b3 = (2*pi/self.a) * np.array([-1, 1, -1])

        # Phi list
        self.nsamples = 100
        self.phi_list = np.linspace(0.0, 2*pi, self.nsamples)
        self.e_list = np.linspace(1.0, 30.0, self.nsamples)

    # Reciprocal Lattice
    def vecG(self, mList):
        return mList[0] * self.b1 + mList[1] * self.b2 + mList[2] * self.b3

    # Atomic form factor squared
    def FA(self, q2, k):
        return (self.z * sqrt(4*pi*ALPHA) * k**2 / ( 1 / (self.r0**2) + q2))**2

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
        return HBARC_KEV_ANG * np.dot(self.vecG(mList), self.vecG(mList)) / (2 * np.dot(self.vecU(theta_z, phi), self.vecG(mList)))

    def Ea2(self, theta, mList):
        return HBARC_KEV_ANG * np.sqrt(np.dot(self.vecG(mList), self.vecG(mList))) / (2 * sin(theta/2))

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

    # Bragg-Primakoff event rate
    def BraggPrimakoff(self, theta_z, phi, E1=2.0, E2=2.5, gagamma=1e-10):
        rate = 0.0
        prefactor = (gagamma / 1e6)**2 * HBARC_KEV_ANG**2 * (self.volume / self.va**2) / 4  # 1e6 to convert to keV^-1
        for mList in self.GetReciprocalLattice():
            sineThetaBy2 = np.dot(self.vecU(theta_z, phi), self.vecG(mList)) / np.dot(self.vecG(mList),self.vecG(mList))
            sineSquaredTheta = 4 * sineThetaBy2**2 * (1 - sineThetaBy2**2)
            formFactorSquared = self.FA(sqrt(np.dot(self.vecG(mList), self.vecG(mList))), self.Ea(theta_z, phi, mList))
            rate += heaviside(self.Ea(theta_z, phi, mList), 0.0) \
                * self.SolarFlux(self.Ea(theta_z, phi, mList), gagamma) * sineSquaredTheta \
                * formFactorSquared * self.S2(mList) * self.FW(self.Ea(theta_z, phi, mList), E1, E2) \
                * (1 / np.dot(self.vecG(mList), self.vecG(mList)))
        
        return prefactor * rate

    def BraggPrimakoffAvgPhi(self, theta_z, E1=2.0, E2=2.5, gagamma=1e-10, use_borrmann=False, use_att=False):
        # Bragg-Primakoff event rate after averaging over polar angle
        prefactor = (gagamma / 1e6)**2 * HBARC_KEV_ANG**2 * (self.volume / self.va**2) / 4  # 1e6 to convert to keV^-1
        mList = self.GetReciprocalLattice()
        def Rate(phi):
            rate = 0.0
            for m in mList:
                ea = abs(self.Ea(theta_z, phi, m))
                sineThetaBy2 = np.dot(self.vecU(theta_z, phi), self.vecG(m)) / np.dot(self.vecG(m),self.vecG(m))
                sineSquaredTheta = 4 * sineThetaBy2**2 * (1 - sineThetaBy2**2)
                formFactorSquared = self.FA(sqrt(np.dot(self.vecG(m), self.vecG(m))), ea)
                l_factor = power(self.volume, 1/3)
                if use_borrmann:
                    l_factor = self.borrmann.anomalous_abs(abs(self.Ea(theta_z, phi, m)), m[0], m[1], m[2])
                elif use_att:
                    l_factor = self.borrmann.n * self.borrmann.abs_xs.sigma_cm2(1e-3*abs(self.Ea(theta_z, phi, m)))
                rate += np.sum(heaviside(ea, 0.0) \
                    * self.SolarFlux(ea, gagamma) * sineSquaredTheta \
                    * formFactorSquared * self.S2Expanded(m) \
                    * self.FW(ea, E1, E2) * (1 / np.dot(self.vecG(m), self.vecG(m)))) \
                    * l_factor / power(self.volume, 1/3)

            return rate
        
        rates = np.array([Rate(this_phi) for this_phi in self.phi_list])
        return prefactor * 2*pi*np.sum(rates)/self.nsamples  # fast MC-based integration

    def BraggPrimakoffScatteringPlane(self, theta, E1=2.0, E2=2.5, gagamma=1e-10):
        # Bragg-Primakoff event rate
        rate = 0.0
        prefactor = (gagamma / 1e6)**2 * HBARC_KEV_ANG**2 * (self.volume / self.va**2) / 4  # 1e6 to convert to keV^-1
        for mList in self.GetReciprocalLattice():
            if (np.all(np.array(mList) % 2 == 1) or (np.all(np.array(mList) % 2 == 0) and np.sum(np.array(mList)) % 4 == 0)):
                sineSquaredTheta = sin(theta)**3
                formFactorSquared = self.FA(sqrt(np.dot(self.vecG(mList), self.vecG(mList))), self.Ea2(theta, mList))
                rate += heaviside(self.Ea2(theta, mList), 0.0) * self.SolarFlux(self.Ea2(theta, mList), gagamma) \
                    * sineSquaredTheta * formFactorSquared * self.S2(mList) \
                    * self.FW(self.Ea2(theta, mList), E1, E2) * (1 / np.dot(self.vecG(mList), self.vecG(mList)))
        
        return prefactor * rate

    def LauePrimakoffAvgPhi(self, theta_z, E1=2.0, E2=2.5, gagamma=1e-10):
        # Laue-Primakoff event rate averaging over polar angle. Absorption effects included
        prefactor = (gagamma / 1e6)**2 * HBARC_KEV_ANG**2 * (self.volume / self.va**2) / 4  # 1e6 to convert to keV^-1
        mList = self.GetReciprocalLattice()
        def Rate(phi):
            rate = 0.0
            for m in mList:
                sineThetaBy2 = np.dot(self.vecU(theta_z, phi), self.vecG(m)) / np.dot(self.vecG(m),self.vecG(m))
                sineSquaredTheta = 4 * sineThetaBy2**2 * (1 - sineThetaBy2**2)
                formFactorSquared = self.FA(sqrt(np.dot(self.vecG(m), self.vecG(m))), self.Ea(theta_z, phi, m))
                rate += np.sum(heaviside(self.Ea(theta_z, phi, m), 0.0) \
                    * self.SolarFlux(self.Ea(theta_z, phi, m), gagamma) * sineSquaredTheta \
                    * formFactorSquared * self.S2Expanded(m) \
                    * self.FW(self.Ea(theta_z, phi, m), E1, E2) * (1 / np.dot(self.vecG(m), self.vecG(m))))
            return rate
        
        return prefactor * 2*pi*np.sum(Rate(self.phi_list))/self.nsamples  # fast MC-based integration

    def AtomicPrimakoffRate(self, E1=2.0, E2=2.5, gagamma=1e-10, ma=1e-4):
        # Solar ALP scattering rate ignoring crystal structure (isolated atomic scattering)
        def AtomicPrimakoffDifferentialRate(Ea, gagamma=1e-10, ma=1e-4):
            return self.ntargets * self.SolarFlux(Ea, gagamma) * (KEV_CM**2) \
                * iprimakoff_sigma(Ea/1e3, gagamma*1e-3, ma, self.z, self.r0)
        
        e_list = np.linspace(E1, E2, self.nsamples)
        mc_volume = (e_list[-1] - e_list[0])/self.nsamples
        return mc_volume*np.sum(AtomicPrimakoffDifferentialRate(e_list, gagamma, ma))





def l_laue(energy, mat: Material, hkl=(2,2,0,)):
    pass




def l_bragg(energy, mat: Material):
    pass




def l_bloch(energy, mat: Material, hkl=(2,2,0,)):
    pass




def l_att(energy, mat: Material):
    pass
