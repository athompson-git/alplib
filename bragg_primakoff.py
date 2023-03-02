# Bragg-Primakoff effect classes
from itertools import product

from .constants import *
from .fmath import *
from .det_xs import iprimakoff_sigma
from .materials import Material
from .crystal import Crystal
from .borrmann import Borrmann, AbsorptionSum

# Global Constants in keV angstroms
M_E_KeV = 1e3 * M_E
HBARC_KEV_ANG = 1.97

# Crystal constants
zGe = 32
r0Ge = 0.53
vCrys = 1e12  # 1 cubic micron in cubic angstroms

class BraggPrimakoff:
    def __init__(self, crys: Crystal, nsamples=100, energy_res=0.73):
        # Lattice params
        self.a = crys.a
        self.z = crys.z
        self.r0 = crys.r0
        self.va = crys.cell_volume
        self.ntargets = crys.ntargets
        self.volume = crys.volume*1e24  # convert cm3 to A3
        self.l_crystal = power(self.volume, 1/3)
        self.fwhm = energy_res

        self.borrmann = Borrmann(Material(crys.mat_name))
        self.absorption_sum = AbsorptionSum(Material(crys.mat_name), n_atoms_side=4, physical_length=power(self.volume,1/3))

        # Primitive basis vectors
        self.a0 = crys.a0 #np.array([0,0,0])
        self.a1 = crys.a1 #(self.a/4) * np.array([1,1,1])

        # Bravais lattice
        self.b1 = crys.b1
        self.b2 = crys.b2
        self.b3 = crys.b3

        # Phi list
        self.nsamples = nsamples
        self.phi_list = np.random.uniform(0.0, 2*pi, self.nsamples)
        self.e_list = np.linspace(1.0, 30.0, self.nsamples)

    # Reciprocal Lattice
    def vecG(self, mList):
        return mList[0] * self.b1 + mList[1] * self.b2 + mList[2] * self.b3

    # Atomic form factor squared
    def FA(self, q2, k):
        # energy in keV
        # q2 in A^-2
        return np.sum(power(self.z * sqrt(4*pi*ALPHA) * (k/HBARC_KEV_ANG)**2 / (q2 + power(self.r0, -2)), 2))

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
        return 0.5 * ( erf((Ea - E1)/(sqrt(2)*self.fwhm/2.35)) - erf((Ea - E2)/(sqrt(2)*self.fwhm/2.35)) )

    # Incident photon vector
    def vecU(self, theta_z, phi):
        return np.array([-sin(theta_z)*cos(phi), -sin(theta_z)*sin(phi), -cos(theta_z)])

    # Bragg condition
    def Ea(self, theta_z, phi, mList):
        return HBARC_KEV_ANG * abs(np.dot(self.vecG(mList), self.vecG(mList)) \
            / (2 * np.dot(self.vecU(theta_z, phi), self.vecG(mList))))

    # Solar ALP flux in keV^-1 cm^-2 s^-1
    def SolarFlux(self, Ea, gagamma):
        return (gagamma * 1e8)**2 * (5.95e14 / 1.103) * (Ea / 1.103)**3 / (exp(Ea / 1.103) - 1)


    # Getter for the list of reciprocal vectors
    def GetReciprocalLattice(self, nmax=5):
        g = []
        for mList in product(np.arange(0,nmax),repeat=3):
            if (np.sum(mList) == 0):
                continue
            if (np.all(np.array(mList) % 2 == 1) or (np.all(np.array(mList) % 2 == 0) and np.sum(np.array(mList)) % 4 == 0)):
                g.append(mList)
        return np.array(g)

    # Bragg-Primakoff event rate
    def PrimakoffRate(self, theta_z, phi, E1=2.0, E2=2.5, gagamma=1e-10, use_att=False, use_borrmann=False, days_exposure=1.0):
        rate = 0.0
        #prefactor = (gagamma / 1e6)**2 * HBARC_KEV_ANG**2 * (self.volume / self.va**2) / 4  # 1e6 to convert to keV^-1
        prefactor = pi*(S_PER_DAY*days_exposure) * (gagamma / 1e6)**2 * HBARC_KEV_ANG**3 \
            * (self.volume / self.va**2) * 1e-16 # 1e6 to convert to keV^-1
        for mList in self.GetReciprocalLattice():
            if np.dot(self.vecU(theta_z, phi), self.vecG(mList)) == 0.0:
                continue
            ea = abs(self.Ea(theta_z, phi, mList))
            sineThetaBy2 = np.dot(self.vecU(theta_z, phi), self.vecG(mList)) / sqrt(np.dot(self.vecG(mList),self.vecG(mList)))
            sineSquaredTheta = 4 * sineThetaBy2**2 * (1 - sineThetaBy2**2)
            formFactorSquared = self.FA(sqrt(np.dot(self.vecG(mList), self.vecG(mList))), ea)
            atten_factor = 1
            if use_borrmann:
                l_borrmann = 1e8*self.borrmann.anomalous_depth(ea, mList[0], mList[1], mList[2])  # 1e8: cm to A conversion
                atten_factor = self.absorption_sum.get_atten_factor(mfp=l_borrmann, hkl=mList, kVec=ea*self.vecU(theta_z, phi))
            elif use_att:
                l_att = 1e8/(self.borrmann.n * self.borrmann.abs_xs.sigma_cm2(1e-3*ea))
                atten_factor = self.absorption_sum.get_atten_factor(mfp=l_att, hkl=mList, kVec=ea*self.vecU(theta_z, phi))
            rate += np.sum(heaviside(ea, 0.0) \
                    * self.SolarFlux(ea, gagamma) * sineSquaredTheta \
                    * formFactorSquared * self.S2Expanded(mList) \
                    * self.FW(ea, E1, E2) * (1 / np.dot(self.vecG(mList), self.vecG(mList)))) \
                    * atten_factor
        
        return prefactor * rate

    def BraggPrimakoffAvgPhi(self, theta_z, E1=2.0, E2=2.5, gagamma=1e-10, days_exposure=1.0,
                                use_borrmann=False, use_att=False):
        # Bragg-Primakoff event rate after averaging over polar angle
        # theta_z: in rad
        # E1, E2 in keV
        # gagamma in GeV^-1
        # days_exposure in days
        prefactor = pi *(S_PER_DAY*days_exposure) * (gagamma / 1e6)**2 * HBARC_KEV_ANG**3 \
            * (self.volume / self.va**2) * 1e-16 # 1e6 to convert to keV^-1
        mList = self.GetReciprocalLattice()
        def Rate(phi):
            rate = 0.0
            for m in mList:
                if np.dot(self.vecU(theta_z, phi), self.vecG(m)) == 0.0:
                    continue
                ea = abs(self.Ea(theta_z, phi, m))
                sineThetaBy2 = np.dot(self.vecU(theta_z, phi), self.vecG(m)) / sqrt(np.dot(self.vecG(m),self.vecG(m)))
                sineSquaredTheta = 4 * sineThetaBy2**2 * (1 - sineThetaBy2**2)
                formFactorSquared = self.FA(sqrt(np.dot(self.vecG(m), self.vecG(m))), ea)
                l_factor = power(self.volume, 1/3)
                if use_borrmann:
                    l_factor = 1e8*self.borrmann.anomalous_depth(ea, m[0], m[1], m[2])  # 1e8: cm to A conversion
                elif use_att:
                    l_factor = 1e8/(self.borrmann.n * self.borrmann.abs_xs.sigma_cm2(1e-3*ea))
                rate += np.sum(heaviside(ea, 0.0) \
                    * self.SolarFlux(ea, gagamma) * sineSquaredTheta \
                    * formFactorSquared * self.S2Expanded(m) \
                    * self.FW(ea, E1, E2) * (1 / np.dot(self.vecG(m), self.vecG(m)))) \
                    * l_factor / power(self.volume, 1/3)

            return rate
        
        rates = np.array([Rate(this_phi) for this_phi in self.phi_list])
        return prefactor * 2*pi*np.sum(rates)/self.nsamples  # fast MC-based integration

    def AtomicPrimakoffRate(self, E1=2.0, E2=2.5, gagamma=1e-10, ma=1e-6, days_exposure=1.0):
        # Solar ALP scattering rate ignoring crystal structure (isolated atomic scattering)
        # energy range in keV, gagamma in GeV^-1, ma in MeV
        def AtomicPrimakoffDifferentialRate(Ee, Ea):
            return self.ntargets * self.SolarFlux(Ea, gagamma) * (HBARC**2) \
                * iprimakoff_sigma(Ea*1e-3, gagamma*1e-3, ma, self.z[0], self.r0[0]) \
                    * exp(-power((Ee - Ea)/(sqrt(2)*self.fwhm/2.35), 2)) / sqrt(2)*self.fwhm/2.35
        
        ea_list = np.linspace(0.1, 30.0, 10*self.nsamples)
        rates = np.array([quad(AtomicPrimakoffDifferentialRate, E1, E2, args=(Ea,))[0] for Ea in ea_list])
        mc_volume = (ea_list[-1] - ea_list[0])/self.nsamples/10
        return (S_PER_DAY*days_exposure) * mc_volume*np.sum(rates)





def l_laue(energy, mat: Material, hkl=(2,2,0,)):
    pass




def l_bragg(energy, mat: Material):
    pass




def l_bloch(energy, mat: Material, hkl=(2,2,0,)):
    pass




def l_att(energy, mat: Material):
    pass
