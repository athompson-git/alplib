# Bragg-Primakoff effect classes
from itertools import product

from .constants import *
from .fmath import *
from .det_xs import iprimakoff_sigma
from .materials import Material
from .crystal import Crystal
from .borrmann import *

# Global Constants in keV angstroms
M_E_KeV = 1e3 * M_E

class BraggPrimakoff:
    def __init__(self, crys: Crystal, ma=0.0001, nsamples=100, energy_res=0.73):
        # Lattice params
        self.crys = crys
        self.a = crys.a
        self.z = crys.z
        self.r0 = crys.r0
        self.va = crys.cell_volume
        self.ntargets = crys.ntargets
        self.volume = crys.volume*1e24  # convert cm3 to A3
        self.l_crystal = power(self.volume, 1/3)
        self.fwhm = energy_res

        self.ma = ma  # in keV

        self.borrmann = Borrmann(Material(crys.mat_name))
        #self.absorption_sum = AbsorptionSum(Material(crys.mat_name), n_atoms_side=4, physical_length=power(self.volume,1/3))
        self.absorption_sum = AbsorptionSumTable()

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
        return self.crys.SF2(mList[0], mList[1], mList[2])

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

    def SolarFlux(self, Ea, gagamma):
        # Solar ALP flux in keV^-1 cm^-2 s^-1
        return (gagamma * 1e8)**2 * (5.95e14 / 1.103) * (Ea / 1.103)**3 / (exp(Ea / 1.103) - 1)

    def SolarFluxMassiveALP(self, Ea, gagamma):
        # Solar ALP flux in keV^-1 cm^-2 s^-1
        # Ea in keV, gagamma in GeV^-1
        if Ea <= self.ma:
            return 0.0
        # Primakoff + Photon Coalescence (0006327v3)
        primakoff_flux = 4.20e10*(gagamma*1e10)**2 * Ea*(Ea**2 - self.ma**2)*(1+0.02 * self.ma)/(np.exp(Ea/1.1) - 0.7)
        coalescence_flux = 1.68e9*(gagamma*1e10)**2 * self.ma**4 * sqrt(Ea**2 - self.ma**2)*(1+0.0006*Ea**3 + 10/(0.2 + Ea**2))*np.exp(-Ea)
        return primakoff_flux + coalescence_flux

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
    def PrimakoffRate(self, theta_z, phi, E1=2.0, E2=2.5, gagamma=1e-10, use_borrmann=False, days_exposure=1.0, fixed_hkl=None):
        rate = 0.0
        #prefactor = (gagamma / 1e6)**2 * HBARC_KEV_ANG**2 * (self.volume / self.va**2) / 4  # 1e6 to convert to keV^-1
        prefactor = pi*(S_PER_DAY*days_exposure) * (gagamma / 1e6)**2 * HBARC_KEV_ANG**3 \
            * (self.volume / self.va**2) * 1e-16 # 1e6 to convert to keV^-1
        for hkl in self.GetReciprocalLattice():
            if fixed_hkl is not None:
                if hkl[0] != fixed_hkl[0] or hkl[1] != fixed_hkl[1] or hkl[2] != fixed_hkl[2]:
                    continue
            if np.dot(self.vecU(theta_z, phi), self.vecG(hkl)) == 0.0:
                continue
            ea = abs(self.Ea(theta_z, phi, hkl))
            sineThetaBy2 = np.dot(self.vecU(theta_z, phi), self.vecG(hkl)) / sqrt(np.dot(self.vecG(hkl),self.vecG(hkl)))
            sineSquaredTheta = 4 * sineThetaBy2**2 * (1 - sineThetaBy2**2)
            formFactorSquared = self.FA(sqrt(np.dot(self.vecG(hkl), self.vecG(hkl))), ea)
            atten_factor = 1.0
            if use_borrmann:
                atten_factor = self.absorption_sum.read_atten_factor_table(theta_z, phi, hkl=hkl)

            rate += np.sum(heaviside(ea, 0.0) \
                    * self.SolarFluxMassiveALP(ea, gagamma) * sineSquaredTheta \
                    * formFactorSquared * self.S2(hkl) \
                    * self.FW(ea, E1, E2) * (1 / np.dot(self.vecG(hkl), self.vecG(hkl)))) \
                    * atten_factor
        
        return prefactor * rate

    def BraggPrimakoffAvgPhi(self, theta_z, E1=2.0, E2=2.5, gagamma=1e-10, days_exposure=1.0,
                                use_borrmann=False, fixed_hkl=None):
        # Bragg-Primakoff event rate after averaging over polar angle
        # theta_z: in rad
        # E1, E2 in keV
        # gagamma in GeV^-1
        # days_exposure in days
        prefactor = pi *(S_PER_DAY*days_exposure) * (gagamma / 1e6)**2 * HBARC_KEV_ANG**3 \
            * (self.volume / self.va**2) * 1e-16 # 1e6 to convert to keV^-1
        def Rate(phi):
            rate = 0.0
            for hkl in self.GetReciprocalLattice():
                if fixed_hkl is not None:
                    if hkl[0] != fixed_hkl[0] or hkl[1] != fixed_hkl[1] or hkl[2] != fixed_hkl[2]:
                        continue
                if np.dot(self.vecU(theta_z, phi), self.vecG(hkl)) == 0.0:
                    continue
                ea = abs(self.Ea(theta_z, phi, hkl))
                sineThetaBy2 = np.dot(self.vecU(theta_z, phi), self.vecG(hkl)) / sqrt(np.dot(self.vecG(hkl),self.vecG(hkl)))
                sineSquaredTheta = 4 * sineThetaBy2**2 * (1 - sineThetaBy2**2)
                formFactorSquared = self.FA(sqrt(np.dot(self.vecG(hkl), self.vecG(hkl))), ea)
                atten_factor = 1.0
                if use_borrmann:
                    atten_factor = self.absorption_sum.read_atten_factor_table(theta_z, phi, hkl=hkl)
                rate += np.sum(heaviside(ea, 0.0) \
                    * self.SolarFluxMassiveALP(ea, gagamma) * sineSquaredTheta \
                    * formFactorSquared * self.S2(hkl) \
                    * self.FW(ea, E1, E2) * (1 / np.dot(self.vecG(hkl), self.vecG(hkl)))) \
                    * atten_factor

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
