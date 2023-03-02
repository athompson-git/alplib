# Compute Borrmann effect parameters for Crystallographic Scattering

import pkg_resources

from .constants import *
from .fmath import *
from .photon_xs import AbsCrossSection
from .crystal import *
import multiprocessing
from multiprocessing import Pool


# read in the ff data




ZjEtaj_L1 = 0.407
ZjEtaj_L23 = 0.555
ZjEtaj_M23 = 0.090






"""
material: string specifying the material/type of crystal, e.g. "Ge", "CsI", etc.
cell_density: No. unit cells per volume cm^-3
abs_coeff: absorption coefficient in cm^-1
"""
class Borrmann:
    def __init__(self, material: Material, verbose=False):
        ge_l1_path = pkg_resources.resource_filename(__name__, "data/borrmann/Ge_L1_f.txt")
        ge_l23_path = pkg_resources.resource_filename(__name__, "data/borrmann/Ge_L23_f.txt")
        ge_m23_path = pkg_resources.resource_filename(__name__, "data/borrmann/Ge_M23_f.txt")
        self.ge_l1 = np.genfromtxt(ge_l1_path, delimiter=",")
        self.ge_l23 = np.genfromtxt(ge_l23_path, delimiter=",")
        self.ge_m23 = np.genfromtxt(ge_m23_path, delimiter=",")
        self.n = material.ndensity # cm^-3
        self.abs_xs = AbsCrossSection(material)
        self.crystal = get_crystal(material.mat_name, volume=1000)
        self.verbose = verbose

    def imff(self, h, k, l):
        energy = self.crystal.energy(h, k, l)
        sigma = self.abs_xs.sigma_cm2(energy*1e-3)
        imff = energy * sigma / (2 * HC * R_E)
        
        if self.verbose == True:
            print("    imff = ", imff)
        return imff

    def debye_waller(self):
        return 1.0

    def sf_ratio(self, h, k, l):  # structure function ratio
        return self.crystal.sfunc(h, k, l)/self.crystal.sfunc(0, 0, 0)

    def zj_etaj_sum(self, energy):
        lam = HC / energy
        mu = self.n * self.abs_xs.sigma_cm2(energy*1e-3)
        ZjEtaj = (M_E * mu) / (2 * HBARC * ALPHA * lam * (self.n/4))

        if self.verbose == True:
            print("    mu = ", mu)
            print("    sum(ZjEtaj) = ", ZjEtaj)
        return ZjEtaj
    
    def f_L1(self, sinThetaByLambda):
        return np.interp(sinThetaByLambda, self.ge_l1[:,0], self.ge_l1[:,1])

    def f_L23(self, sinThetaByLambda):
        return np.interp(sinThetaByLambda, self.ge_l23[:,0], self.ge_l23[:,1])

    def f_M23(self, sinThetaByLambda):
        return np.interp(sinThetaByLambda, self.ge_m23[:,0], self.ge_m23[:,1])

    def epsilon(self, energy, h, k, l):
        gvec = self.crystal.G(h, k, l)
        sinThetaByLambda = sqrt(np.dot(gvec, gvec))/4/pi
        debye = 0.981
        l1 =  ZjEtaj_L1*self.f_L1(sinThetaByLambda)
        l23 = ZjEtaj_L23*self.f_L23(sinThetaByLambda)
        m23 = ZjEtaj_M23*self.f_M23(sinThetaByLambda)
        denominator = ZjEtaj_L1 + ZjEtaj_L23 + ZjEtaj_M23
        epsilon = debye * (l1 + l23 + m23)/denominator
        return epsilon

    #def epsilon(self, energy, h, k, l):
    #    return self.sf_ratio(h, k, l) * self.debye_waller() * self.imff(h, k, l) / self.zj_etaj_sum(energy)

    def anomalous_abs(self, energy, h, k, l):
        mu = self.n * self.abs_xs.sigma_cm2(1e-3*energy)
        return mu * (1 - self.epsilon(energy, h, k, l))

    def anomalous_depth(self, energy, h, k, l):
        return 1/self.anomalous_abs(energy, h, k, l)


# Class for calculating the atomic form factors, shell-by-shell
class HydrogenicWaveFunction:
    def __init__(self, n=0, l=0, m=0):
        pass

    def radial_wf(self, r, n=0, l=0):
        pass

    def spherical_harmonic(self, theta, phi, l=0, m=0):
        pass

    def integral(self):
        pass




class AbsorptionSum:
    def __init__(self, material: Material, n_atoms_side=10, physical_length=5.0):
        """
        physical_length: physical length of crystal cube in cm
        """
        self.physical_length = physical_length
        self.crystal = get_crystal(material.mat_name, volume=physical_length**3)
        
        # Non-physical length of sample atomic positions
        self.cube_length = n_atoms_side * np.sqrt(np.dot(self.crystal.a3,self.crystal.a3))  # length of cube in angstroms, for comparison to MFP / Lz

        hs = np.arange(0,n_atoms_side,1)
        ks = np.arange(0,n_atoms_side,1)
        ls = np.arange(0,n_atoms_side,1)

        # generate list of position vectors
        self.position_vectors = []
        self.idx = []

        # initialize basis vectors for cube
        for h in hs:
            for k in ks:
                for l in ls:
                    self.position_vectors.append(self.crystal.a1 * h + self.crystal.a2 * k + self.crystal.a3 * l + self.crystal.alpha[0])  # alpha0 primitive (0,0,0)
                    self.position_vectors.append(self.crystal.a1 * h + self.crystal.a2 * k + self.crystal.a3 * l + self.crystal.alpha[1])  # alpha1 primitive

        # make map of (i,j) pairs
        self.N = len(self.position_vectors)
        indices = np.arange(0, self.N)
        PI, PJ = np.meshgrid(indices, indices)
        self.idx = PI.flatten()
        del PI
        self.jdx = PJ.flatten()
        del PJ
    
    def parallel_sum(self, start, end, kprime_hat, mfp=0.1):
        # coherent double sum over i,j pairs
        m2 = 0.0
        for k in range(start, end):
            if k >= self.N**2:
                continue
            
            ri_rj = self.position_vectors[self.idx[k]] - self.position_vectors[self.jdx[k]]
            
            if np.dot(ri_rj, ri_rj) == 0.0:
                continue
            
            dot_product = np.dot(kprime_hat, ri_rj)
            m2 += np.exp(-abs(dot_product) / (2*mfp))
        return m2
    
    def get_atten_factor(self, mfp=1e-3, hkl=[2,2,0], kVec=[5.0,0.0,0.0]):
        # scale mfp
        toy_mfp = mfp * (self.cube_length / self.physical_length)

        Gvec = HBARC_KEV_ANG * (hkl[0]*self.crystal.b1 + hkl[1]*self.crystal.b2 + hkl[2]*self.crystal.b3)  # angstroms^-1 to keV
        kprime = kVec - Gvec
        kprime_hat = kprime / np.sqrt(np.dot(kprime, kprime))

        p = Pool(8)

        chunk_size = int(self.N**2 / 10)
        start_indices = np.arange(0,self.N**2+chunk_size,chunk_size)
        results = []

        for i in range(start_indices.shape[0]-1):
            results.append(p.apply_async(self.parallel_sum, (start_indices[i], start_indices[i+1], kprime_hat, toy_mfp)))


        totals = [res.get() for res in results]
        return np.sum(totals)/self.N**2