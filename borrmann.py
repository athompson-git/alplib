# Compute Borrmann effect parameters for Crystallographic Scattering

import pkg_resources

from .constants import *
from .fmath import *
from .photon_xs import AbsCrossSection
from .crystal import *

import multiprocessing
from multiprocessing import Pool

from scipy.interpolate import RegularGridInterpolator

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
    def __init__(self, material: Material, verbose=False, cryogenic=True):
        # Set coefficients from Peng et al. Assumes Cryogenic temps
        self.a_coeffs = []
        self.b_coeffs = []

        if material.mat_name == "Ge":
            # At cryogenic temps
            self.a_coeffs = [-0.0099, 0.0514, 0.0351, 0.0238, 0.0044]
            self.b_coeffs = [0.0267, 0.1536, 0.4845, 1.3795, 5.4966]
        elif material.mat_name == "NaI":
            if cryogenic:
                self.a_coeffs = [-0.0068, 0.0569, 0.0251, -0.0211, 0.0055]
                self.b_coeffs = [0.2277, 1.7152, 9.4654, 13.1051, 32.9902]
            else:
                # At room temp
                self.a_coeffs = [-0.0015, -0.0145, 0.0953, 0.0113, 0.0041]
                self.b_coeffs = [0.2083, 0.9749, 4.3959, 8.7251, 36.0870]
        elif material.mat_name == "Si":
            # At cryogenic temps
            self.a_coeffs = [-0.0028, 0.0127, 0.0108, 0.0058, 0.0024]
            self.b_coeffs = [0.0382, 0.2025, 0.5845, 1.7728, 10.6593]
        elif material.mat_name == "CsI":
            # At room temps
            self.a_coeffs = [-0.0314, -0.0827, -0.1396, 2.0856, 0.0988]
            self.b_coeffs = [0.4061, 1.6180, 2.5843, 10.5874, 35.9707]
        else:
            # Assume Ge
            self.a_coeffs = [-0.0099, 0.0514, 0.0351, 0.0238, 0.0044]
            self.b_coeffs = [0.0267, 0.1536, 0.4845, 1.3795, 5.4966]

        # Constants and cross sections
        self.n = material.ndensity # cm^-3
        self.abs_xs = AbsCrossSection(material)
        self.crystal = get_crystal(material.mat_name, volume=1000)
        self.verbose = verbose

    def debye_waller(self):
        return 1.0
    
    def imff(self, s):
        return np.sum([self.a_coeffs[i] * np.exp(-self.b_coeffs[i]*s**2) for i in range(5)])

    def sf_ratio(self, h, k, l):  # structure function ratio
        return sqrt(self.crystal.SF2(h, k, l)/self.crystal.SF2(0, 0, 0))
    
    def borrmann_factor(self, h, k, l):
        gvec = self.crystal.G(h, k, l)
        sinThetaByLambda = sqrt(np.dot(gvec, gvec))/4/pi
        return self.sf_ratio(h,k,l) * self.imff(sinThetaByLambda) / self.imff(0.0)
    
    def imff_ratio(self, sinThetaByLambda):
        return self.imff(sinThetaByLambda) / self.imff(0.0)

    def anomalous_abs(self, energy, h, k, l):
        mu = self.n * self.abs_xs.sigma_cm2(1e-3*energy)
        return mu * (1 - self.borrmann_factor(h, k, l))

    def anomalous_depth(self, energy, h, k, l):
        return 1/self.anomalous_abs(energy, h, k, l)




# Old Borrmann calculation based on Batterman + Wagenfield calcs
class BattermanBorrmannFactor:
    def __init__(self, material: Material, verbose=False):
        # Batterman files
        ge_l1_path = pkg_resources.resource_filename(__name__, "data/borrmann/Ge_L1_f.txt")
        ge_l23_path = pkg_resources.resource_filename(__name__, "data/borrmann/Ge_L23_f.txt")
        ge_m23_path = pkg_resources.resource_filename(__name__, "data/borrmann/Ge_M23_f.txt")
        self.ge_l1 = np.genfromtxt(ge_l1_path, delimiter=",")
        self.ge_l23 = np.genfromtxt(ge_l23_path, delimiter=",")
        self.ge_m23 = np.genfromtxt(ge_m23_path, delimiter=",")

        # Wagenfield Files
        imff_path = pkg_resources.resource_filename(__name__, "data/borrmann/imff_{}_full.txt".format(material.mat_name))
        imff_quad_path = pkg_resources.resource_filename(__name__, "data/borrmann/imff_{}_quad.txt".format(material.mat_name))
        self.full_dat = np.genfromtxt(imff_path)
        self.quad_dat = np.genfromtxt(imff_quad_path)

        # Constants and cross sections
        self.n = material.ndensity # cm^-3
        self.abs_xs = AbsCrossSection(material)
        self.crystal = get_crystal(material.mat_name, volume=1000)
        self.verbose = verbose
    
    def imff(self, k):
        # return imff given k in keV
        return np.interp(k * 1e-6, self.full_dat[:,0], self.full_dat[:,1])
    
    def imff_quad(self, k):
        # return Quadrupole moment imff given k in keV
        return np.interp(k * 1e-6, self.quad_dat[:,0], self.quad_dat[:,1])

    def debye_waller(self):
        return 1.0

    def sf_ratio(self, h, k, l):  # structure function ratio
        return sqrt(self.crystal.SF2(h, k, l)/self.crystal.SF2(0, 0, 0))
    
    def f_L1(self, sinThetaByLambda):
        return np.interp(sinThetaByLambda, self.ge_l1[:,0], self.ge_l1[:,1])

    def f_L23(self, sinThetaByLambda):
        return np.interp(sinThetaByLambda, self.ge_l23[:,0], self.ge_l23[:,1])

    def f_M23(self, sinThetaByLambda):
        return np.interp(sinThetaByLambda, self.ge_m23[:,0], self.ge_m23[:,1])
    
    def epsilon_sinThetaByLambda(self, sinThetaByLambda):
        # Batterman's
        l1 =  ZjEtaj_L1*self.f_L1(sinThetaByLambda)
        l23 = ZjEtaj_L23*self.f_L23(sinThetaByLambda)
        m23 = ZjEtaj_M23*self.f_M23(sinThetaByLambda)
        denominator = ZjEtaj_L1 + ZjEtaj_L23 + ZjEtaj_M23
        epsilon = (l1 + l23 + m23)/denominator
        return epsilon
    
    def borrmann_factor(self, energy, h, k, l):
        gvec = self.crystal.G(h, k, l)
        sinThetaByLambda = sqrt(np.dot(gvec, gvec))/4/pi
        return self.sf_ratio(h,k,l) * (1 - 2 * (sinThetaByLambda * 2*pi*HBARC_KEV_ANG/k)**2 \
            * (self.imff_quad(energy)/self.imff(energy))*0.4)

    def anomalous_abs(self, energy, h, k, l):
        mu = self.n * self.abs_xs.sigma_cm2(1e-3*energy)
        return mu * (1 - self.borrmann_factor(energy, h, k, l))

    def anomalous_depth(self, energy, h, k, l):
        return 1/self.anomalous_abs(energy, h, k, l)




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
    
    def get_atten_factor(self, mfp=1e-3, hkl=[2,2,0], kVec=[5.0,0.0,0.0], n_workers=8):
        # scale mfp
        toy_mfp = mfp * (self.cube_length / self.physical_length)

        Gvec = HBARC_KEV_ANG * (hkl[0]*self.crystal.b1 + hkl[1]*self.crystal.b2 + hkl[2]*self.crystal.b3)  # angstroms^-1 to keV
        kprime = kVec - Gvec
        kprime_hat = kprime / np.sqrt(np.dot(kprime, kprime))

        p = Pool(n_workers)

        chunk_size = int(self.N**2 / n_workers)
        start_indices = np.arange(0,self.N**2+chunk_size,chunk_size)
        results = []

        for i in range(start_indices.shape[0]-1):
            results.append(p.apply_async(self.parallel_sum, (start_indices[i], start_indices[i+1], kprime_hat, toy_mfp)))


        totals = [res.get() for res in results]
        p.close()
        p.join()
        return np.sum(totals)/self.N**2
    



class AbsorptionSumTable:
    def __init__(self, mat_name="Ge"):
        self.path_prefix = "data/borrmann/abssum_"
        self.file_extension = "_withBorrmann.txt"
        fpath_111 = pkg_resources.resource_filename(__name__, self.path_prefix + mat_name + "_111" + self.file_extension)
        fpath_220 = pkg_resources.resource_filename(__name__, self.path_prefix + mat_name + "_220" + self.file_extension)
        fpath_202 = pkg_resources.resource_filename(__name__, self.path_prefix + mat_name + "_202" + self.file_extension)
        fpath_022 = pkg_resources.resource_filename(__name__, self.path_prefix + mat_name + "_022" + self.file_extension)
        fpath_113 = pkg_resources.resource_filename(__name__, self.path_prefix + mat_name + "_113" + self.file_extension)
        fpath_131 = pkg_resources.resource_filename(__name__, self.path_prefix + mat_name + "_131" + self.file_extension)
        fpath_311 = pkg_resources.resource_filename(__name__, self.path_prefix + mat_name + "_311" + self.file_extension)
        fpath_133 = pkg_resources.resource_filename(__name__, self.path_prefix + mat_name + "_133" + self.file_extension)
        fpath_331 = pkg_resources.resource_filename(__name__, self.path_prefix + mat_name + "_331" + self.file_extension)
        fpath_333 = pkg_resources.resource_filename(__name__, self.path_prefix + mat_name + "_333" + self.file_extension)
        fpath_400 = pkg_resources.resource_filename(__name__, self.path_prefix + mat_name + "_400" + self.file_extension)

        self.data_111 = np.genfromtxt(fpath_111)[:,2].reshape(300,300)
        self.data_220 = np.genfromtxt(fpath_220)[:,2].reshape(300,300)
        self.data_202 = np.genfromtxt(fpath_202)[:,2].reshape(300,300)
        self.data_022 = np.genfromtxt(fpath_022)[:,2].reshape(300,300)
        self.data_113 = np.genfromtxt(fpath_113)[:,2].reshape(300,300)
        self.data_131 = np.genfromtxt(fpath_131)[:,2].reshape(300,300)
        self.data_311 = np.genfromtxt(fpath_311)[:,2].reshape(300,300)
        self.data_331 = np.genfromtxt(fpath_331)[:,2].reshape(300,300)
        self.data_133 = np.genfromtxt(fpath_133)[:,2].reshape(300,300)
        self.data_333 = np.genfromtxt(fpath_333)[:,2].reshape(300,300)
        self.data_400 = np.genfromtxt(fpath_400)[:,2].reshape(300,300)

        self.theta_arr = np.linspace(1/300, pi, 300)
        self.phi_arr = np.linspace(1/300, 2*pi, 300)

        self.interp_111 = RegularGridInterpolator((self.theta_arr, self.phi_arr), self.data_111, bounds_error=False)
        self.interp_220 = RegularGridInterpolator((self.theta_arr, self.phi_arr), self.data_220, bounds_error=False)
        self.interp_202 = RegularGridInterpolator((self.theta_arr, self.phi_arr), self.data_202, bounds_error=False)
        self.interp_022 = RegularGridInterpolator((self.theta_arr, self.phi_arr), self.data_022, bounds_error=False)
        self.interp_113 = RegularGridInterpolator((self.theta_arr, self.phi_arr), self.data_113, bounds_error=False)
        self.interp_131 = RegularGridInterpolator((self.theta_arr, self.phi_arr), self.data_131, bounds_error=False)
        self.interp_311 = RegularGridInterpolator((self.theta_arr, self.phi_arr), self.data_311, bounds_error=False)
        self.interp_331 = RegularGridInterpolator((self.theta_arr, self.phi_arr), self.data_331, bounds_error=False)
        self.interp_133 = RegularGridInterpolator((self.theta_arr, self.phi_arr), self.data_133, bounds_error=False)
        self.interp_333 = RegularGridInterpolator((self.theta_arr, self.phi_arr), self.data_333, bounds_error=False)
        self.interp_400 = RegularGridInterpolator((self.theta_arr, self.phi_arr), self.data_400, bounds_error=False)

    def read_atten_factor_table(self, theta, phi, hkl=[1,1,1]):
        hkl_str = ''.join(map(str, hkl))
        if hkl_str == '111':
            return self.interp_111([theta, phi])[0]
        elif hkl_str == '220':
            return self.interp_220([theta, phi])[0]
        elif hkl_str == '202':
            return self.interp_202([theta, phi])[0]
        elif hkl_str == '022':
            return self.interp_022([theta, phi])[0]
        elif hkl_str == '113':
            return self.interp_113([theta, phi])[0]
        elif hkl_str == '131':
            return self.interp_131([theta, phi])[0]
        elif hkl_str == '311':
            return self.interp_311([theta, phi])[0]
        elif hkl_str == '331':
            return self.interp_331([theta, phi])[0]
        elif hkl_str == '133':
            return self.interp_133([theta, phi])[0]
        elif hkl_str == '333':
            return self.interp_333([theta, phi])[0]
        elif hkl_str == '400':
            return self.interp_400([theta, phi])[0]
        return 0.0