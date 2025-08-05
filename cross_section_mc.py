"""
Cross section class and MC
"""

from alplib.constants import *
from alplib.fmath import *
from alplib.matrix_element import *

import vegas




class Vector3:
    def __init__(self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.vec = np.array([v1, v2, v3])
    
    def __str__(self):
        return "({0},{1},{2})".format(self.v1, self.v2, self.v3)
    
    def __add__(self, other):
        v1_new = self.v1 + other.v1
        v2_new = self.v2 + other.v2
        v3_new = self.v3 + other.v3
        return Vector3(v1_new, v2_new, v3_new)
    
    def __sub__(self, other):
        v1_new = self.v1 - other.v1
        v2_new = self.v2 - other.v2
        v3_new = self.v3 - other.v3
        return Vector3(v1_new, v2_new, v3_new)
    
    def __neg__(self):
        return Vector3(-self.v1, -self.v2, -self.v3)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):  # Scalar multiplication
            return Vector3(self.v1 * other, self.v2 * other, self.v3 * other)
        elif isinstance(other, Vector3):  # Dot product
            return np.dot(self.vec, other.vec)
        else:
            raise TypeError("Unsupported operand type for *: 'Vector' and '{}'".format(type(other)))
    
    def __rmul__(self, other):
        return self.__mul__(other)  # Reuse __mul__
    
    def unit_vec(self):
        v = self.mag()
        return Vector3(self.v1/v, self.v2/v, self.v3/v)
    
    def mag2(self):
        return np.dot(self.vec, self.vec)
    
    def mag(self):
        return np.sqrt(np.dot(self.vec, self.vec))

    def phi(self):
        return arctan2(self.v2, self.v1)
    
    def theta(self):
        return arccos(self.v3 / self.mag())
    
    def set_v3(self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.vec = np.array([v1, v2, v3])




class LorentzVector:
    def __init__(self, p0=0.0, p1=0.0, p2=0.0, p3=0.0):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.pmu = np.array([p0, p1, p2, p3])
        self.mt = np.array([1, -1, -1, -1])
        self.momentum3 = Vector3(self.p1, self.p2, self.p3)
    
    def __str__(self):
        return "({0},{1},{2},{3})".format(self.p0, self.p1, self.p2, self.p3)
    
    def __add__(self, other):
        p0_new = self.p0 + other.p0
        p1_new = self.p1 + other.p1
        p2_new = self.p2 + other.p2
        p3_new = self.p3 + other.p3
        return LorentzVector(p0_new, p1_new, p2_new, p3_new)
    
    def __sub__(self, other):
        p0_new = self.p0 - other.p0
        p1_new = self.p1 - other.p1
        p2_new = self.p2 - other.p2
        p3_new = self.p3 - other.p3
        return LorentzVector(p0_new, p1_new, p2_new, p3_new)
    
    def __mul__(self, other):
        return np.dot(self.pmu*other.pmu, self.mt)
    
    def __rmul__(self, other):
        return np.dot(self.pmu*other.pmu, self.mt)
    
    def mass2(self):
        return np.dot(self.pmu**2, self.mt)
    
    def mass(self):
        return np.sqrt(np.dot(self.pmu**2, self.mt))
    
    def energy(self):
        return self.p0
    
    def cosine(self):
        return self.p3 / self.momentum()
    
    def phi(self):
        return arctan2(self.p2, self.p1)
    
    def theta(self):
        return arccos(self.p3 / self.momentum())
    
    def momentum(self):
        return self.momentum3.mag()
    
    def set_p4(self, p0, p1, p2, p3):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.pmu = np.array([p0, p1, p2, p3])
        self.momentum3 = Vector3(self.p1, self.p2, self.p3)
    
    def get_3momentum(self):
        return Vector3(self.p1, self.p2, self.p3)
    
    def get_3velocity(self):
        return Vector3(self.p1/self.p0, self.p2/self.p0, self.p3/self.p0)




class Scatter2to2MC:
    def __init__(self, mtrx2: MatrixElement2, p1=LorentzVector(), p2=LorentzVector(), n_samples=1000):
        self.mtrx2 = mtrx2

        self.m1 = mtrx2.m1
        self.m2 = mtrx2.m2
        self.m3 = mtrx2.m3
        self.m4 = mtrx2.m4

        self.lv_p1 = p1
        self.lv_p2 = p2

        # TODO: add methods to change masses, couplings of matrix element

        self.n_samples = n_samples
        self.p3_cm_4vectors = []
        self.p3_lab_4vectors = []
        self.p3_cm_3vectors = []
        self.p3_lab_3vectors = []
        self.p4_lab_3vectors = []
        self.p4_lab_4vectors = []
        self.dsigma_dcos_cm_wgts = np.array([])
    
    def set_new_scattter(self, new_p1: LorentzVector, new_p2: LorentzVector):
        self.lv_p1 = new_p1
        self.lv_p2 = new_p2

        self.p3_cm_4vectors = []
        self.p3_lab_4vectors = []
        self.p3_cm_3vectors = []
        self.p3_lab_3vectors = []
        self.p4_lab_3vectors = []
        self.p4_lab_4vectors = []
        self.dsigma_dcos_cm_wgts = np.array([])

    def dsigma_dt(self, s, t):
        return self.mtrx2(s, t)/(16*np.pi*(s - (self.m1 + self.m2)**2)*(s - (self.m1 - self.m2)**2))

    def p1_cm(self, s):
        return np.sqrt((np.power(s - self.m1**2 - self.m2**2, 2) - np.power(2*self.m1*self.m2, 2))/(4*s))

    def p3_cm(self, s):
        return np.sqrt((np.power(s - self.m3**2 - self.m4**2, 2) - np.power(2*self.m3*self.m4, 2))/(4*s))

    def scatter_sim(self, log_sampling=False):
        # Takes in initial energy-momenta for p1, p2
        # Computes CM frame energies
        # Simulates events in CM frame

        # Draw random variates on the 2-sphere
        if log_sampling:
            # if using log-sampling, we (sample the forward cosine spectrum heavily (towards u = 0)
            phi_rnd = 2*pi*np.random.ranf(self.n_samples)
            logu = np.random.uniform(-12, 0, self.n_samples)
            theta_rnd = arccos(1 - 2*exp(logu))
        else:
            phi_rnd = 2*pi*np.random.ranf(self.n_samples)
            theta_rnd = arccos(1 - 2*np.random.ranf(self.n_samples))

        # Declare momenta and energy in the CM frame
        cm_p4 = self.lv_p1 + self.lv_p2
        e_in = cm_p4.energy()
        p_in = cm_p4.get_3momentum()
        v_in = Vector3(p_in.v1 / e_in, p_in.v2 / e_in, p_in.v3 / e_in)
        s = cm_p4.mass2()
        if s < (self.m3 + self.m4)**2:
            return

        p1_cm = self.p1_cm(s)
        p3_cm = self.p3_cm(s)
        e1_cm = np.sqrt(p1_cm**2 + self.m1**2)
        e3_cm = np.sqrt(p3_cm**2 + self.m3**2)

        t_rnd = self.m1**2 + self.m3**2 + 2*(p1_cm*p3_cm*cos(theta_rnd) - e1_cm*e3_cm)

        if log_sampling:
            # don't forget to change the monte carlo jacobian factor if we use log sampling
            mc_volume = exp(logu)*2*p1_cm*p3_cm/self.n_samples
        else:
            mc_volume = 4*p1_cm*p3_cm/self.n_samples
        self.dsigma_dcos_cm_wgts = mc_volume * self.dsigma_dt(s, t_rnd)

        # Boosts back to original frame
        self.p3_cm_4vectors = [LorentzVector(e3_cm,
                            p3_cm*cos(phi_rnd[i])*sin(theta_rnd[i]),
                            p3_cm*sin(phi_rnd[i])*sin(theta_rnd[i]),
                            p3_cm*cos(theta_rnd[i])) for i in range(self.n_samples)]
        self.p3_lab_4vectors = [lorentz_boost(p3, -v_in) for p3 in self.p3_cm_4vectors]
        self.p3_cm_3vectors = [p3_cm.get_3velocity() for p3_cm in self.p3_cm_4vectors]
        self.p3_lab_3vectors = [p3_lab.get_3velocity() for p3_lab in self.p3_lab_4vectors]

        self.p4_lab_4vectors = [self.lv_p1 + self.lv_p2 - p3_lab for p3_lab in self.p3_lab_4vectors]
        self.p4_lab_3vectors = [p4_lab.get_3velocity() for p4_lab in self.p4_lab_4vectors]
    
    def get_total_xs(self, s):
        p1_cm = self.p1_cm(s)
        p3_cm = self.p3_cm(s)

        t0 = power(self.m1**2 - self.m3**2 - self.m2**2 + self.m4**2, 2)/(4*s) - power(p1_cm - p3_cm, 2)
        t1 = power(self.m1**2 - self.m3**2 - self.m2**2 + self.m4**2, 2)/(4*s) - power(p1_cm + p3_cm, 2)

        #e1_cm = np.sqrt(p1_cm**2 + self.m1**2)
        #e3_cm = np.sqrt(p3_cm**2 + self.m3**2)
        #t0 = self.m1**2 + self.m3**2 + 2*(p1_cm*p3_cm - e1_cm*e3_cm)
        #t1 = self.m1**2 + self.m3**2 + 2*(-p1_cm*p3_cm - e1_cm*e3_cm)

        def integrand(t):
            return self.dsigma_dt(s, t)

        return quad(integrand, t1, t0)[0]

    def get_cosine_lab_weights(self):
        lab_cosines = np.array([self.p3_lab_4vectors[i].cosine() for i in range(self.n_samples)])
        #cosine_weights = abs(np.array([power(self.p3_lab_3vectors[i].mag()/self.p3_cm_3vectors[i].mag(), 2) * \
        #                (self.p3_cm_3vectors[i]*self.p3_lab_3vectors[i])/(self.p3_cm_3vectors[i].mag()*self.p3_lab_3vectors[i].mag()) \
        #                 * self.dsigma_dcos_cm_wgts[i] for i in range(self.n_samples)]))
        # TODO(AT): check jacobian is needed?
        cosine_weights = self.dsigma_dcos_cm_wgts

        return lab_cosines, cosine_weights
    
    def get_e3_lab_weights(self):
        lab_energies = np.array([self.p3_lab_4vectors[i].energy() for i in range(self.n_samples)])
        return lab_energies, self.dsigma_dcos_cm_wgts  # jacobian * mc volume = 1
    
    def get_e4_lab_weights(self):
        lab_energies = np.array([self.p4_lab_4vectors[i].energy() for i in range(self.n_samples)])
        return lab_energies, self.dsigma_dcos_cm_wgts  # jacobian * mc volume = 1




class Scatter2to3MC:
    """
    2 -> 3 scattering for general masses in fixed target frame
    Based on Byckling, Kajantie ch. V.4
    
    ATTN: assumes particle A is at rest.
    """
    def __init__(self, mtrx2: MatrixElement2to3, p1=LorentzVector(), p2=LorentzVector(), n_samples=100):
        self.mtrx2 = mtrx2

        self.ma = mtrx2.ma
        self.mb = mtrx2.mb
        self.m1 = mtrx2.m1
        self.m2 = mtrx2.m2
        self.m3 = mtrx2.m3

        self.lv_p1 = p1
        self.lv_p2 = p2

        self.n_samples = n_samples

        self.p1_lab_4vectors = []
        self.p1_lab_energies = []
        self.p1_lab_angles = []
        self.dsigma = []

    def cayley_det(self, x, y, z, u, v, w):
        # Gramm-Schmidt det but with invariants
        # Eq. 5.24 of Byckling, Kajantie
        return -0.5 * np.linalg.det([[0, 1, 1, 1, 1],
                                    [1, 0, v, x, z],
                                    [1, v, 0, u, y],
                                    [1, x, u, 0, w],
                                    [1, z, y, w, 0]])

    def kallen(self, x, y, z):
        return np.power((x - y - z), 2) - 4*y*z
    
    def flux_factor(self, s):
        return power(2*sqrt(self.kallen(s, self.ma**2, self.mb**2))*(2*pi)**5, -1)
    
    def s2MaxMin(self, s):
        return (sqrt(s) - self.m1)**2, (self.m2 + self.m3)**2
    
    def t1MaxMin(self, s, s2):
        return self.ma**2 + self.m1**2 - ((s + self.ma**2 - self.mb**2)*(s - s2 + self.m1**2) \
            - np.nan_to_num(sqrt(self.kallen(s, self.ma**2, self.mb**2)*self.kallen(s, s2, self.m1**2))))/(2*s), \
                self.ma**2 + self.m1**2 - ((s + self.ma**2 - self.mb**2)*(s - s2 + self.m1**2) \
            + np.nan_to_num(sqrt(self.kallen(s, self.ma**2, self.mb**2)*self.kallen(s, s2, self.m1**2))))/(2*s)
    
    def phase_space_heaviside(self, s, t1, s2):
        return np.heaviside(np.nan_to_num(-self.cayley_det(s, t1, s2, self.ma**2, self.mb**2, self.m1**2)), 0.0) \
            * np.heaviside(self.kallen(s, self.m1**2, s2), 0.0) \
            * np.heaviside(self.kallen(s2, self.m2**2, self.m3**2), 0.0)
    
    def t2_from_angle(self, cosThetab3, t1, s2):
        # In R23 frame
        return self.mb**2 + self.m3**2 - (s2 + self.mb**2 - t1)*(s2 + self.m3**2 - self.m2**2)/(2*s2) \
            + cosThetab3 * np.nan_to_num(sqrt(self.kallen(s2, self.mb**2, t1)*self.kallen(s2, self.m3**2, self.m2**2)))/(2*s2)

    def s1_from_angle(self, phib, s, t1, s2, t2):
        # in R23 frame
        return s + self.m3**2 - (1/self.kallen(s2, t1, self.mb**2)) \
            * (np.linalg.det([[2*self.mb**2, s2-t1+self.mb**2, self.mb**2 + self.m3**2 - t2],
                             [s2-t1+self.mb**2, 2*s2, s2-self.m2**2 + self.m3**2],
                             [s-self.ma**2 + self.mb**2, s+s2-self.m1**2, 0.0]]) \
                + 2*np.nan_to_num(sqrt(self.cayley_det(s,t1,s2,self.ma**2,self.mb**2,self.m1**2)*self.cayley_det(s2,t2,self.m3**2,t1,self.mb**2,self.m2**2)))*cos(phib))

    def paR23(self, s, t1, s2):
        return sqrt(((s + t1 - self.mb**2 - self.m1**2)/(2*sqrt(s2)))**2 - self.ma**2)
    
    def pbR23(self, s2, t1):
        return sqrt(self.kallen(s2, self.mb**2, t1)/s2) / 2
    
    def p1R23(self, s, s2):
        e1_cm = (s + self.m1**2 - s2)/(2*sqrt(s))
        return np.sqrt(e1_cm**2 - self.m1**2)

    def sinTheta_b1(self, s, t1, s2):
        # R23 frame
        return -4 * s2 * self.cayley_det(s, t1, s2, self.ma**2, self.mb**2, self.m1**2) \
            / (self.kallen(s2, self.mb**2, t1)*self.kallen(s, s2, self.m1**2))
    
    def cosTheta_ab(self, s, s2, t1):
        sinTheta_b1 = self.sinTheta_b1(s, t1, s2)
        cosTheta_b1 = sqrt(1 - sinTheta_b1**2)
        return (self.p1R23(s, s2)*self.pbR23(s2, t1)*cosTheta_b1 - self.pbR23(s2, t1)**2) / (self.pbR23(s2, t1)*self.paR23(s, t1, s2))

    def dsigma_ds2dt1dOmega3(self, s, s2, t1, cosThetab3, phib):
        t2 = self.t2_from_angle(cosThetab3, t1, s2)
        s1 = self.s1_from_angle(phib, s, t1, s2, t2)
        return pi * self.phase_space_heaviside(s, t1, s2) * self.mtrx2(s, s2, t1, s1, t2) * self.flux_factor(s) \
            * np.nan_to_num(sqrt(self.kallen(s, self.ma**2, self.mb**2)/self.kallen(s2, self.m2**2, self.m3**2)))/(16*s2)

    def r23_velocity(self, s2, t1):
        # boost from R23 velocity to lab frame velocity
        return Vector3(0.0, 0.0, -sqrt(self.kallen(s2, self.mb**2, t1))/(s2 + self.mb**2 - t1))

    def get_total_xs(self, s, nitn=30, neval=1000):
        def f(x):
            dsigma = self.dsigma_ds2dt1dOmega3(s, x[0], x[1], x[2], x[3])
            return dsigma
        
        s2Max, s2Min = self.s2MaxMin(s)
        t1Max1, t1Min1 = self.t1MaxMin(s, s2Min)
        t1Max2, t1Min2 = self.t1MaxMin(s, s2Max)
        t1_crit_points = [t1Max1, t1Min1, t1Max2, t1Min2]
        t1Max = max(t1_crit_points)
        t1Min = min(t1_crit_points)
        s2_cutoff = (s2Max - s2Min)*0.000001

        integ = vegas.Integrator([[s2Min, s2Max], [t1Min, t1Max], [-0.99, 0.99], [-pi+0.01, pi-0.01]])

        integ(f, nitn=nitn, neval=neval)
        result = integ(f, nitn=nitn, neval=neval)

        return float(result.mean)
    
    def dsigma_ds2dt1(self, s, s2, t1, nitn=30, neval=1000):
        def f(x):
            dsigma = self.dsigma_ds2dt1dOmega3(s, s2, t1, x[0], x[1])
            return dsigma
        
        integ = vegas.Integrator([[-0.95, 0.95], [-pi, pi]])

        integ(f, nitn=nitn, neval=neval)
        result = integ(f, nitn=nitn, neval=neval)
        
        return float(result.mean)

    def simulate_particle1(self, s):
        s2Max, s2Min = self.s2MaxMin(s)
        t1Max1, t1Min1 = self.t1MaxMin(s, s2Min)
        t1Max2, t1Min2 = self.t1MaxMin(s, s2Max)
        t1Max = max(t1Max1, t1Max2)
        t1Min = min(t1Min1, t1Min2)

        s2_rnd = np.random.uniform(s2Min, s2Max, self.n_samples)
        t1_rnd = np.random.uniform(t1Min, t1Max, self.n_samples)
        cos_rnd = np.random.uniform(-1, 1, self.n_samples)
        phi_rnd = np.random.uniform(0.0, 2*pi, self.n_samples)

        print("t1 min, max = ", t1Min, t1Max)
        print("s2 min, max = ", s2Min, s2Max)
        
        mc_volume = (t1Max - t1Min)*(s2Max - s2Min)*4*pi / self.n_samples
        
        # Get particle 1's CM spectra
        for i in range(self.n_samples):
            if self.phase_space_heaviside(s, t1_rnd[i], s2_rnd[i]) < 1.0:
                continue

            weights = self.dsigma_ds2dt1dOmega3(s, s2_rnd[i], t1_rnd[i], cos_rnd[i], phi_rnd[i])
            e1_cm = (s + self.m1**2 - s2_rnd[i])/(2*sqrt(s))
            p1_cm = np.sqrt(e1_cm**2 - self.m1**2)
            theta_b1_cm = arcsin(sqrt(-4*s2_rnd[i]*self.cayley_det(s, t1_rnd[i], s2_rnd[i], self.ma**2, self.mb**2, self.m1**2) \
                / (self.kallen(s2_rnd[i], self.mb**2, t1_rnd[i])*self.kallen(s, s2_rnd[i], self.m1**2))))
            lorentz_vector_p1_cm = LorentzVector(e1_cm, p1_cm*sin(theta_b1_cm), 0.0, p1_cm*cos(theta_b1_cm))

            # Boost to lab frame: use pa velocity assuming pa is at rest
            pa_R23 = self.paR23(s, t1_rnd[i], s2_rnd[i])
            ea_R23 = sqrt(pa_R23**2 + self.ma**2)
            cosTheta_ab_R23 = self.cosTheta_ab(s, s2_rnd[i], t1_rnd[i])
            va_R23_3vec = Vector3(-pa_R23*sqrt(1-cosTheta_ab_R23**2)/ea_R23, 0.0, -pa_R23*cosTheta_ab_R23/ea_R23)

            #v_cm_to_lab = self.r23_velocity(s2_rnd[i], t1_rnd[i])
            lorentz_vector_p1_lab = lorentz_boost(lorentz_vector_p1_cm, va_R23_3vec)
            self.p1_lab_4vectors.append(lorentz_vector_p1_lab)
            self.p1_lab_energies.append(lorentz_vector_p1_lab.p0)
            #print("dsigma = {}, CM energy = {}, Lab energy = {}".format(weights, lorentz_vector_p1_cm.p0, lorentz_vector_p1_lab.p0))
            self.p1_lab_angles.append(lorentz_vector_p1_lab.theta())
            self.dsigma.append(weights * mc_volume)
    
    def simulate_particle1_phase_space_sampled(self, s):
        s2Max, s2Min = self.s2MaxMin(s)

        s2_grid = np.linspace(s2Min, s2Max, self.n_samples)
        s2_points = (s2_grid[1:] + s2_grid[:-1])/2
        delta_s2 = s2_grid[1] - s2_grid[0]

        cos_rnd = np.random.uniform(-1, 1, self.n_samples)
        phi_rnd = np.random.uniform(0.0, 2*pi, self.n_samples)
        
        # Get particle 1's CM spectra
        for s2 in s2_points:
            t1Max, t1Min = self.t1MaxMin(s, s2)
            t1_rnd = np.random.uniform(t1Min, t1Max, self.n_samples)
            mc_volume = (t1Max - t1Min)*delta_s2*4*pi / self.n_samples
            for i, t1 in enumerate(t1_rnd):
                if self.phase_space_heaviside(s, t1, s2) < 1.0:
                    continue

                weights = self.dsigma_ds2dt1dOmega3(s, s2, t1, cos_rnd[i], phi_rnd[i])
                e1_cm = (s + self.m1**2 - s2)/(2*sqrt(s))
                p1_cm = np.sqrt(e1_cm**2 - self.m1**2)
                theta_b1_cm = arcsin(sqrt(-4*s2*self.cayley_det(s, t1, s2, self.ma**2, self.mb**2, self.m1**2) \
                    / (self.kallen(s2, self.mb**2, t1)*self.kallen(s, s2, self.m1**2))))
                lorentz_vector_p1_cm = LorentzVector(e1_cm, p1_cm*sin(theta_b1_cm), 0.0, p1_cm*cos(theta_b1_cm))

                # Boost to lab frame: use pa velocity assuming pa is at rest
                pa_R23 = self.paR23(s, t1, s2)
                ea_R23 = sqrt(pa_R23**2 + self.ma**2)
                cosTheta_ab_R23 = self.cosTheta_ab(s, s2, t1)
                va_R23_3vec = Vector3(-pa_R23*sqrt(1-cosTheta_ab_R23**2)/ea_R23, 0.0, -pa_R23*cosTheta_ab_R23/ea_R23)

                lorentz_vector_p1_lab = lorentz_boost(lorentz_vector_p1_cm, va_R23_3vec)
                self.p1_lab_4vectors.append(lorentz_vector_p1_lab)
                self.p1_lab_energies.append(lorentz_vector_p1_lab.p0)
                self.p1_lab_angles.append(lorentz_vector_p1_lab.theta())
                self.dsigma.append(weights * mc_volume)




    
class Decay2Body:
    def __init__(self, p_parent: LorentzVector, m1, m2, n_samples=1000):
        self.mp = p_parent.mass()  # parent particle
        self.m1 = m1  # decay body 1
        self.m2 = m2  # decay body 2

        self.lv_p = p_parent

        self.n_samples = n_samples
        self.p1_cm_4vectors = []
        self.p1_lab_4vectors = []
        self.p2_cm_4vectors = []
        self.p2_lab_4vectors = []
        self.weights = np.array([])
    
    def set_new_decay(self, p_parent: LorentzVector, m1, m2):
        self.mp = p_parent.mass()  # parent particle
        self.m1 = m1  # decay body 1
        self.m2 = m2  # decay body 2

        self.lv_p = p_parent

        self.p1_cm_4vectors = []
        self.p1_lab_4vectors = []
        self.p2_cm_4vectors = []
        self.p2_lab_4vectors = []
        self.weights = np.array([])
    
    def decay(self):
        p_cm = power((self.mp**2 - (self.m2 - self.m1)**2)*(self.mp**2 - (self.m2 + self.m1)**2), 0.5)/(2*self.mp)
        e1_cm = sqrt(p_cm**2 + self.m1**2)
        e2_cm = sqrt(p_cm**2 + self.m2**2)

        # Draw random variates on the 2-sphere
        phi1_rnd = 2*pi*np.random.ranf(self.n_samples)
        theta1_rnd = arccos(1 - 2*np.random.ranf(self.n_samples))

        v_in = -self.lv_p.get_3velocity()

        self.p1_cm_4vectors = [LorentzVector(e1_cm,
                            p_cm*cos(phi1_rnd[i])*sin(theta1_rnd[i]),
                            p_cm*sin(phi1_rnd[i])*sin(theta1_rnd[i]),
                            p_cm*cos(theta1_rnd[i])) for i in range(self.n_samples)]
        self.p2_cm_4vectors = [LorentzVector(e2_cm,
                            -p_cm*cos(phi1_rnd[i])*sin(theta1_rnd[i]),
                            -p_cm*sin(phi1_rnd[i])*sin(theta1_rnd[i]),
                            -p_cm*cos(theta1_rnd[i])) for i in range(self.n_samples)]
        self.p1_lab_4vectors = [lorentz_boost(p1, v_in) for p1 in self.p1_cm_4vectors]
        self.p2_lab_4vectors = [lorentz_boost(p2, v_in) for p2 in self.p2_cm_4vectors]
        self.weights = np.ones(self.n_samples) / self.n_samples
    
    def decay_from_flux(self):
        p_cm = power((self.mp**2 - (self.m2 - self.m1)**2)*(self.mp**2 - (self.m2 + self.m1)**2), 0.5)/(2*self.mp)
        e1_cm = sqrt(p_cm**2 + self.m1**2)
        e2_cm = sqrt(p_cm**2 + self.m2**2)

        # Draw random variates on the 2-sphere
        phi1_rnd = 2*pi*np.random.ranf(self.n_samples)
        theta1_rnd = arccos(1 - 2*np.random.ranf(self.n_samples))
        phi2_rnd = np.pi + phi1_rnd
        theta2_rnd = np.pi - theta1_rnd

        v_in = [-lv.get_3velocity() for lv in self.lv_p]

        self.p1_cm_4vectors = [LorentzVector(e1_cm,
                            p_cm*cos(phi1_rnd[i])*sin(theta1_rnd[i]),
                            p_cm*sin(phi1_rnd[i])*sin(theta1_rnd[i]),
                            p_cm*cos(theta1_rnd[i])) for i in range(self.n_samples)]
        self.p2_cm_4vectors = [LorentzVector(e2_cm,
                            p_cm*cos(phi2_rnd[i])*sin(theta2_rnd[i]),
                            p_cm*sin(phi2_rnd[i])*sin(theta2_rnd[i]),
                            p_cm*cos(theta2_rnd[i])) for i in range(self.n_samples)]
        self.p1_lab_4vectors = [lorentz_boost(p1, v_in) for p1 in self.p1_cm_4vectors]
        self.p2_lab_4vectors = [lorentz_boost(p2, v_in) for p2 in self.p2_cm_4vectors]
        self.weights = np.ones(self.n_samples)




class Decay3Body:
    """
    Performs a weighted MC sampling of the differential 3-body decay width
    by choosing a final-state particle of interest (particle #3 by convention)
    and drawing random variates in the rest-frame of the parent; particle 3
    has a random angle on the 2-sphere and a random energy between some E_max and E_min.
    The weight is given by dGamma / dE_3 dOmega
    Finally, we boost to the lab frame with the appropriate Jacobian factor.
    """
    def __init__(self, mtrx2: MatrixElementDecay3, p: LorentzVector, n_samples=1000, total_width=None):
        self.mtrx2 = mtrx2
        self.n_samples = n_samples

        self.parent_p4 = p

        self.m_parent = mtrx2.m_parent
        self.m1 = mtrx2.m1
        self.m2 = mtrx2.m2
        self.m3 = mtrx2.m3

        self.p3_cm_4vectors = []
        self.p3_lab_4vectors = []
        self.weights = np.array([])

        if total_width is None:
            self.total_width = self.partial_width()
        else:
            self.total_width = total_width

    def dGammadE3(self, E3):
        m212 = self.m_parent**2 + self.m3**2 - 2*self.m_parent*E3
        e2star = (m212 - self.m1**2 + self.m2**2)/(2*sqrt(m212))
        e3star = (self.m_parent**2 - m212 - self.m3**2)/(2*sqrt(m212))

        if self.m3 > e3star:
            return 0.0

        m223Max = (e2star + e3star)**2 - (sqrt(e2star**2 - self.m2**2) - sqrt(e3star**2 - self.m3**2))**2
        m223Min = (e2star + e3star)**2 - (sqrt(e2star**2 - self.m2**2) + sqrt(e3star**2 - self.m3**2))**2

        def MatrixElement2(m223):
            return self.mtrx2(m212, m223)

        return (2*self.m_parent)/(32*power(2*pi*self.m_parent, 3))*quad(MatrixElement2, m223Min, m223Max)[0]

    def partial_width(self):
        ea_max = (self.m_parent**2 + self.m3**2 - self.m2**2 - self.m1**2)/(2*self.m_parent)
        return quad(self.dGammadE3, self.m3, ea_max)[0]

    def simulate_decay(self):
        # Simulates the weighted-MC 3-body decay, outputting the 4-vectors of particle #3
        # TODO(AT): add calculations for particles 1,2
        ea_min = self.m3
        ea_max = (self.m_parent**2 + self.m3**2 - self.m2**2 - self.m1**2)/(2*self.m_parent)

        # Boost to lab frame
        beta = self.parent_p4.momentum() / self.parent_p4.energy()
        boost = power(1-beta**2, -0.5)
        beta_parent = -self.parent_p4.get_3velocity()

        # Draw random variate energies and angles in the parent rest frame
        e3_rnd = np.random.uniform(ea_min, ea_max, self.n_samples)
        p3_rnd = sqrt(e3_rnd**2 - self.m3**2)
        theta3_rnd = arccos(1 - 2*np.random.ranf(self.n_samples))
        phi3_rnd = np.random.uniform(0, 2*pi, self.n_samples)

        self.p3_cm_4vectors = [LorentzVector(e3_rnd[i],
                            -p3_rnd[i]*cos(phi3_rnd[i])*sin(theta3_rnd[i]),
                            -p3_rnd[i]*sin(phi3_rnd[i])*sin(theta3_rnd[i]),
                            -p3_rnd[i]*cos(theta3_rnd[i])) for i in range(self.n_samples)]
        self.p3_lab_4vectors = [lorentz_boost(p1, beta_parent) for p1 in self.p3_cm_4vectors]

        # Draw weights from the PDF: (Jacobian) * dGamma/dE_CM * MC volume
        mc_factor = (ea_max - ea_min)/self.total_width/self.n_samples 
        self.weights = np.array([mc_factor*(self.p3_lab_4vectors[i].momentum()/self.p3_cm_4vectors[i].momentum())*self.dGammadE3(e3_rnd[i]) \
                                    for i in range(self.n_samples)])




def lorentz_boost(momentum: LorentzVector, v: Vector3):
    """
    Lorentz boost momentum to a new frame with velocity v
    :param momentum: four vector
    :param v: velocity of new frame, 3-dimention
    :return: boosted momentum
    """
    n = v.unit_vec().vec
    beta = v.mag()
    if beta == 0.0:
        return momentum
    gamma = 1/np.sqrt(1-beta**2)
    mat = np.array([[gamma, -gamma*beta*n[0], -gamma*beta*n[1], -gamma*beta*n[2]],
                    [-gamma*beta*n[0], 1+(gamma-1)*n[0]*n[0], (gamma-1)*n[0]*n[1], (gamma-1)*n[0]*n[2]],
                    [-gamma*beta*n[1], (gamma-1)*n[1]*n[0], 1+(gamma-1)*n[1]*n[1], (gamma-1)*n[1]*n[2]],
                    [-gamma*beta*n[2], (gamma-1)*n[2]*n[0], (gamma-1)*n[2]*n[1], 1+(gamma-1)*n[2]*n[2]]])
    boosted_p4 = mat @ momentum.pmu
    return LorentzVector(boosted_p4[0], boosted_p4[1], boosted_p4[2], boosted_p4[3])