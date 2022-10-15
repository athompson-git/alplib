"""
Cross section class and MC
"""

from alplib.constants import *
from alplib.fmath import *
from alplib.matrix_element import *





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
    
    def __mul__(self, other):
        return np.dot(self.vec, other.vec)
    
    def __rmul__(self, other):
        return np.dot(self.vec, other.vec)
    
    def unit_vec(self):
        v = self.mag()
        return Vector3(self.v1/v, self.v2/v, self.v3/v)
    
    def mag2(self):
        return np.dot(self.vec, self.vec)
    
    def mag(self):
        return np.sqrt(np.dot(self.vec, self.vec))
    
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
        return arctan(self.p2 / self.p1)
    
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


    def dsigma_dt(self, s, t):
        return self.mtrx2(s, t)/(16*np.pi*(s - (self.m1 + self.m2)**2)*(s - (self.m1 - self.m2)**2))

    def p1_cm(self, s):
        return np.sqrt((np.power(s - self.m1**2 - self.m2**2, 2) - np.power(2*self.m1*self.m2, 2))/(4*s))

    def p3_cm(self, s):
        return np.sqrt((np.power(s - self.m3**2 - self.m4**2, 2) - np.power(2*self.m3*self.m4, 2))/(4*s))

    def scatter_sim(self):
        # Takes in initial energy-momenta for p1, p2
        # Computes CM frame energies
        # Simulates events in CM frame

        # Draw random variates on the 2-sphere
        phi_rnd = 2*pi*np.random.ranf(self.n_samples)
        theta_rnd = arccos(1 - 2*np.random.ranf(self.n_samples))

        # Declare momenta and energy in the CM frame
        cm_p4 = self.lv_p1 + self.lv_p2
        e_in = cm_p4.energy()
        p_in = cm_p4.get_3momentum()
        v_in = Vector3(-p_in.v1 / e_in, -p_in.v2 / e_in, -p_in.v3 / e_in)
        s = cm_p4.mass2()
        if s < (self.m3 + self.m4)**2:
            return

        p1_cm = self.p1_cm(s)
        p3_cm = self.p3_cm(s)
        e1_cm = np.sqrt(p1_cm**2 + self.m1**2)
        e3_cm = np.sqrt(p3_cm**2 + self.m3**2)

        t_rnd = self.m1**2 + self.m3**2 + 2*(p1_cm*p3_cm*cos(theta_rnd) - e1_cm*e3_cm)

        self.dsigma_dcos_cm_wgts = 4*p1_cm*p3_cm*self.dsigma_dt(s, t_rnd)/self.n_samples

        # Boosts back to original frame
        self.p3_cm_4vectors = [LorentzVector(e3_cm,
                            p3_cm*cos(phi_rnd[i])*sin(theta_rnd[i]),
                            p3_cm*sin(phi_rnd[i])*sin(theta_rnd[i]),
                            p3_cm*cos(theta_rnd[i])) for i in range(self.n_samples)]
        self.p3_lab_4vectors = [lorentz_boost(p3, v_in) for p3 in self.p3_cm_4vectors]
        self.p3_cm_3vectors = [p3_cm.get_3velocity() for p3_cm in self.p3_cm_4vectors]
        self.p3_lab_3vectors = [p3_lab.get_3velocity() for p3_lab in self.p3_lab_4vectors]

        self.p4_lab_4vectors = [self.lv_p1 + self.lv_p2 - p3_lab for p3_lab in self.p3_lab_4vectors]
        self.p4_lab_3vectors = [p4_lab.get_3velocity() for p4_lab in self.p4_lab_4vectors]
    
    def get_total_xs(self, s):
        p1_cm = self.p1_cm(s)
        p3_cm = self.p3_cm(s)

        t0 = power(self.m1**2 - self.m3**2 - self.m2**2 + self.m4**2, 2)/(4*s) - power(p1_cm - p3_cm, 2)
        t1 = power(self.m1**2 - self.m3**2 - self.m2**2 + self.m4**2, 2)/(4*s) - power(p1_cm + p3_cm, 2)

        def integrand(t):
            return self.dsigma_dt(s, t)

        return quad(integrand, t1, t0)[0]

    def get_cosine_lab_weights(self):
        lab_cosines = np.array([self.p3_lab_4vectors[i].cosine() for i in range(self.n_samples)])
        cosine_weights = abs(np.array([power(self.p3_lab_3vectors[i].mag()/self.p3_cm_3vectors[i].mag(), 2) * \
                        (self.p3_cm_3vectors[i]*self.p3_lab_3vectors[i])/(self.p3_cm_3vectors[i].mag()*self.p3_lab_3vectors[i].mag()) \
                         * self.dsigma_dcos_cm_wgts[i] for i in range(self.n_samples)]))

        return lab_cosines, cosine_weights
    
    def get_e3_lab_weights(self):
        lab_energies = np.array([self.p3_lab_4vectors[i].energy() for i in range(self.n_samples)])
        return lab_energies, self.dsigma_dcos_cm_wgts  # jacobian * mc volume = 1




class Decay2Body:
    def __init__(self, mtrx2: MatrixElementDecay2, p: LorentzVector, n_samples=1000):
        self.mtrx2 = mtrx2
        self.mp = mtrx2.m_parent  # parent particle
        self.m1 = mtrx2.m1  # decay body 1
        self.m2 = mtrx2.m2  # decay body 2

        self.lv_p = p

        self.n_samples = n_samples
        self.p1_cm_4vectors = []
        self.p1_lab_4vectors = []
        self.p2_cm_4vectors = []
        self.p2_lab_4vectors = []
        self.weights = np.array([])
    
    def decay(self):
        p_cm = power((self.mp**2 - (self.m2 - self.m1)**2)*(self.mp**2 - (self.m2 + self.m1)**2), 0.5)/(2*self.mp)
        e1_cm = sqrt(p_cm**2 + self.m1**2)
        e2_cm = sqrt(p_cm**2 + self.m2**2)
        decay_width = (p_cm / (8*np.pi*self.mp**2)) * self.mtrx2()

        # Draw random variates on the 2-sphere
        phi1_rnd = 2*pi*np.random.ranf(self.n_samples)
        theta1_rnd = arccos(1 - 2*np.random.ranf(self.n_samples))

        v_in = self.lv_p.get_3velocity()

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
        self.weights = decay_width * np.ones(self.n_samples)
    
    def decay_from_flux(self):
        p_cm = power((self.mp**2 - (self.m2 - self.m1)**2)*(self.mp**2 - (self.m2 + self.m1)**2), 0.5)/(2*self.mp)
        e1_cm = sqrt(p_cm**2 + self.m1**2)
        e2_cm = sqrt(p_cm**2 + self.m2**2)
        decay_width = (p_cm / (8*np.pi*self.mp**2)) * self.mtrx2()

        # Draw random variates on the 2-sphere
        phi1_rnd = 2*pi*np.random.ranf(self.n_samples)
        theta1_rnd = arccos(1 - 2*np.random.ranf(self.n_samples))
        phi2_rnd = np.pi + phi1_rnd
        theta2_rnd = np.pi - theta1_rnd

        v_in = [lv.get_3velocity() for lv in self.lv_p]

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
        self.weights = decay_width * np.ones(self.n_samples)




class Decay3Body:
    def __init__(self, mtrx2: MatrixElementDecay3, p: LorentzVector, n_samples=1000):
        pass

    def decay(self):
        pass




def lorentz_boost(momentum: LorentzVector, v: Vector3):
    """
    Lorentz boost momentum to a new frame with velocity v
    :param momentum: four vector
    :param v: velocity of new frame, 3-dimention
    :return: boosted momentum
    """
    n = v.unit_vec().vec
    beta = v.mag()
    gamma = 1/np.sqrt(1-beta**2)
    mat = np.array([[gamma, -gamma*beta*n[0], -gamma*beta*n[1], -gamma*beta*n[2]],
                    [-gamma*beta*n[0], 1+(gamma-1)*n[0]*n[0], (gamma-1)*n[0]*n[1], (gamma-1)*n[0]*n[2]],
                    [-gamma*beta*n[1], (gamma-1)*n[1]*n[0], 1+(gamma-1)*n[1]*n[1], (gamma-1)*n[1]*n[2]],
                    [-gamma*beta*n[2], (gamma-1)*n[2]*n[0], (gamma-1)*n[2]*n[1], 1+(gamma-1)*n[2]*n[2]]])
    boosted_p4 = mat @ momentum.pmu
    return LorentzVector(boosted_p4[0], boosted_p4[1], boosted_p4[2], boosted_p4[3])