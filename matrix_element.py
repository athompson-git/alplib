"""
Matrix element class
"""

from .fmath import *
from .constants import *
from .form_factors import *



class MatrixElement2:
    def __init__(self, m1, m2, m3, m4):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4

    def __call__(self, s, t):
        return 0.0




class MatrixElementDecay2:
    def __init__(self, m_parent, m1, m2):
        self.m_parent = m_parent
        self.m1 = m1
        self.m2 = m2
    
    def __call__(self):
        return 0.0




class MatrixElementDecay3:
    def __init__(self, m_parent, m1, m2, m3):
        self.m_parent = m_parent
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
    
    def __call__(self):
        return 0.0




class M2VectorDecayToFermions(MatrixElementDecay2):
    def __init__(self, m_parent, m):
        super().__init__(m_parent, m, m)
    
    def __call__(self, coupling=1.0):
        return 4*(coupling**2)*(self.m_parent**2 - 2*self.m1**2)



class M2Chi2ToChi1Vector(MatrixElementDecay2):
    def __init__(self, m_chi2, m_chi1, m_v):
        super().__init__(m_chi2, m_chi1, m_v)
        # m_parent = m_chi2
        # m1 = m_chi1
        # m2 = m_v
    
    def __call__(self, coupling=1.0):
        return coupling**2 * (12*self.m1*self.m_parent - 2*self.m_parent**2 \
            - 2*power(self.m_parent/self.m2, 2) * (self.m_parent**2 - self.m1**2 - self.m2**2))



class M2DMUpscatter(MatrixElement2):
    """
    Dark matter upscattering (chi1 + N -> chi2 + N) via heavy mediator V
    """
    def __init__(self, mchi1, mchi2, mV, mN):
        super().__init__(mchi1, mN, mchi2, mN)
        self.mV = mV
        self.mchi1 = mchi1
        self.mchi2 = mchi2
        self.mN = mN
        self.ff = ProtonFF()
    
    def __call__(self, s, t, coupling_product=1.0):
        prefactor = ALPHA * self.ff(t) * coupling_product**2
        propagator = power(t - self.mV**2, 2)
        numerator = 8*(2*power(self.mN,4) + 4*self.mN**2 * (self.mchi1 * self.mchi2 - s) \
            + 2*(self.mchi1**2 - s)*(self.mchi2**2-s) - t*(self.mchi1-self.mchi2)**2 + 2*s*t + t**2)
        return prefactor * numerator / propagator




class M2DarkPrimakoff(MatrixElement2):
    """
    Dark Primakoff scattering (a + N -> gamma + N) via heavy mediator Zprime
    """
    def __init__(self, ma, mN, mZp):
        super().__init__(ma, mN, 0, mN)
        self.mZp = mZp
        self.mN = mN
        self.ma = ma
    
    def __call__(self, s, t, coupling_product=1.0):
        prefactor = ALPHA * coupling_product**2
        propagator = power(t - self.mZp**2, 2)
        numerator = (2*self.mN**2 * (self.ma**2 - 2*s - t) + 2*self.mN**4 - 2*self.ma**2 * (s + t) + self.ma**4 + 2*s**2 + 2*s*t + t**2)
        return prefactor * numerator / propagator




class M2VectorScalarPrimakoff(MatrixElement2):
    """
    Zp + N -> gamma + N via massive scalar mediator
    """
    def __init__(self, mphi, mZp, mat: Material):
        super().__init__(mZp, mat.m[0], 0, mat.m[0])
        self.mZp = mZp
        self.mN = mat.m[0]
        self.mphi = mphi
        self.ff2 = NuclearHelmFF(mat)
    
    def __call__(self, s, t, coupling_product=1.0):
        return self.ff2(np.sqrt(abs(t))) * coupling_product**2 * (2*self.mN**2 - t) * power((self.mZp**2 - t)/(2*(self.mphi**2 - t)),2) 




class M2VectorPseudoscalarPrimakoff(MatrixElement2):
    """
    Zp + N -> gamma + N via massive scalar mediator
    """
    def __init__(self, mphi, mZp, mat: Material):
        super().__init__(mZp, mat.m[0], 0, mat.m[0])
        self.mZp = mZp
        self.mN = mat.m[0]
        self.mphi = mphi
        self.ff2 = NuclearHelmFF(mat)
    
    def __call__(self, s, t, coupling_product=1.0):
        return - self.ff2(np.sqrt(abs(t))) * coupling_product**2 * t * power((self.mZp**2 - t)/(2*(self.mphi**2 - t)),2)




class M2PairProduction:
    """
    a + N -> e+ e- N    ALP-driven pair production
    """
    def __init__(self, ma, mat: Material):
        self.ma = ma
        self.mN = mat.m[0]
        self.ff2 = NuclearHelmFF(mat)
    
    def m2(self, Ea, Ep, tp, tm, phi, coupling_product=1.0):
        # k: ALP momentum
        # p1: positron momentum
        # p2: electron momentum
        # l: initial nucleus momentum
        # q: final nucleus momentum
        c1 = 1.0 #cos(tp)
        c2 = 1.0 #cos(tm)
        s1 = tp #sin(tp)
        s2 = tm #sin(tm)
        cphi = 0.0 #cos(phi)

        p1 = sqrt(Ep**2 - M_E**2)
        Em = Ea - Ep
        p2 = sqrt(Em**2 - M_E**2)
        k = sqrt(Ea**2 - self.ma**2)

        # 3-vector dot products
        l2_dot_k = self.ma**2 - k*p1*c1 - k*p2*c2
        l2_dot_p1 = k*p1*c1 - M_E**2 - p1*p2*(s1*s2*cphi + c1*c2)
        l2_dot_p2 = k*p2*c2 - M_E**2 - p1*p2*(s1*s2*cphi + c1*c2)
        p1_dot_p2 = p1*p2*(s1*s2*cphi + c1*c2)

        # 4-vector scalar products
        kp1 = Ea*Ep - k*p1*c1
        kp2 = Ea*Em - k*p2*c2
        kl1 = Ea*self.mN
        kl2 = Ea*self.mN - l2_dot_k
        p1p2 = Ep*Em - p1_dot_p2
        p1l1 = Ep*self.mN
        p2l1 = Em*self.mN
        p1l2 = Ep*self.mN - l2_dot_p1
        p2l2 = Em*self.mN - l2_dot_p2

        m1_2 = -32 * ( (M_E**2 - kp1)*(2*kl2*p2l1 + 2*kl1*p2l2) + (self.ma**2 - 2*M_E**2)*(p2l1*p1l2 + p1l1*p2l2) )
        m2_2 = -32 * ( (M_E**2 - kp2)*(2*kl2*p1l1 + 2*kl1*p1l2) + (self.ma**2 - 2*M_E**2)*(p2l1*p1l2 + p1l1*p2l2) )
        m2_m1 = -32 * ( kp1 * (kl2*p2l1 + kl1*p2l2 - power(M_E*self.mN, 2)) \
                        + kp2 * (kl2*p1l1 + kl1*p1l2 - power(M_E*self.mN, 2)) \
                        - 2*kl1*kl2*p1p2 - p2l1*p1l2*self.ma**2 + (M_E**2 - self.ma**2)*p1l1*p2l2 \
                        + p2l1*p1l2*M_E**2 + p1p2*power(M_E*self.ma,2) + power(self.mN*M_E**2, 2) )
        
        q2 = self.ma**2 + 2*M_E**2 - 2*kp1 - 2*kp2 + 2*p1p2

        propagator1 = q2*(self.ma**2 - 2*kp1)
        propagator2 = q2*(self.ma**2 - 2*kp2)

        prefactor = power(4*pi*ALPHA*coupling_product, 2) * self.ff2(sqrt(abs(q2)))

        return prefactor * (m1_2 / power(propagator1, 2) \
                            + m2_2 / power(propagator2, 2) \
                            + 2 * m2_m1 / propagator2 / propagator1)


        
        