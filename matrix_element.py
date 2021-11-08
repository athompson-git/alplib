"""
Matrix element class
"""

from .fmath import *
from .constants import *



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
        self.m1 = mchi1
        self.m2 = mchi2
        self.mN = mN
    
    def __call__(self, s, t, coupling_product=1.0):
        prefactor = ALPHA * coupling_product**2
        propagator = power(t - self.mV**2, 2)
        numerator = 8*(2*power(self.mN,4) + 4*self.mN**2 * (self.m1 * self.m2 - s) \
            + 2*(self.m1**2 - s)*(self.m2**2-s) - t*(self.m1-self.m2)**2 + 2*s*t + t**2)
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




