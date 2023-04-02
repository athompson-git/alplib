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




class MatrixElement2to3:
    def __init__(self, ma, mb, m1, m2, m3):
        self.ma = ma
        self.mb = mb
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

    def __call__(self, s, s2, t1, s1, t2):
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

    def __call__(self, m122, m232):
        return 0.0

    def get_sp_from_dalitz(self, m122, m232):
        sp03 = (self.m_parent**2 + self.m3**2 - m122)/2
        sp01 = (self.m_parent**2 + self.m1**2 - m232)/2
        sp02 = (m122 + m232 - self.m1**2 - self.m3**2)/2
        sp13 = (self.m_parent**2 + self.m2**2 - m122 - m232)/2
        sp23 = (m232 - self.m2**2 - self.m3**2)/2
        sp12 = (m122 - self.m1**2 - self.m2**2)/2

        return sp01, sp02, sp03, sp12, sp13, sp23




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




class M2DiphotonDecay(MatrixElementDecay2):
    def __init__(self, m_parent):
        super().__init__(m_parent, 0.0, 0.0)

    def __call__(self, coupling=1.0):
        # Takes coupling in MeV^-1
        return (coupling**2)*(self.m_parent**4)/8




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
    def __init__(self, ma, mZp, mN, n, z):
        super().__init__(ma, mN, 0, mN)
        self.mZp = mZp
        self.mN = mN
        self.ma = ma
        self.z = z
        self.ff2 = ProtonFF() #NuclearHelmFF(n, z)

    def __call__(self, s, t, coupling_product=1.0):
        prefactor = coupling_product**2
        propagator = power(t - self.mZp**2, 2)
        numerator = -t*(2*self.mN**2 * (self.ma**2 - 2*s - t) + 2*self.mN**4 - 2*self.ma**2 * (s + t) + self.ma**4 + 2*s**2 + 2*s*t + t**2)
        return self.z * self.ff2(t) * prefactor * numerator / propagator




class M2VectorScalarPrimakoff(MatrixElement2):
    """
    Zp + N -> gamma + N via massive scalar mediator
    """
    def __init__(self, mphi, mZp, mN, n, z):
        super().__init__(mZp, mN, 0, mN)
        self.mZp = mZp
        self.mN = mN
        self.mphi = mphi
        self.ff2 = NuclearHelmFF(n, z)

    def __call__(self, s, t, coupling_product=1.0):
        return 1.5*abs(self.ff2(np.sqrt(abs(t))) * coupling_product**2 * (4*self.mN**2 - t) * power((self.mZp**2 - t)/(2*(self.mphi**2 - t)),2))




class M2VectorPseudoscalarPrimakoff(MatrixElement2):
    """
    Zp + N -> gamma + N via massive scalar mediator
    """
    def __init__(self, mphi, mZp, mN, n, z):
        super().__init__(mZp, mN, 0, mN)
        self.mZp = mZp
        self.mN = mN
        self.mphi = mphi
        self.ff2 = NuclearHelmFF(n, z)

    def __call__(self, s, t, coupling_product=1.0):
        return abs(self.ff2(np.sqrt(abs(t))) * coupling_product**2 * (4*self.mN**2 - t) * power((self.mZp**2 - t)/(2*(self.mphi**2 - t)),2))




class M2PairProduction:
    """
    a + N -> e+ e- N    ALP-driven pair production
    """
    def __init__(self, ma, mN, n, z, ml=M_E):
        self.ma = ma
        self.mN = mN
        self.ff2 = AtomicElasticFF(z)
        self.ml = ml

    def sub_elements(self, kp1, kp2, kl1, kl2, p1p2, p1l1, p2l1, p1l2, p2l2, case="alp"):
        if case == "alp":
            m1_2 = -32 * ( (M_E**2 - kp1)*(2*kl2*p2l1 + 2*kl1*p2l2) + (self.ma**2 - 2*M_E**2)*(p2l1*p1l2 + p1l1*p2l2) )
            m2_2 = -32 * ( (M_E**2 - kp2)*(2*kl2*p1l1 + 2*kl1*p1l2) + (self.ma**2 - 2*M_E**2)*(p2l1*p1l2 + p1l1*p2l2) )
            m2_m1 = -32 * ( kp1 * (kl2*p2l1 + kl1*p2l2 - power(M_E*self.mN, 2)) \
                            + kp2 * (kl2*p1l1 + kl1*p1l2 - power(M_E*self.mN, 2)) \
                            - 2*kl1*kl2*p1p2 - p2l1*p1l2*self.ma**2 + (M_E**2 - self.ma**2)*p1l1*p2l2 \
                            + p2l1*p1l2*M_E**2 + p1p2*power(M_E*self.ma,2) + power(self.mN*M_E**2, 2) )
            return m1_2, m2_2, m2_m1
        elif case == "vector":
            return 0.0
        elif case == "sm":
            m1_2 = -128.0 * ( kl2*p2l1*(M_E**2 - kp1) + p2l2*(kl1*(M_E**2 - kp1) - M_E**2 * p1l1) - M_E**2 * p1l2*p2l1 )
            m2_2 = -128.0 * ( kl2*p1l1*(M_E**2 - kp2) + p1l2*(kl1*(M_E**2 - kp2) - M_E**2 * p2l1) - M_E**2 * p2l2*p1l1 )
            m2_m1 = -64.0 * ( -kp1*(p2l1*(p1l2 - 2*p2l2) + p1l1*p2l2 + (M_E*self.mN)**2) \
                            - kp2*(p1l1*(p2l2 - 2*p1l2) + p2l1*p1l2 + (M_E*self.mN)**2) \
                            + p2l2*(M_E**2 * (kl1 + p1l1 - 2*p2l1) - p1p2*(kl1 - 2*p1l1)) \
                            + M_E**2 * (kl2*p1l1 + kl1*p1l2 + kl2*p2l1 + p2l1*p1l2 - 2*p1l1*p1l2) \
                            - kl2*p1l1*p1p2 - kl2*p2l1*p1p2 - kl1*p1p2*p1l2 + 2*p2l1*p1p2*p1l2 \
                            - power(M_E*self.mN, 2)*p1p2 + power(M_E, 4)*power(self.mN, 2))
            return m1_2, m2_2, m2_m1
        else:
            print("case=", case, " not found in M2PairProduction.")
            raise Exception()


    def m2(self, Ea, Ep, tp, tm, phi, coupling=1.0, case="alp"):
        # k: ALP momentum
        # p1: positron momentum
        # p2: electron momentum
        # l1: initial nucleus momentum
        # l2: final nucleus momentum
        c1 = cos(tp)
        c2 = cos(tm)
        s1 = sin(tp)
        s2 = sin(tm)
        cphi = cos(phi)

        p1 = sqrt(Ep**2 - M_E**2)
        Em = Ea - Ep
        p2 = sqrt(Em**2 - M_E**2)
        k = sqrt(Ea**2 - self.ma**2)

        # 3-vector dot products
        l2_dot_k = k**2 - k*p1*c1 - k*p2*c2
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

        m1_2, m2_2, m2_m1 = self.sub_elements(kp1, kp2, kl1, kl2, p1p2, p1l1, p2l1, p1l2, p2l2, case)

        q2 = self.ma**2 + 2*M_E**2 - 2*kp1 - 2*kp2 + 2*p1p2

        propagator1 = q2*(self.ma**2 - 2*kp1)
        propagator2 = q2*(self.ma**2 - 2*kp2)

        prefactor = power(4*pi*ALPHA*coupling, 2) * self.ff2(sqrt(abs(q2)))

        return prefactor * (m1_2 / power(propagator1, 2) \
                            + m2_2 / power(propagator2, 2) \
                            + 2 * m2_m1 / propagator2 / propagator1)

    def m2_separated(self, Ea, Ep, tp, tm, phi, coupling=1.0, case="alp"):
        # k: ALP momentum
        # p1: positron momentum
        # p2: electron momentum
        # l1: initial nucleus momentum
        # l2: final nucleus momentum
        c1 = cos(tp)
        c2 = cos(tm)
        s1 = sin(tp)
        s2 = sin(tm)
        cphi = cos(phi)

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

        m1_2, m2_2, m2_m1 = self.sub_elements(kp1, kp2, kl1, kl2, p1p2, p1l1, p2l1, p1l2, p2l2, case)

        q2 = self.ma**2 + 2*M_E**2 - 2*kp1 - 2*kp2 + 2*p1p2

        propagator1 = q2*(self.ma**2 - 2*kp1)
        propagator2 = q2*(self.ma**2 - 2*kp2)

        prefactor = power(4*pi*ALPHA*coupling, 2) * self.ff2(sqrt(abs(q2)))

        return prefactor * m1_2 / power(propagator1, 2), \
                prefactor * m2_2 / power(propagator2, 2), \
                prefactor * 2 * m2_m1 / (propagator2*propagator1)
    
    def m2_v2(self, Ea, Ep, tp, tm, phi, coupling=1.0):
        # k: ALP momentum
        # p1: positron momentum
        # p2: electron momentum
        # l1: initial nucleus momentum
        # l2: final nucleus momentum
        c1 = cos(tp)
        c2 = cos(tm)
        s1 = sin(tp)
        s2 = sin(tm)
        cphi = cos(phi)

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
        kl1 = Ea*Ep - k*p1*c1
        kl2 = Ea*Em - k*p2*c2
        kp1 = Ea*self.mN
        kp2 = Ea*self.mN - l2_dot_k
        l1l2 = Ep*Em - p1_dot_p2
        l1p1 = Ep*self.mN
        l2p1 = Em*self.mN
        l1p2 = Ep*self.mN - l2_dot_p1
        l2p2 = Em*self.mN - l2_dot_p2
        p1p2 = self.mN**2

        q2 = self.ma**2 + 2*M_E**2 - 2*kp1 - 2*kp2 + 2*p1p2
        prefactor = power(4*pi*ALPHA*coupling / q2, 2) * self.ff2(sqrt(abs(q2)))
        lh = (1/((self.ma**2-2*kl1)**2*(self.ma**2-2*kl2)**2))*16*((2*((p1p2-3*self.mN**2)*M_E**2+2*l1p1*(l2p1+l2p2)-l1l2*(self.mN**2+p1p2))*self.ma**2+Ea*self.mN*(2*(self.mN**2-p1p2)*M_E**2+l1p1*(l2p2-l2p1))+kp2*(2*(self.mN**2-p1p2)*M_E**2+l1p1*(l2p1-l2p2)+4*Ea*(M_E**2+l1l2)*self.mN))*self.ma**4-2*kl2*((2*(p1p2-3*self.mN**2)*M_E**2+l1p1*(self.mN**2+4*l2p1+4*l2p2-p1p2)-2*l1l2*(self.mN**2+p1p2))*self.ma**2+kp2*(-l2p1*M_E**2-p1p2*M_E**2+self.mN*(4*Ea*(M_E**2+l1l2)-l1l2*self.mN)+l1p1*(M_E**2+2*self.ma**2+3*l2p1-l2p2))+Ea*self.mN*(-(-4*self.mN**2+l2p2+3*p1p2)*M_E**2+l1l2*self.mN**2+l1p1*(M_E**2+2*self.ma**2-2*l2p1)))*self.ma**2+4*kl2**3*(M_E**2-l1p1)*self.mN**2+4*kl1**3*(M_E**2+2*kl2-l2p2)*self.mN**2-2*kl1**2*(7*self.mN**2*self.ma**2*M_E**2+2*Ea*l1p1*self.mN*M_E**2-8*kl2**2*self.mN**2+2*l1l2*self.mN**2*self.ma**2-3*l2p2*self.mN**2*self.ma**2-2*l1p1*l2p1*self.ma**2-2*l1p1*l2p2*self.ma**2-2*Ea*l2p2*self.mN*self.ma**2-2*Ea*l1p1*l2p2*self.mN+(l2p2-3*M_E**2)*self.ma**2*p1p2+2*kl2*(2*kp2*(l1p1+l2p1)+self.mN*(2*Ea*(l1p1+l2p2)+self.mN*(M_E**2+4*self.ma**2+l2p2))-(l1p1+l2p2)*p1p2)+2*kp2*(p1p2*M_E**2-(2*M_E**2+l1l2)*self.mN**2-l2p1*self.ma**2+l1p1*(M_E**2+l2p1)))+2*kl2**2*((3*p1p2*M_E**2-(7*M_E**2+2*l1l2)*self.mN**2+l1p1*(3*self.mN**2+2*l2p1+2*l2p2-p1p2))*self.ma**2+2*kp2*(l1p1*(self.ma**2+l2p1)-l2p1*M_E**2)+2*Ea*self.mN*(-(-2*self.mN**2+l2p2+p1p2)*M_E**2+l1l2*self.mN**2+l1p1*(self.ma**2-l2p1)))+kl1*(8*self.mN**2*kl2**3-4*((M_E**2+4*self.ma**2+l1p1+l2p1-l2p2)*self.mN**2+2*Ea*(l1p1+l2p2)*self.mN+2*kp2*(l1p1+l2p1)-(l1p1+l2p1)*p1p2)*kl2**2+2*(((2*M_E**2+4*self.ma**2+2*l2p1-l2p2)*self.mN**2+l1p1*(self.mN**2+4*l2p1+4*l2p2-3*p1p2)-(2*M_E**2+4*l1l2+2*l2p1+l2p2)*p1p2)*self.ma**2+2*Ea*self.mN*(-p1p2*M_E**2-l1l2*self.mN**2+(l1p1+l2p2)*(M_E**2+3*self.ma**2))+2*kp2*(-p1p2*M_E**2+self.mN*(4*Ea*(M_E**2+l1l2)-l1l2*self.mN)+l2p1*(M_E**2+3*self.ma**2)+l1p1*(M_E**2+3*self.ma**2+l2p1-l2p2)))*kl2-self.ma**2*(((-12*M_E**2-4*l1l2+l2p1+l2p2)*self.mN**2+8*l1p1*(l2p1+l2p2)-(-4*M_E**2+4*l1l2+l2p1+l2p2)*p1p2)*self.ma**2+2*kp2*(-3*p1p2*M_E**2-l1p1*(M_E**2+l2p1+l2p2)+4*Ea*l1l2*self.mN+self.mN*(4*(Ea+self.mN)*M_E**2+l1l2*self.mN)+l2p1*(M_E**2+2*self.ma**2))-2*Ea*self.mN*(p1p2*M_E**2+l1l2*self.mN**2+l1p1*(M_E**2-2*l2p2)-l2p2*(M_E*2+2*self.ma**2)))))
        return prefactor * lh



class M2Meson3BodyDecay(MatrixElementDecay3):
    # Radiative decay M -> l(1) nu(2) X(3), X radiated from hadronic meson current
    def __init__(self, mx, meson="pion", lepton_mass=M_E, interaction_model="vector_ib2", abd=(0,0,0)):
        param_dict = {
            "pion": [M_PI, V_UD, F_PI, PION_WIDTH],
            "kaon": [M_K, V_US, F_K, KAON_WIDTH]
        }
        decay_params = param_dict[meson]
        m_meson = decay_params[0]
        self.abd = abd  # QCD free parameters
        self.ckm = decay_params[1]
        self.fM = decay_params[2]
        self.total_width = decay_params[3]
        super().__init__(m_meson, lepton_mass, 0.0, mx)
        # 0: parent meson
        # 1: lepton
        # 2: neutrino
        # 3: dark boson
        self.interaction = interaction_model
        # vector_ib2: vector radiated off pion leg
        # scalar_ib2: scalar radiated off pion leg
        # vector_contact: 4-point contact interaction
        # sd: structure-dependent interaction with vector meson form factors

    def __call__(self, m122, m232, c0=0.0, coupling=1.0):
        lp, pq, kp, lq, kl, kq = super().get_sp_from_dalitz(m122, m232)
        if self.interaction == "scalar_ib1":
            e3star = (self.m_parent**2 - m122 - self.m3**2)/(2*sqrt(m122))
            ev = (m122 + m232 - self.m1**2 - self.m3**2)/(2*self.m_parent)
            emu = (self.m_parent**2 - m232 + self.m1**2)/(2*self.m_parent)
            q2 = self.m_parent**2 - 2*self.m_parent*ev

            prefactor = heaviside(e3star-self.m3,0.0)*(coupling*G_F*self.fM*self.ckm/(q2 - self.m1**2))**2
            return prefactor*((2*self.m_parent*emu*q2 * (q2 - self.m1**2) \
                - (q2**2 - (self.m1*self.m_parent)**2)*(q2 + self.m1**2 - self.m3**2)) \
                    + (2*q2*self.m1**2 * (self.m_parent**2 - q2)))
        if self.interaction == "pseudoscalar_ib1":
            e3star = (self.m_parent**2 - m122 - self.m3**2)/(2*sqrt(m122))
            ev = (m122 + m232 - self.m1**2 - self.m3**2)/(2*self.m_parent)
            emu = (self.m_parent**2 - m232 + self.m1**2)/(2*self.m_parent)
            q2 = self.m_parent**2 - 2*self.m_parent*ev

            prefactor = heaviside(e3star-self.m3,0.0)*(coupling*G_F*self.fM*self.ckm/(q2 - self.m1**2))**2
            return prefactor*((2*self.m_parent*emu*q2 * (q2 - self.m1**2) \
                - (q2**2 - (self.m1*self.m_parent)**2)*(q2 + self.m1**2 - self.m3**2)) \
                    - (2*q2*self.m1**2 * (self.m_parent**2 - q2)))
        if self.interaction == "vector_ib1":
            q2 = self.m_parent**2 - 2*self.m_parent*(m122 + m232 - self.m1**2 - self.m3**2)/(2*self.m_parent)
            prefactor = 8*power(G_F*self.fM*self.ckm/(q2 - self.m1**2)/self.m3, 2)
            kl, kq, kp, lq, lp, pq = super().get_sp_from_dalitz(m122, m232)
            cr = coupling
            cl = coupling
            return -prefactor * ((power(cr*self.m_parent*self.m1,2) - power(cl*q2,2)) * (lq*self.m3**2 + 2*lp*pq) \
                - 2*cr*self.m1**2 * kq * (cr*self.m3**2 * kl + 2*cr*kp*lp - 3*cl*q2*self.m3**2))
        if self.interaction == "scalar_ib2":
            return 0.0
        if self.interaction == "vector_ib2":
            prefactor_IB2 = 2*power(coupling * G_F * self.fM * self.m1, 2)
            return -prefactor_IB2 * ((self.m1**2-m122)*((self.m_parent**2-m122+self.m3**2)**2 \
                - 4*self.m_parent**2 * self.m3**2))/(self.m3**2 * (self.m_parent**2-m122)**2)
        if self.interaction == "vector_ib9":
            a, b, d = self.abd
            return self.MIB9(g=coupling, m122=m122, m232=m232, mV=self.m3, M=self.m_parent,
                ml=self.m1, a=a, b=b, d=d, f_pi=self.fM, Gf=G_F)
        if self.interaction == "sd":
            lp, pq, kp, lq, kl, kq = super().get_sp_from_dalitz(m122, m232)
            prefactor_SD = 8 * power(coupling * G_F * CABIBBO / sqrt(2) / self.m_parent, 2)
            fv = 0.0265
            fa = 0.58 * fv
            return prefactor_SD * (2 * kl * (power(fa - fv, 2)*kp*pq - self.m_parent**2 * (fa**2 + fv**2)*kq) \
                    + self.m3**2 * (power(fa*self.m_parent, 2) * lq - 2*fv**2 * lp * pq) \
                    + 2 * power(fa + fv, 2) * kp * kq * lp)
        if self.interaction == "vector_contact":
            denom = power((c0 + 1) * self.m3 * (self.m3**2 - kp), 2)
            numerator1 = 8 * self.m3**2 * (2*c0*(kq*lp)*(kp-self.m3**2) \
                - lq*((1-c0**2)*kp**2 - 2*self.m3**2 * kp + power(c0*self.m_parent*self.m3,2) + self.m3**4))
            numerator2 = 16*kl*(kq * ((c0+1)*kp + self.m3 * (c0*self.m_parent-self.m3))*(self.m3*(c0*self.m_parent+self.m3)-(c0+1)*kp)+c0*self.m3**2 * pq * (kp-self.m3**2))
            return -power(coupling*G_F*self.fM*self.ckm/sqrt(2), 2) * (numerator1 + numerator2)/denom

        return super().__call__(m122, m232)

    def MIB9(self, g, m122, m232, mV, M, ml, a, b, d, f_pi=130, Gf=1.17e-11, F=1):
        """
        Dark photon for Eq.16 in hep-ph/0111249v1, see the mathematica notebook for the derivation
        g: coupling constant
        a, b, d: free parameters
        F: form factor default to 1
        everything else is the same in the notebook
        """

        res = 1/2 * mV**-2 *(b**2 * (M**2 + -1 * m122 + -1 * m232)**3 * (m232 + -1 * mV**2) + 2 * b * d * ml**2 * (m232 + -1 * mV**2)**3 + 4 * b * (a
            * m122 + (a + 2 * d * m122) * ml**2) * mV**2 * (-1 * m232 + mV**2) + -4 * (m122 + -1 * ml**2) * mV**2 * (-1 * a**2 + 2 * a * d * ml**2
            + d**2 * m122 * ml**2 + (-1 * b **2 * m122 + F**2 * (m122 + ml**2)) * mV**2)
            + (m232 + -1 * mV**2)**2 * (4 * a * d * ml**2 + d**2 * ml**2 * (m122 + -1 * ml**2) + (b**2 * (-1 * m122 + ml**2)
            + 2 * F **2 * (m122 + ml**2)) * mV**2) + 2 * (-1 * M**2 + m122 + m232)**2 * (b * (2 * a + d * ml**2)
            * (m232 + -1 * mV**2) + b**2 * (m232 + -1 * mV**2)**2 + 1 / 2 * (m122 + -1 * ml**2) * (d**2 * ml**2 + -1 *
            (b**2 + -2 * F**2) * mV**2)) + (M**2 + -1 * m122 + -1 * m232) * (4 * a * b * (-1 * m122 + ml**2) * mV**2
            + 4 * b * (a + d * ml **2) * (m232 + -1 * mV**2)**2 + b**2 * (m232 + -1 * mV**2)**3 + 2 * (m232 + -1
            * mV**2) * (2 * a**2 + 2 * a * d * ml**2 + d**2 * ml**2 * (m122 + -1 * ml**2) + b **2 * (-3 * m122 + ml**2) * mV**2)))
        return g**2 * res * Gf**2 * f_pi**2



class M2Meson3BodyDecayLeptonic(MatrixElementDecay3):
    # Radiative decay M -> l(1) nu(2) X(3), X radiated from leptonic leg
    def __init__(self, mx, boson_rep="S", meson="pion", lepton_mass=M_E):
        param_dict = {
            "pion": [M_PI, V_UD, F_PI, PION_WIDTH],
            "kaon": [M_K, V_US, F_K, KAON_WIDTH]
        }
        decay_params = param_dict[meson]
        m_meson = decay_params[0]
        self.ckm = decay_params[1]
        self.fM = decay_params[2]
        self.total_width = decay_params[3]
        super().__init__(m_meson, lepton_mass, 0.0, mx)
        # 0: parent meson
        # 1: lepton
        # 2: neutrino
        # 3: dark boson
        self.boson_rep = boson_rep

    def __call__(self, m122, m232, coupling=1.0):
        if self.boson_rep == "S":
            pass
        if self.boson_rep == "P":
            pass
        if self.boson_rep == "V":
            q2 = self.m_parent**2 - 2*self.m_parent*(m122 + m232 - self.m1**2 - self.m3**2)/(2*self.m_parent)

            prefactor = 8*power(G_F*self.fM*self.ckm/(q2 - self.m1**2)/self.m3, 2)

            lp, pq, kp, lq, kl, kq = super().get_sp_from_dalitz(m122, m232)

            cr = coupling
            cl = coupling

            # Dmu(self.m_parent/kl)*
            return -prefactor * ((power(cr*self.m_parent*self.m1,2) - power(cl*q2,2)) * (lq*self.m3**2 + 2*lp*pq) \
                - 2*cr*self.m1**2 * kq * (cr*self.m3**2 * kl + 2*cr*kp*lp - 3*cl*q2*self.m3**2))

        return super().__call__(m122, m232)




class M2MesonToGammaDarkPhoton(MatrixElementDecay2):
    """
    Decay of a neutral pseudoscalar meson to gamma + dark vector boson
    """
    def __init__(self, maprime, meson_mass=M_PI0):
        super().__init__(meson_mass, maprime, 0.0)

    def __call__(self, coupling=1.0):
        return 2 * (coupling)**2 * (1 - power(self.m1 / self.m_parent, 2))**3 / sqrt(4*pi*ALPHA)




class M2Pi0ToAGammaGamma(MatrixElementDecay3):
    # Radiative decay M^0 -> gamma(1) gamma(2) a(3) from alp-photon coupling
    def __init__(self, ma):
        self.fM = F_PI
        self.total_width = PI0_WIDTH
        super().__init__(M_PI0, 0.0, 0.0, ma)
        # 0: parent meson
        # 1: gamma
        # 2: gamma
        # 3: ALP

    def __call__(self, m122, m232, coupling=1.0):
        prefactor = 2 * power(ALPHA * coupling / pi / self.fM, 2)
        mpi02 = M_PI0**2
        mpi04 = M_PI0**4
        mpi06 = M_PI0**6

        Pq1, Pq2, Pk, q1q2, q1k, q2k = super().get_sp_from_dalitz(m122, m232)

        #return prefactor * (mpi02*power(q1q2, 3) + power(q1q2, 2) * (Pq1**2 + Pq2**2 - 3*mpi02*Pq1 - 3*mpi02*Pq2 + 3*mpi04) \
         #   + 4*power(Pq1*Pq2, 2) + 2*Pq1*Pq2*q1q2*(Pq1 + Pq2 - 3*mpi02))

        denom = (mpi02-2*(Pq1))**2 * (mpi02-2*(Pq2))**2
        num =  (mpi02*(mpi02-2*Pq1)*(mpi02-2*Pq2)*q1q2**3 \
            + 4*power(Pq1*Pq2, 2)*(Pq1+Pq2-mpi02)**2 \
                + q1q2**2 * (7*mpi04*Pq2**2 + (8*Pq2*(Pq2 - 2*mpi02) + 7*mpi04)*Pq1**2 \
                    - 9*mpi06*Pq2 - Pq1*(3*M_PI0**3 - 4*M_PI0*Pq2)**2 + 3*power(mpi04, 2)) \
                        + 2*Pq1*Pq2*q1q2*(Pq1 + Pq2 - mpi02)*(4*Pq1*(Pq2 - mpi02) - 4*mpi02*Pq2 + 3*mpi04))

        return prefactor * num / denom



# Sterile Neutrio Matrix Elements


class M2ElectronPositronToSterileNu(MatrixElement2):
    """
    e+ e- --> nu N, via s-channel gamma dipole portal
    """
    def __init__(self, mN):
        super().__init__(M_E, M_E, mN, 0)
        self.mN = mN

    def __call__(self, s, t, coupling_product=1.0):
        return - 4*pi*ALPHA * coupling_product**2 * (2 * M_E**4 + 2 * M_E**2 * (self.mN**2 - s - 2*t) \
            + self.mN**4 - self.mN**2 * (s + 2*t) + 2*t*(s + t)) / s




class M2Compton(MatrixElement2):
    """
    Compton scattering (gamma + e- -> a + e-)
    """
    def __init__(self, ma, z):
        super().__init__(0, M_E, ma, M_E)
        self.ma = ma
        self.z = z

    def __call__(self, s, t, coupling_product=1.0):
        u = 2*M_E**2 + self.ma**2 - s - t
        prefactor = self.z*4*pi*ALPHA * coupling_product**2
        Ms2 = prefactor * (M_E**4 + M_E**2 * (3*self.ma**2 - 2*s - t) + s*(s + t - self.ma**2))/power(M_E**2 - s, 2)
        Mu2 = prefactor * (M_E**4 + M_E**2 * (3*self.ma**2 - 2*s - t) + s*(s + t - self.ma**2))/power(M_E**2 - u, 2)
        MuMt = prefactor * (M_E**4 + M_E**2 * (3*self.ma**2 - 2*s - t) - (self.ma**2 - s)*(s + t))/((M_E**2 - s)*(M_E**2 - u))

        return Ms2 + Mu2 + 2*MuMt




class M2InverseCompton(MatrixElement2):
    """
    Compton scattering (a + e- -> gamma + e-)
    """
    def __init__(self, ma, z):
        super().__init__(ma, M_E, 0, M_E)
        self.ma = ma
        self.z = z

    def __call__(self, s, t, coupling_product=1.0):
        u = 2*M_E**2 + self.ma**2 - s - t
        prefactor = self.z*4*pi*ALPHA * coupling_product**2
        Ms2 = prefactor * (M_E**4 + M_E**2 * (3*self.ma**2 - 2*s - t) + s*(s + t - self.ma**2))/power(M_E**2 - s, 2)
        Mu2 = prefactor * (M_E**4 + M_E**2 * (3*self.ma**2 - 2*s - t) + s*(s + t - self.ma**2))/power(M_E**2 - u, 2)
        MuMt = prefactor * (M_E**4 + M_E**2 * (3*self.ma**2 - 2*s - t) - (self.ma**2 - s)*(s + t))/((M_E**2 - s)*(M_E**2 - u))

        return 2*(Ms2 + Mu2 + 2*MuMt)




class M2InversePrimakoff(MatrixElement2):
    """
    Inverse Primakoff scattering (a + N -> gamma + N)
    """
    def __init__(self, ma, mN, z):
        super().__init__(ma, mN, 0, mN)
        self.mN = mN
        self.ma = ma
        self.ff2 = AtomicElasticFF(z)

    def __call__(self, s, t, coupling_product=1.0):
        prefactor = 2 * coupling_product**2
        propagator = power(t, 2)
        numerator = -t*(2*self.mN**2 * (self.ma**2 - 2*s - t) + 2*self.mN**4 - 2*self.ma**2 * (s + t) + self.ma**4 + 2*s**2 + 2*s*t + t**2)
        return self.ff2(np.sqrt(abs(t))) * prefactor * numerator / propagator




class M2Primakoff(MatrixElement2):
    """
    Primakoff scattering (gamma + N -> a + N)
    """
    def __init__(self, ma, mN, z):
        super().__init__(0, mN, ma, mN)
        self.mN = mN
        self.ma = ma
        self.ff2 = AtomicElasticFF(z)

    def __call__(self, s, t, coupling_product=1.0):
        prefactor = coupling_product**2
        propagator = power(t, 2)
        numerator = -t*(2*self.mN**2 * (self.ma**2 - 2*s - t) + 2*self.mN**4 - 2*self.ma**2 * (s + t) + self.ma**4 + 2*s**2 + 2*s*t + t**2)
        return self.ff2(np.sqrt(abs(t))) * prefactor * numerator / propagator




class M2AssociatedProduction(MatrixElement2):
    """
    Associated ALP production (e+ + e- -> a + gamma)
    """
    def __init__(self, ma):
        super().__init__(M_E, M_E, ma, 0.0)
        self.ma = ma
    
    def __call__(self, s, t, coupling_product=1.0):
        u_prop = (M_E**2 + self.ma**2 - s - t)
        t_prop = (M_E**2 - t)
        tmast = t * (-self.ma**2 + s + t)

        Mt2 = -4*((-M_E**2 * (s + self.ma**2)) + 3*M_E**4 + tmast)/t_prop**2
        Mu2 = -4*((M_E**2 * (self.ma**2 - 3*s - 4*t)) + 7*M_E**4 + tmast)/u_prop**2
        MtMu = 4*((M_E**2 * (s - 2*t)) - 3*M_E**4 + tmast)/(u_prop*t_prop)

        M2 = Mt2 + Mu2 + 2*MtMu

        prefactor = (4*pi*ALPHA) * coupling_product**2
        
        return prefactor * M2




class M2NeutronAxionBrem(MatrixElement2to3):
    def __init__(self, ma, mNucleus, n, z):
        super().__init__(M_N, mNucleus, ma, mNucleus, M_N)
        self.ff2 = NuclearHelmFF(n, z)
    
    def __call__(self, s, s2, t1, s1, t2, fa=1.0):
        ma = self.m1
        m = self.ma
        M = self.mb
        q2_transfer = 2*M**2 + m**2 - s2 + t1 - t2
        prefactor = ((0.495*self.ma)/fa)**2 * self.ff2(sqrt(abs(q2_transfer)))
        return prefactor*(1/((-m**4-2*M**4+M_PI0**2+s2-t1+t2)**2))*4*(m**4+2*M**2-s2+t1-t2)*((2*m**6+m**4*(2*M**4-s2-t2-2)+m**2*(-3*M**4-2*ma**4-s+s1+2*s2+t2)+M**4*(t1-2*ma**4)+ma**4*s2+ma**4*t2+s*t1-s1*t1-s2*t1)/(m**2-t1)**2+(3*m**8-2*m**6+m**4*(5*M**4-3*ma**4-3*s2+3*t1-2*t2-4)-2*m**2*(5*M**4+ma**4+s-2*s1-3*s2-t2)-M**8+M**4*(-3*ma**4+s+s1+2*t1)+t2*(ma**4-s+s2)+2*ma**4*s2+s*t1-s1*s2-s1*t1-2*s2*t1)/((m**2-t1)*(-3*m**4+m**2-3*M**4+ma**4+s-s1+s2-2*t1+2*t2))+(6*m**8-8*m**6+m**4*(4*M**4-3*ma**4-2*s+5*s1-2*s2+3*t1-5*t2-2)+m**2*(-7*M**4+s-s1+4*s2+t2)-M**8+M**4*(-ma**4+s+t1)+s2*(ma**4-s1-t1+t2))/(-3*m**4+m**2-3*M**4+ma**4+s-s1+s2-2*t1+2*t2)**2)





class M2AxionPairProduction(MatrixElement2to3):
    def __init__(self, ma, mN, n, z, ml=M_E):
        super().__init__(ma, mN, ml, ml, mN)
        self.ff2 = AtomicPlusNuclearFF(n, z)
    
    def __call__(self, s, s2, t1, s1, t2, gae=1.0):
        ma = self.ma
        m = self.m1
        M = self.mb
        prefactor = self.ff2(sqrt(abs(t2))) * power(gae * 4*pi*ALPHA, 2.0) * power(t2**2 * (m**2-t1)**2 * (m**2+ma**2-s1-t1+t2)**2, -1.0)
        return prefactor * 4*(m**4*(2*M**4*(ma**2+t2)+2*M**2*(3*ma**4+ma**2*(3*t2-2*(s+s1))-t2*(2*s+s1)+s1**2)+2*(ma**3-ma*s)**2+t2*(ma**4+2*ma**2*(s1-s)+2*s**2-2*s*s1+s1**2)+2*t2**2*(s-s1)+t2**3)+m**2*(2*M**4*(2*ma**4+ma**2*(3*t2-2*(s1+t1))+t2*(-s1-2*t1+t2))+2*M**2*(3*ma**6-ma**4*(2*s+5*s1+2*s2+4*t1-5*t2)+ma**2*(2*s*(s1+2*t1-2*t2)+3*s1**2+2*s1*(s2+t1)-5*s1*t2+2*t2*(-s2-2*t1+t2))-(s1**2-t2*(2*s+s1))*(s1+2*t1-t2))+ma**6*(t2-4*s2)+ma**4*(4*s*(s2+t1)+s1*(4*s2-4*t1-t2)+t2*(-4*s2+2*t1+3*t2))+ma**2*(2*s**2*(t2-2*t1)-2*s*s1*(2*s2-2*t1+t2)+2*s*t2*(2*s2+t2)+s1**2*t2-4*s1*t2*(t1+t2)+3*t2**3)-t2*(2*s**2+2*s*(t2-s1)+(s1-t2)**2)*(s1+2*t1-t2))+2*M**4*(ma**2-s1-t1+t2)*(ma**4-ma**2*(s1+t1-t2)-t1*t2)-2*M**2*(t2**2*(2*ma**2*s2-t1*(2*s+s1))+(ma**2-s1-t1)*(ma**4*(2*s2+t1)-2*ma**2*(s*t1+s1*s2)+s1**2*t1)+t2*(ma**4*(4*s2+t1)-ma**2*(t1*(4*s+2*s2+t1)+s1*(4*s2+t1))+t1*(2*s*(s1+t1)+s1*(2*s1+t1))))+2*ma**2*(ma**2*s2-s*t1+s1*(t1-s2))**2+t2**4*(ma**2-t1)+t2*(2*ma**2*s2*(ma**2-s1)*(ma**2-s1+2*s2)+t1*(-4*ma**2*s2*(ma**2+s-2*s1)-(ma**2-s1)*(ma**4+2*s**2-2*s*s1+s1**2))+t1**2*(ma**4-2*s1*(ma**2+s)+2*ma**2*s+2*s**2+s1**2))+t2**3*(2*ma**4+ma**2*(-2*s1+2*s2-3*t1)+t1*(-2*s+3*s1+t1))+t2**2*(ma**6+ma**4*(-2*s1+4*s2-3*t1)+ma**2*(-2*s*t1+s1**2+4*s1*(t1-s2)+2*(s2-t1)**2)+t1*(-2*s**2+2*s*(2*s1+t1)-s1*(3*s1+2*t1))))