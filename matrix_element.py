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
        return self.ff2(np.sqrt(abs(t))) * coupling_product**2 * (2*self.mN**2 - t) * power((self.mZp**2 - t)/(2*(self.mphi**2 - t)),2) 




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
        return - self.ff2(np.sqrt(abs(t))) * coupling_product**2 * t * power((self.mZp**2 - t)/(2*(self.mphi**2 - t)),2)




class M2PairProduction:
    """
    a + N -> e+ e- N    ALP-driven pair production
    """
    def __init__(self, ma, mN, n, z):
        self.ma = ma
        self.mN = mN
        self.ff2 = NuclearHelmFF(n, z)
    
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




class M2Meson3BodyDecay(MatrixElementDecay3):
    # Radiative decay M -> l(1) nu(2) X(3), X radiated from hadronic meson current
    def __init__(self, mx, meson="pion", lepton_mass=M_E, interaction_model="vector_ib2"):
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