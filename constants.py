# Physical constants
# Default values are in s, cm, MeV
# Alternate values provided for common constants

# Set the base units
GeV = 1.0e3
MeV = 1.0
keV = 1e-3
CM = 1.0
ang = 1e-8
sec = 1.0

# Dimensionless constants
ALPHA = 1/137
ALPHA_STRONG = 0.3068
E_QED = 0.30282212087
kPI = 3.14159

# Fundamental constants
C_LIGHT  = 2.99792458e10 * (CM / sec)  # Speed of light
HBAR = 6.58212e-22 * MeV * sec  # Reduced Planck's Constant (hbar)
HBARC = 1.97236e-11 * MeV * CM  # hbar * c
HC = 1.239e-10 * MeV * CM  # h c
KB = 8.617333262145e-11 # Boltzmann's constant in MeV K^-1
MU0 = 1.25663706212e-6  # Permeability of vacuum in kg / (A * s^2)
CHARGE_COULOMBS = 1.602176634e-19  # elementary charge in coulombs

# Scientific Constants
AVOGADRO = 6.022e23

# Particle Masses
M_E = 0.511 * MeV  # electron
M_P = 938.0 * MeV  # proton
M_MU = 105.658369 * MeV  # muon
M_TAU = 1.77699e3 * MeV  # tau
M_U = 2.2 * MeV  # up quark
M_D = 4.7 * MeV  # down quark
M_C = 1.280e3 * MeV  # charm quark
M_BOTTOM = 4.183 * GeV
M_TOP = 172.56 * GeV
M_PI = 139.57 * MeV  # pi+ / pi-
M_PI0 = 134.98 * MeV  # pi0
M_ETA = 547.862 * MeV  # eta
M_ETA_PRIME = 957.78 * MeV # eta prime
M_D_MESON = 1869.62 * MeV  # D^+/- meson
M_DS_MESON = 1968.47 * MeV  # D_s^+/- meson
M_RHO0 = 775.11 * MeV  # rho^0 and rho^+/- mesons
M_RHO_PLUS = 775.26 * MeV  # rho^0 and rho^+/- mesons
M_OMEGA = 782.66 * MeV  # omega vector meson mass
M_PHI = 1019.461 * MeV  # phi vector meson mass
M_N = 939.56 * MeV  # neutron
M_A = 1014 * MeV  # axial mass
M_K = 493.677  # charged Kaon
M_KLONG = 493.611  # charged Kaon
M_KSTAR = 891.66  # charged vector Kstar meson
M_W = 80.377 * GeV  # W boson
M_Z = 91.1876 * GeV  # Z boson

# conversion between units
METER_BY_MEV = 6.58212e-22 * 2.998e8  # MeV*m
MEV_PER_KG = 5.6095887e29  # MeV/kg
MEV_PER_HZ = 6.58e-22  # 1 s^-1 = 6.58e-22 MeV
CM_PER_ANG = 1e-8  # cm to angstroms
KEV_CM = HBARC / keV / ang
MEV2_CM2 = (METER_BY_MEV * 100)**2
S_PER_DAY = 3600*24
HBARC_KEV_ANG = 1.97

### SM parameters

# Electroweak physics
SSW = 0.2312  # Sine-squared of the Weinberg angle at Z mass
G_F = 1.16638e-11  # MeV^-2
CABIBBO = 0.9743  # Cosine of the Cabibbo angle
R_E = 2.81794e-13  # classical electron radius in cm

# CKM (absolute values)
V_UD = 0.97373  # ckm u-d
V_US = 0.2243  # ckm u-s
V_UB = 0.00382  # ckm u-b
V_CD = 0.221  # ckm c-d
V_CS = 0.975  # ckm c-s
V_CB = 0.0408  # ckm c-b
V_TD = 0.0086  # ckm t-d
V_TS = 0.0415  # ckm t-s
V_TB = 1.014  # ckm t-b

CKM_MATRIX = [[V_UD, V_US, V_UB], [V_CD, V_CS, V_CB], [V_TD, V_TS, V_TB]]

# Decay constants: mesons
F_PI = 130.2 * MeV  # pion decay constant
F_K = 155.7 * MeV  # kaon decay constant
F_D = 212.0 * MeV  # D meson decay constant
F_DS = 249.0 * MeV  # D_s meson decay constant
F_RHO = 0.171e6 * MeV*MeV  # Rho vector meson decay constant
F_OMEGA = 0.155e6 * MeV*MeV  # Omega vector meson decay constant
F_PHI = 0.232e6 * MeV*MeV  # Phi vector meson decay constant
F_KSTAR = 0.178e6 * MeV*MeV  # Kstar vector meson decay constant
ETA_F_0 = 148.0 * MeV  # Eta^0 decay constant
ETA_F_8 = 165.0 * MeV  # Eta^8 decay constant
THETA_0 = -0.12043  # Eta^0 rotation angle to physical basis in rad
THETA_8 = -0.37001  # Eta^8 rotation angle to physical basis in rad

# Lifetimes
KAON_LIFETIME = 1.238e-8 * sec
KSHORT_LIFETIME = 8.954e-11 * sec
KLONG_LIFETIME = 5.116e-8 * sec
PION_LIFETIME = 2.6e-8 * sec
PI0_LIFETIME = 8.5e-17 * sec

# Total Widths in MeV
PION_WIDTH = 2.524e-14 * MeV
PI0_WIDTH = 1.0
KAON_WIDTH = 5.317e-14 * MeV
KLONG_WIDTH = 1.28655e-14 * MeV


# Crystal constants
Z_GE = 32
R0_GE = 0.53
VCELL_GE = 181
LATTICE_CONST_GE = 5.66

R0_CSI = 1.0


# Astronomical constants
L_SUN_EARTH = 15.13e12  # cm
R_SUN = 6.957e10  # cm