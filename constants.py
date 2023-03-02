# Physical constants
# Default values are in s, cm, MeV
# Alternate values provided for common constants

# Set the base units
MeV = 1.0
keV = 1e-3
cm = 1.0
ang = 1e-8
sec = 1.0

# Dimensionless constants
ALPHA = 1/137
kPI = 3.14159

# Fundamental constants
C_LIGHT  = 2.99792458e10 * (cm / sec)  # Speed of light
HBAR = 6.58212e-22 * MeV * sec  # Reduced Planck's Constant (hbar)
HBARC = 1.97236e-11 * MeV * cm  # hbar * c
HC = 1.239e-10 * MeV * cm  # h c
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
M_PI = 139.57 * MeV  # pi+ / pi-
M_PI0 = 134.98 * MeV  # pi0
M_ETA = 547.862 * MeV  # eta
M_N = 939.56 * MeV  # neutron
M_A = 1014 * MeV  # axial mass
M_K = 493.677  # charged Kaon

# conversion between units
METER_BY_MEV = 6.58212e-22 * 2.998e8  # MeV*m
MEV_PER_KG = 5.6095887e29  # MeV/kg
MEV_PER_HZ = 6.58e-22  # 1 s^-1 = 6.58e-22 MeV
CM_PER_ANG = 1e-8  # cm to angstroms
KEV_CM = HBARC / keV / ang
MEV2_CM2 = (METER_BY_MEV * 100)**2
S_PER_DAY = 3600*24
HBARC_KEV_ANG = 1.97

# SM parameters
SSW = 0.2312  # Sine-squared of the Weinberg angle
G_F = 1.16638e-11  # MeV^-2
CABIBBO = 0.9743  # Cosine of the Cabibbo angle
R_E = 2.81794e-13  # classical electron radius in cm
V_UD = 0.9737  # ckm ud
V_US = 0.2245  # ckm us
F_PI = 130.2 * MeV  # pion decay constant
F_K = 155.7 * MeV  # kaon decay constant

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


# Crystal constants
Z_GE = 32
R0_GE = 0.53
VCELL_GE = 181
LATTICE_CONST_GE = 5.66

R0_CSI = 1.0


# Astronomical constants
L_SUN_EARTH = 15.13e12  # cm
R_SUN = 6.957e10  # cm