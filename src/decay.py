# Decay widths and lifetimes
# All lifetimes in s
# All decay widths in MeV

from constants import *
import numpy as np

# a -> gamma gamma
# g_agamma in MeV^-1
def W_gg(g_agamma, ma):
    return g_agamma**2 * ma**3 / (64*kPI)

# a -> e+ e-
def W_ee(g_ae, ma):
    return g_ae**2 * ma * np.sqrt(1 - (2 * kME / ma)**2) / (8 * kPI)

# Get the lifetime in the rest frame in s
def Tau(width):
    pass

# Get the lifetime in the lab frame in s
def Tau_lab(width, va):
    pass

# Probability that the ALP will survive a distance l from production site
def prob_surv(width, va, l):
    pass

# Probability that the ALP will decay within a region (l, l + dl)
def prob_decay(width, va, l, dl):
    pass


