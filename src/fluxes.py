# ALP Fluxes, DM Fluxes
# All fluxes in cm^-2 s^-1 or cm^-2 s^-1 MeV^-1

import numpy as np
from numpy import exp, log, log10, sqrt, pi, heaviside
from scipy.special import erf

from constants import *


##### DARK MATTER FLUXES #####
rho_chi = 0.4e6 #(* keV / cm^3 *)
vesc = 544.0e6
v0 = 220.0e6
ve = 244.0e6
nesc = erf(vesc/v0) - 2*(vesc/v0) * exp(-(vesc/v0)**2) * sqrt(pi)

def fv(v):  # Velocity profile ( v ~ [0,1] )
    return (1.0 / (nesc * np.power(pi,3/2) * v0**3)) * exp(-((v + ve)**2 / v0**2))

def DMFlux(v, m):  # DM differential flux as a function of v~[0,1] and the DM mass
    return heaviside(v + v0 - vesc) * 4*pi*C_LIGHT*(rho_chi / m) * (C_LIGHT*v)**3 * fv(C_LIGHT*v)


##### NUCLEAR COUPLING FLUXES #####

def Fe57Flux(gp):  # Monoenergetic flux at 14.4 keV from the Sun
    return (4.56e23) * gp**2


##### ELECTRON COUPLING FLUXES #####


##### PHOTON COUPLING #####