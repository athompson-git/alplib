# Useful math and helper functions

import numpy as np
from numpy import log, log10, exp, pi, sqrt, power, \
    sin, cos, tan, arccos, arctan, arcsin, heaviside, dot, cross
from scipy.integrate import quad, dblquad
from scipy.special import exp1, erf
from scipy.stats import norm

import mpmath as mp
from mpmath import mpmathify, fsub
mp.dps = 15    


# Fast 1D monte carlo, regenerating random variates each time
def fastMC1D(func, a, b, n_samples):
    vars = np.random.uniform(a, b, n_samples)
    return (b-a)*np.sum(func(vars))/n_samples




# Fast 2d monte carlo
def fastMC2D(func, a, b, c, d, n_samples):
    pass




def lorentz_boost(momentum, v):
    """
    Lorentz boost momentum to a new frame with velocity v
    :param momentum: four vector
    :param v: velocity of new frame, 3-dimention
    :return: boosted momentum
    """
    n = v/np.sqrt(np.sum(v**2))
    beta = np.sqrt(np.sum(v**2))
    gamma = 1/np.sqrt(1-beta**2)
    mat = np.array([[gamma, -gamma*beta*n[0], -gamma*beta*n[1], -gamma*beta*n[2]],
                    [-gamma*beta*n[0], 1+(gamma-1)*n[0]*n[0], (gamma-1)*n[0]*n[1], (gamma-1)*n[0]*n[2]],
                    [-gamma*beta*n[1], (gamma-1)*n[1]*n[0], 1+(gamma-1)*n[1]*n[1], (gamma-1)*n[1]*n[2]],
                    [-gamma*beta*n[2], (gamma-1)*n[2]*n[0], (gamma-1)*n[2]*n[1], 1+(gamma-1)*n[2]*n[2]]])
    return mat @ momentum