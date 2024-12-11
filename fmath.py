# Useful math and helper functions

import numpy as np
from numpy import log, log10, exp, pi, sqrt, power, \
    sin, cos, tan, arccos, arctan, arcsin, arctan2, heaviside, dot, cross
from scipy.integrate import quad, dblquad
from scipy.special import exp1, erf, gamma
from scipy.stats import norm, chisquare, expon



def fastMC1D(func, a, b, n_samples, **kwargs):
    # Fast 1D monte carlo, regenerating random variates each time
    vars = np.random.uniform(a, b, n_samples)
    return (b-a)*np.sum(func(vars, kwargs))/n_samples




# Fast 2d monte carlo
def fastMC2D(func, a, b, c, d, n_samples):
    pass




def kallen_alplib(x, y, z):
    return x*x + y*y + z*z - 2*x*y - 2*y*z - 2*z*x