# Efficiency class

from .fmath import *
from scipy.interpolate import interp1d

class Efficiency:
    """
    TODO: efficiency super class
    should be able to pass in a function or an array of points to interpolate between
    can take in a (N, 2) array with energies in the first column and efficiencies in the second column
    __call__ returns an efficiency function
    """
    def __init__(self, control_points=None, flat_efficiency=1.0):
        self.flat_eff = flat_efficiency
        if control_points is None:
            self.func_type = 'uniform'
        else:
            self.func_type = 'spline'
        self.control_points = control_points

    def __call__(self, energy):
        if self.func_type == 'uniform':
            return self.flat_eff
        elif self.func_type == 'spline':
            return np.clip(np.interp(energy, self.control_points[:,0], self.control_points[:,1]), 0.0, 1.0)




# Deprecated function from materials.py
"""
class Efficiency:
    #TODO: efficiency super class
    #should be able to pass in a function or an array of points to interpolate between
    #can take in a (N, 2) array with energies in the first column and efficiencies in the second column
    #__call__ returns an efficiency function
    def __init__(self, eff_data=None):
        self.eff_data = eff_data

    def __call__(self, energy):
        if self.eff_data is not None:
            return np.interp(energy, self.eff_data[:,0], self.eff_data[:,1])
        return 1.0
"""