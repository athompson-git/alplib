# Efficiency class

from .fmath import *
from scipy.interpolate import interp1d

EFF_TYPES = ['uniform', 'spline']

class Efficiency:
    def __init__(self, func_type='uniform', control_points=None):
        if func_type not in EFF_TYPES:
            raise Exception('Efficiency function type is not known; select from ', EFF_TYPES)
        self.func_type = func_type
        self.eff_spline = interp1d(control_points[:,0], control_points[:,1],
            kind='cubic', bounds_error=False, fill_value='extrapolate') if func_type=='spline' else None
    
    def __call__(self, energy):
        if self.func_type == 'uniform':
            return 1.0
        elif self.func_type == 'spline':
            return np.heaviside(self.eff_spline(energy), 0.0) * self.eff_spline(energy)