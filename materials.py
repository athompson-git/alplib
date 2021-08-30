# Class to hold material-specific constants, dimensions, and responses
# for beam targets and detectors, for example.

from .constants import *
from .fmath import *

import json
import pkg_resources


class Material:
    """
    detector class
    """
    def __init__(self, material_name, fiducial_mass=1.0, volume=1.0, density=1.0, efficiency=None):
        """
        initializing Material,
        it reads ./mat_params.json for material information,
        :param mat_type: name of the material
        all units in MeV, kg, cm, unless otherwise stated (e.g. lattice params)
        """
        self.mat_type = material_name
        self.efficiency = efficiency
        fpath = pkg_resources.resource_filename(__name__, 'data/mat_params.json')
        f = open(fpath, 'r')
        mat_file = json.load(f)
        f.close()
        if material_name in mat_file:
            mat_info = mat_file[material_name.lower()]
            self.iso = mat_info['iso']
            self.z = np.array(mat_info['z'])
            self.n = np.array(mat_info['n'])
            self.m = np.array(mat_info['m'])
            self.frac = np.array(mat_info['frac'])
            self.lattice_const = np.array([mat_info['lattice_const']])  # Angstroms
            self.cell_volume = np.array([mat_info['cell_volume']])  # Angstroms^3
            self.r0 = np.array([mat_info['atomic_radius']])  # Angstroms
            self.er_min = mat_info['er_min']
            self.er_max = mat_info['er_max']
            self.bg = mat_info['bg']
            self.bg_un = mat_info['bg_un']
            self.density = mat_info['density']  # g/cm^3
            self.fid_mass = fiducial_mass  # kg
            self.volume = volume  # cm^3
            self.ntargets = density * volume / (np.sum(self.m) / MEV_PER_KG / 1e-3)
        else:
            raise Exception("No such detector in mat_params.json.")
