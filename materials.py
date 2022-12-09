# Class to hold material-specific constants, dimensions, and responses
# for beam targets and detectors, for example.

from .constants import *
from .fmath import *
from .efficiency import Efficiency

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
        :param mat_name: name of the material
        all units in MeV, kg, cm, unless otherwise stated (e.g. lattice params)
        """
        self.mat_name = material_name
        self.efficiency = efficiency
        fpath = pkg_resources.resource_filename(__name__, 'data/mat_params.json')
        f = open(fpath, 'r')
        mat_file = json.load(f)
        f.close()
        if material_name in mat_file:
            mat_info = mat_file[material_name]
            self.iso = mat_info['iso']
            self.z = np.array(mat_info['z'])
            self.n = np.array(mat_info['n'])
            self.m = np.array(mat_info['m'])
            self.frac = np.array(mat_info['frac'])
            self.lattice_const = np.array([mat_info['lattice_const']])  # Angstroms
            self.cell_volume = np.array([mat_info['cell_volume']])  # Angstroms^3
            self.r0 = np.array([mat_info['atomic_radius']])  # Angstroms
            self.density = mat_info['density']  # g/cm^3
            self.fid_mass = fiducial_mass  # kg
            self.volume = volume  # cm^3
            self.ntargets = density * volume / (np.dot(self.m, self.iso*self.frac) / MEV_PER_KG / 1e-3)
            self.ndensity = self.ntargets / self.volume
            self.rad_length = mat_info['rad_length']
        else:
            raise Exception("No such detector in mat_params.json.")




class DetectorGeometry:
    def __init__(self, distance_to_source, geometry="box", params=[1.0, 1.0, 1.0], n_samples=10000):
        if distance_to_source <= 0.0:
            raise Exception("Can't have the detector closer than 0.0 (distance_to_source < 0.0)")
        
        self.l0 = distance_to_source
        self.n_samples = n_samples
        self.u1 = np.random.uniform(0, 1, n_samples)
        self.u2 = np.random.uniform(0, 1, n_samples)
        self.u3 = np.random.uniform(0, 1, n_samples)
        self.geom = geometry
        self.params = params

        if geometry == "box":
            # params = [w, h, l]
            if len(params) != 3:
                raise Exception("params must be length 3 for box geometry (input [w, h, l])")
            self.u1 = params[0]*(self.u1 - 0.5)
            self.u2 = params[1]*(self.u2 - 0.5)
            self.u3 = params[2]*(self.u3 - 0.5)
        elif geometry == "sphere":
            # params = [r]
            if len(params) != 1:
                raise Exception("params must be length 1 for sphere geometry (input [radius])")
            self.u1 = params[0]*self.u1
            self.u2 = arccos(1 - 2*self.u2)
            self.u3 = 2*pi*self.u3
        elif geometry == "cylinder":
            # params = [r, h]
            if len(params) != 2:
                raise Exception("params must be length 2 for cylinder geometry (input [radius, height])")
            self.u1 = params[0]*self.u1
            self.u2 = 2*pi*self.u3
            self.u3 = params[1]*(self.u3 - 0.5)
        else:
            raise Exception("Geometry %s not found", geometry)
    
    def l_cart(self, x, y, z):
        return sqrt(x**2 + y**2 + (z - self.l0)**2)
    
    def l_cyl(self, r, theta, z):
        return sqrt(r**2 + self.l0**2 - 2*r*self.l0*cos(theta) + z**2)

    def l_sph(self, r, theta, phi):
        return sqrt(r**2 + self.l0**2 - 2*r*self.l0*sin(theta)*cos(phi))
    
    def integrate_box(self, f_int):
        # x: u1
        # y: u2
        # z: u3
        ls = self.l_cart(self.u1, self.u2, self.u3)  # w, h, l
        return (self.params[0]*self.params[1]*self.params[2])*np.sum(f_int(ls)) / self.n_samples

    def integrate_sphere(self, f_int):
        # r: u1
        # theta: u2
        # phi: u3
        ls = self.l_sph(self.u1, self.u2, self.u3)
        return (4*pi*self.params[0]**3 / 3) * np.sum(self.u1**2 * sin(self.u2) * f_int(ls)) / self.n_samples

    def integrate_cylinder(self, f_int):
        # r: u1
        # phi: u2
        # z: u3
        ls = self.l_cyl(self.u1, self.u2, self.u3)
        return (self.params[1]*pi*self.params[0]**2) * np.sum(self.u1 * f_int(ls)) / self.n_samples
    
    def integrate(self, f_int):
        if self.geom == "box":
            return self.integrate_box(f_int)
        elif self.geom == "sphere":
            return self.integrate_sphere(f_int)
        elif self.geom == "cylinder":
            return self.integrate_cylinder(f_int)



