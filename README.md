# Welcome to ALPlib.

### This is a python library for performing physics calculations for axion-like-particles (ALPs).

Author: Adrian Thompson
Contact: a.thompson@northwestern.edu


### Examples and usage instructions incoming.


# Required tools
* python >3.7
    * numpy
    * scipy
    * mpmath
    * multiprocessing



# Classes and Methods

## Constants and Conventions
* Global constants (SM parameters, unit conversions, etc.) are stored in `constants.py` and have the naming convention `GLOBAL_CONSTANT_NAME`
* All units in alplib are in MeV, cm, kg, and s by default unless specifically stated, for example densities given in g/cm^2.

## The Material class
The `Material` class is a container for the physical constants and parameters pertaining to the materials used in experimental beam targets and detectors.
There are a number of material parameters stored in a JSON dictionary in `data/mat_params.json`, named according to the chemical name of the material, e.g. 'Ar' or 
'CsI'. One would initialize a detector or beam target, for instance, in the following way;
```
target_DUNE = Material('C')
det_DUNE = Material('Ar')
```
Further specifications for the volumes of the target/detector can also be specified for each instance that you may be interested in. The optional parameters
`fiducial_mass` (in kg)
```
det_DUNE = Material('Ar', fiducial_mass=50000)
```

All the material properties are set in `data/mat_params.json` with a JSON format for each entry; for example, in the case of cesium iodide we have
```
"CsI":
    {
      "iso": 2,
      "z": [55, 53],
      "n": [78, 74],
      "frac": [0.5, 0.5],
      "lattice_const": 4.503,
      "cell_volume": 22.82,
      "atomic_radius": [2.0, 2.0],
      "m": [123.8e3, 118.21e3],
      "density": 4.51,
      "er_min": 4.25e-3,
      "er_max":26e-3,
      "bg": 5e-3,
      "bg_un": 0.05
    },
```
You can extend `mat_params.json` with any material using this format.

## The AxionFlux super class
The class `fluxes.AxionFlux` is a super-class that can be inherited by any class that models a specific instance of a source of axion flux. It's most basic members are the `axion_energy` and `axion_flux` arrays, which together make a list of pairs of energies (in MeV) and event weights (in counts/second). Any class that inherits `AxionFlux` should populate `axion_energy` and `axion_flux` during its simulation routine - this flux class can then be passed to event generators (e.g. `fluxes.PhotonEventGenerator()`) to generate scattering or decay event weights at a detector module.

`AxionFlux` also has a default propagate method (which can be modified depending on the specific instance of the class inheriting `AxionFlux`) that looks at `AxionFlux.lifetime()` to propagate the flux weights to the detector. 



### Generators and Fluxes that inherit AxionFlux
There are several fluxes that inherit AxionFlux as a super class; for example, for isotropic fluxes we have `FluxPrimakoffIsotropic` (Primakoff ALP production from a photon flux in material), `FluxComptonIsotropic` (Compton ALP production from a photon flux in material), `FluxBremIsotropic` (ALP-bremsstrahlung from electron or positron fluxes in material), `FluxResonanceIsotropic` and `FluxPairAnnihilationIsotropic` (resonant and associated e+ e- annihilation into ALP production from a positron flux in material), `FluxNuclearIsotropic` (ALP production from nuclear decays), `FluxChargedMeson3BodyIsotropic` (ALP production from charged meson 3-body decay), and `FluxPi0Isotropic` (ALP production from pi0 decay at rest).

Each class will have its own initialization arguments in addition to those inherited from `AxionFlux`. For example, to simulate an ALP flux from Primakoff production of 100 MeV gammas in a tungsten target, we can use the following

```
wtarget = Material("W")

gammas = np.array([100.0, 1.0e12])  # 100 MeV, 1e12 photons / s

flux_p = FluxPrimakoffIsotropic(photon_flux=gammas, target=wtarget, det_dist=4.0, det_length=0.2,
                                det_area=0.04, axion_mass=0.1, axion_coupling=1e-5, n_samples=1000)

flux_p.simulate()  # simulate the production flux; flux_p.axion_flux is now populated with weights
flux_p.propagate()  # propagate gammas to detector, taking into account decays
```

One can then pass this simulated flux to an event generator class from `generators.py` to simulate the spectrum at the detector.

### Detection Classes and Event Rates
One can use ```PhotonEventGenerator``` and ```ElectronEventGenerator``` to simulate the detection channels of the ALPs in material.

For example, suppose we want to detect the ALPs we simulated in ```flux_p``` above. We may use

```
gen = PhotonEventGenerator(flux_p, Material("Ar"))
gen.decays(days_exposure=100, threshold=5.0)
```
This will calculate the weights for decays a -> gamma gamma coming from the flux per second into the detector in ```flux_p```, normalized to 100 days of exposure of a liquid argon detector with a threshold of 5 MeV. To access the individual weights per ALP in the flux, use
```
weights = gen.decay_weights
```
Alternatively, one can perform a 2-body decay monte carlo to obtain the Lorentz vectors of each decay photon, and the event-by-event weights, like so:
```
p41, p42, wgt = gen.simulate_decay_4vectors(days_exposure=100, n_samples=10000)

```

## Production and Detection Cross Sections

## MatrixElement and Monte Carlo Methods
The super class `MatrixElement2` and its inheritors offers a way to embed any 2->2 scattering process 1 2 -> 3 4. One simply needs to input the masses `m1`, `m2`, `m3`, `m4`, and the `__call__` method will return the squared matrix element as a function of the Mandelstam variables `s` and `t`. Below we outline the monte carlo simulation algorithm for 2-to-2 scattering as an example;

![](/documentation/alplib_mc_notes1.png)

![](/documentation/alplib_mc_notes2.png)

As an example, in `generators.py` we call the class `Scatter2to2MC` from `cross_section_mc.py`. Generating samples should look like this;
```
mc.lv_p1 = LorentzVector(Ea0, 0.0, 0.0, np.sqrt(Ea0**2 - self.mx**2))
mc.lv_p2 = LorentzVector(self.det_m, 0.0, 0.0, 0.0)
mc.scatter_sim()

cosines, dsdcos = mc.get_cosine_lab_weights()
e3, dsde = mc.get_e3_lab_weights()
```
where we have made use of the `LorentzVector` class.

## Decay Modes

## Crystal Scattering

# Examples

### (1) NA64 Flux from axion-bremsstrahlung

First we need the `FluxBremIsotropic` class from `alplib.fluxes`, and if we are interested in looking for visible energy events in a detector, we also need classes from `generators.py` like `ElectronEventGenerator`.

```
from alplib.fluxes import FluxBremIsotropic
from alplib.generators import ElectronEventGenerator
```

It is usually a good idea to define a set of constants for our setup, so I aggregate all the experimental parameters I want to assume for NA64 (and you may choose your own naming convention):

```
beam_energy = 1e5  # 100 GeV
eot = 2.84e11
days = 365
na64_dump = Material("W")
na64_ecal = Material("Pb")
ecal_length = 1.0  # meters
hcal_length = 6.5  # meters
hcal_area = 0.6*0.6  # 60cm across
ecal_dist = 43.3  # meters: beam d
ecal_area = 0.23  # ecal area in m^2; 43cm across
dump_length = 2.0  # meters
ecal_thresh = 1.0e3  # check
na64_e_flux = np.array([[beam_energy, eot/S_PER_DAY/days]])
det_am = na64_ecal.m[0] # mass of target atom in MeV
ecal_n = 100.0 * MEV_PER_KG / det_am
```

Now we can write a function to get the flux energies and weights with these inputs:

```
def get_flux(ma, g, use_loop=False):
    flux_brem = FluxBremIsotropic(na64_e_flux, target=na64_dump, det_dist=ecal_dist, det_length=ecal_length,
                                    det_area=ecal_area, target_length=dump_length, axion_mass=ma,
                                    axion_coupling=g, loop_decay=use_loop, is_isotropic=False, n_samples=10000)
    
    flux_brem.simulate()
    flux_brem.propagate()

    return np.array(flux_brem.axion_energy), np.array(flux_brem.axion_flux)
```
Here the `use_loop` option can account for the case where we want our ALP to include an effective coupling to photons through an electron loop. If we set `use_loop=True` this would permit decays to 2 photons for masses below twice the electron mass (albeit at 1-loop, this is relatively suppressed).

we can also write a `get_events()` function that takes care of propagating the flux to a detector:

```
def get_events(ma, g, use_loop=False):
    flux_brem = FluxBremIsotropic(na64_e_flux, target=na64_dump, det_dist=ecal_dist, det_length=ecal_length,
                                    det_area=ecal_area, target_length=dump_length, axion_mass=ma,
                                    axion_coupling=g, loop_decay=use_loop, is_isotropic=False, n_samples=10000)
    
    flux_brem.simulate()
    flux_brem.propagate()
    
    events_gen = ElectronEventGenerator(flux_brem, na64_ecal)
    events_gen.compton(ge=g, ma=ma, ntargets=ecal_n, days_exposure=days, threshold=ecal_thresh)
    events_gen.decays(days_exposure=days, threshold=ecal_thresh)
    return events_gen.axion_energy, events_gen.decay_weights + events_gen.scatter_weights
```

In both `get_events()` and `get_flux()` functions we are returning numpy arrays of the monte carlo energies and weights, which we can then pass into a histogram object like pyplot's `hist()`. For example,


```
ma = 0.5
g = 1e-5
energies, flux_wgts = get_flux(ma, g, use_loop=True)
plt.hist(1e-3*energies, weights=flux_wgts, histtype='step', bins=100, label="lives")
plt.xlabel(r"$E$ [GeV]")
plt.yscale('log')
plt.show()
```
