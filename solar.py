# Classes and methods to get solar radiation angle of incidence on a surface
# Based on "Solar Position Algorithm for Solar Radiation Applications"
# by Ibrahim Reda and Afshin Andreas

from alplib.constants import *
from alplib.fmath import *


# 3.1: Time scales ~~~

def tt_from_tai(tai):
    # terrestrial time from international atomic time
    return tai + 32.184

def jd(y, m, d, b=0.0):
    # Julian Day
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5

def jde(jd, deltaT):
    # Julian Ephemeris Day
    return jd + deltaT/86400

def jc(jd):
    # Julian Century
    return (jd - 2451545)/36525

def jce(jde):
    # Julian Ephemeris Century
    return (jde - 2451545)/36525

def jme(jce):
    # Julian Ephemeris Millenium
    return jce / 10




# 3.2: Earth Heliocentric Longitude, Latitude, Radius Vector

l0_data = np.genfromtxt("data/solar/periodic_earth_terms_L0.txt", skip_header=1)
l1_data = np.genfromtxt("data/solar/periodic_earth_terms_L1.txt", skip_header=1)
l2_data = np.genfromtxt("data/solar/periodic_earth_terms_L2.txt", skip_header=1)
l3_data = np.genfromtxt("data/solar/periodic_earth_terms_L3.txt", skip_header=1)
l4_data = np.genfromtxt("data/solar/periodic_earth_terms_L4.txt", skip_header=1)
l5_data = np.genfromtxt("data/solar/periodic_earth_terms_L5.txt", skip_header=1)

def earth_hc_long(jme):
    # Returns Earth heliocentric longitude in degrees (0.0, 360.0)
    # l0i = ai * cos(bi + ci JME)
    # l0 = sum l0i
    l0 = np.sum(np.array([l0_data[i, 1] * np.cos(l0_data[i, 2] + l0_data[i, 3] * jme) \
        for i in range(l0_data.shape[0])]))
    l1 = np.sum(np.array([l1_data[i, 1] * np.cos(l1_data[i, 2] + l1_data[i, 3] * jme) \
        for i in range(l1_data.shape[0])]))
    l2 = np.sum(np.array([l2_data[i, 1] * np.cos(l2_data[i, 2] + l2_data[i, 3] * jme) \
        for i in range(l2_data.shape[0])]))
    l3 = np.sum(np.array([l3_data[i, 1] * np.cos(l3_data[i, 2] + l3_data[i, 3] * jme) \
        for i in range(l3_data.shape[0])]))
    l4 = np.sum(np.array([l4_data[i, 1] * np.cos(l4_data[i, 2] + l4_data[i, 3] * jme) \
        for i in range(l4_data.shape[0])]))
    l5 = np.sum(np.array([l5_data[i, 1] * np.cos(l5_data[i, 2] + l5_data[i, 3] * jme) \
        for i in range(l5_data.shape[0])]))
    
    longitude = (180.0/np.pi) * 1e-8 * \
        (l0 + l1*jme + l2*power(jme,2) + l3*power(jme,3) \
            + l4*power(jme,4) + l5*power(jme,5))
    
    if longitude > 0.0:
        return longitude % 360.0
    elif longitude < 0.0:
        return 360.0 - longitude % 360.0 
    return 0.0

b0_data = np.genfromtxt("data/solar/periodic_earth_terms_B0.txt", skip_header=1)
b1_data = np.genfromtxt("data/solar/periodic_earth_terms_B1.txt", skip_header=1)

def earth_hc_lat(jme):
    # Returns Earth heliocentric latitude in degrees (0.0, 360.0)
    # b0i = ai * cos(bi + ci JME)
    b0 = np.sum(np.array([b0_data[i, 1] * np.cos(b0_data[i, 2] + b0_data[i, 3] * jme) \
        for i in range(b0_data.shape[0])]))
    b1 = np.sum(np.array([b1_data[i, 1] * np.cos(b1_data[i, 2] + b1_data[i, 3] * jme) \
        for i in range(b1_data.shape[0])]))
    
    latitude = (180.0/np.pi) * 1e-8 * (b0 + b1*jme)
    
    if latitude > 0.0:
        return latitude % 360.0
    elif latitude < 0.0:
        return 360.0 - latitude % 360.0 
    return 0.0

r0_data = np.genfromtxt("data/solar/periodic_earth_terms_R0.txt", skip_header=1)
r1_data = np.genfromtxt("data/solar/periodic_earth_terms_R1.txt", skip_header=1)
r2_data = np.genfromtxt("data/solar/periodic_earth_terms_R2.txt", skip_header=1)
r3_data = np.genfromtxt("data/solar/periodic_earth_terms_R3.txt", skip_header=1)
r4_data = np.genfromtxt("data/solar/periodic_earth_terms_R4.txt", skip_header=1)

def earth_hc_radius(jme):
    # Returns Earth heliocentric longitude in degrees (0.0, 360.0)
    # r0i = ai * cos(bi + ci JME)
    # r0 = sum r0i
    r0 = np.sum(np.array([r0_data[i, 1] * np.cos(r0_data[i, 2] + r0_data[i, 3] * jme) \
        for i in range(r0_data.shape[0])]))
    r1 = np.sum(np.array([r1_data[i, 1] * np.cos(r1_data[i, 2] + r1_data[i, 3] * jme) \
        for i in range(r1_data.shape[0])]))
    r2 = np.sum(np.array([r2_data[i, 1] * np.cos(r2_data[i, 2] + r2_data[i, 3] * jme) \
        for i in range(r2_data.shape[0])]))
    r3 = np.sum(np.array([r3_data[i, 1] * np.cos(r3_data[i, 2] + r3_data[i, 3] * jme) \
        for i in range(r3_data.shape[0])]))
    r4 = np.sum(np.array([r4_data[i, 1] * np.cos(r4_data[i, 2] + r4_data[i, 3] * jme) \
        for i in range(r4_data.shape[0])]))
    
    radius = (180.0/np.pi) * 1e-8 * \
        (r0 + r1*jme + r2*power(jme,2) + r3*power(jme,3) + r4*power(jme,4))
    
    if radius > 0.0:
        return radius % 360.0
    elif radius < 0.0:
        return 360.0 - radius % 360.0 
    return 0.0




# 3.3: Calculate the geocentric long and lat (theta, beta)
def theta_gc_long(jme):
    theta = earth_hc_long(jme) + 180.0
    if theta > 0.0:
        return theta % 360.0
    elif theta < 0.0:
        return 360.0 - theta % 360.0 
    return 0.0

def beta_gc_lat(jme):
    return -earth_hc_lat(jme)




# 3.4: Calculate the nutation in longitude and obliquity (DeltaPsi and DeltaEpsilon):

def x0(jce):
    # Mean elongation
    return 297.85036 + 445267.111480*jce - 0.0019142*power(jce,2) + power(jce,3)/189474

def x1(jce):
    # Mean anomaly sun (earth)
    return 357.52772 + 35999.050340*jce - 0.0001603*power(jce,2) - power(jce,3)/300000

def x2(jce):
    # mean anomaly of the moon
    return 134.96298 + 477198.867398*jce - 0.0086972*power(jce,2) + power(jce,3)/56250

def x3(jce):
    # moon's argument of latitude
    return 93.27191 + 483202.017538*jce - 0.0036825*power(jce,2) + power(jce,3)/327270

def x4(jce):
    # longitude of ascending node of moon's orbit on the eliptic, measured from mean equinox
    return 125.04452 + 1934.136261*jce - 0.0020708*power(jce,2) + power(jce,3)/450000




nuta_data = np.genfromtxt("data/solar/nutation_longitude_obliquity.txt", skip_header=1)
ai = nuta_data[:,5]
bi = nuta_data[:,6]
ci = nuta_data[:,7]
di = nuta_data[:,8]
yi0 = nuta_data[:,0]
yi1 = nuta_data[:,1]
yi2 = nuta_data[:,2]
yi3 = nuta_data[:,3]
yi4 = nuta_data[:,4]


def delta_psi_i(jce, i):
    # nutation longitude
    return (ai[i] + bi[i] * jce) * sin(x0(jce) * yi0[i] + x1(jce) * yi1[i] \
                                        + x2(jce) * yi2[i] + x3(jce) * yi3[i] + x4(jce) * yi4[i])

def delta_epsilon_i(jce, i):
    # nutation obliquity
    return (ci[i] + di[i] * jce) * cos(x0(jce) * yi0[i] + x1(jce) * yi1[i] \
                                        + x2(jce) * yi2[i] + x3(jce) * yi3[i] + x4(jce) * yi4[i])

def delta_psi(jce):
    return np.sum([delta_psi_i(jce, i) for i in range(63)]) / 36000000

def delta_epsilon(jce):
    return np.sum([delta_epsilon_i(jce, i) for i in range(63)]) / 36000000