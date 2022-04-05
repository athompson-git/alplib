# Classes and methods to get solar radiation angle of incidence on a surface
# Based on "Solar Position Algorithm for Solar Radiation Applications"
# by Ibrahim Reda and Afshin Andreas

from alplib.constants import *
from alplib.fmath import *


class SPA:
    """
    Solar Position Algorithm for Solar Radiation Applications
    """
    def __init__(self):
        self.r0_data = np.genfromtxt("data/solar/periodic_earth_terms_R0.txt", skip_header=1)
        self.r1_data = np.genfromtxt("data/solar/periodic_earth_terms_R1.txt", skip_header=1)
        self.r2_data = np.genfromtxt("data/solar/periodic_earth_terms_R2.txt", skip_header=1)
        self.r3_data = np.genfromtxt("data/solar/periodic_earth_terms_R3.txt", skip_header=1)
        self.r4_data = np.genfromtxt("data/solar/periodic_earth_terms_R4.txt", skip_header=1)

        self.b0_data = np.genfromtxt("data/solar/periodic_earth_terms_B0.txt", skip_header=1)
        self.b1_data = np.genfromtxt("data/solar/periodic_earth_terms_B1.txt", skip_header=1)

        self.l0_data = np.genfromtxt("data/solar/periodic_earth_terms_L0.txt", skip_header=1)
        self.l1_data = np.genfromtxt("data/solar/periodic_earth_terms_L1.txt", skip_header=1)
        self.l2_data = np.genfromtxt("data/solar/periodic_earth_terms_L2.txt", skip_header=1)
        self.l3_data = np.genfromtxt("data/solar/periodic_earth_terms_L3.txt", skip_header=1)
        self.l4_data = np.genfromtxt("data/solar/periodic_earth_terms_L4.txt", skip_header=1)
        self.l5_data = np.genfromtxt("data/solar/periodic_earth_terms_L5.txt", skip_header=1)

        nuta_data = np.genfromtxt("data/solar/nutation_longitude_obliquity.txt", skip_header=1)
        self.ai = nuta_data[:,5]
        self.bi = nuta_data[:,6]
        self.ci = nuta_data[:,7]
        self.di = nuta_data[:,8]
        self.yi0 = nuta_data[:,0]
        self.yi1 = nuta_data[:,1]
        self.yi2 = nuta_data[:,2]
        self.yi3 = nuta_data[:,3]
        self.yi4 = nuta_data[:,4]


    # 3.1: Time scales ~~~

    def tt_from_tai(self, tai):
        # terrestrial time from international atomic time
        return tai + 32.184

    def jd(self, y, m, d, b=0.0):
        # Julian Day
        return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5

    def jde(self, jd, deltaT):
        # Julian Ephemeris Day
        return jd + deltaT/86400

    def jc(self, jd):
        # Julian Century
        return (jd - 2451545)/36525

    def jce(self, jde):
        # Julian Ephemeris Century
        return (jde - 2451545)/36525

    def jme(self, jce):
        # Julian Ephemeris Millenium
        return jce / 10


    # 3.2: Earth Heliocentric Longitude, Latitude, Radius Vector

    def earth_hc_long(self, jme):
        # Returns Earth heliocentric longitude in degrees (0.0, 360.0)
        # l0i = ai * cos(bi + ci JME)
        # l0 = sum l0i
        l0 = np.sum(np.array([self.l0_data[i, 1] * np.cos(self.l0_data[i, 2] + self.l0_data[i, 3] * jme) \
            for i in range(self.l0_data.shape[0])]))
        l1 = np.sum(np.array([self.l1_data[i, 1] * np.cos(self.l1_data[i, 2] + self.l1_data[i, 3] * jme) \
            for i in range(self.l1_data.shape[0])]))
        l2 = np.sum(np.array([self.l2_data[i, 1] * np.cos(self.l2_data[i, 2] + self.l2_data[i, 3] * jme) \
            for i in range(self.l2_data.shape[0])]))
        l3 = np.sum(np.array([self.l3_data[i, 1] * np.cos(self.l3_data[i, 2] + self.l3_data[i, 3] * jme) \
            for i in range(self.l3_data.shape[0])]))
        l4 = np.sum(np.array([self.l4_data[i, 1] * np.cos(self.l4_data[i, 2] + self.l4_data[i, 3] * jme) \
            for i in range(self.l4_data.shape[0])]))
        l5 = np.sum(np.array([self.l5_data[i, 1] * np.cos(self.l5_data[i, 2] + self.l5_data[i, 3] * jme) \
            for i in range(self.l5_data.shape[0])]))
        
        longitude = (180.0/np.pi) * 1e-8 * \
            (l0 + l1*jme + l2*power(jme,2) + l3*power(jme,3) \
                + l4*power(jme,4) + l5*power(jme,5))
        
        if longitude > 0.0:
            return longitude % 360.0
        elif longitude < 0.0:
            return 360.0 - longitude % 360.0 
        return 0.0

    def earth_hc_lat(self, jme):
        # Returns Earth heliocentric latitude in degrees (0.0, 360.0)
        # b0i = ai * cos(bi + ci JME)
        b0 = np.sum(np.array([self.b0_data[i, 1] * np.cos(self.b0_data[i, 2] + self.b0_data[i, 3] * jme) \
            for i in range(self.b0_data.shape[0])]))
        b1 = np.sum(np.array([self.b1_data[i, 1] * np.cos(self.b1_data[i, 2] + self.b1_data[i, 3] * jme) \
            for i in range(self.b1_data.shape[0])]))
        
        latitude = (180.0/np.pi) * 1e-8 * (b0 + b1*jme)
        
        if latitude > 0.0:
            return latitude % 360.0
        elif latitude < 0.0:
            return 360.0 - latitude % 360.0 
        return 0.0

    def earth_hc_radius(self, jme):
        # Returns Earth heliocentric longitude in degrees (0.0, 360.0)
        # r0i = ai * cos(bi + ci JME)
        # r0 = sum r0i
        r0 = np.sum(np.array([self.r0_data[i, 1] * np.cos(self.r0_data[i, 2] + self.r0_data[i, 3] * jme) \
            for i in range(self.r0_data.shape[0])]))
        r1 = np.sum(np.array([self.r1_data[i, 1] * np.cos(self.r1_data[i, 2] + self.r1_data[i, 3] * jme) \
            for i in range(self.r1_data.shape[0])]))
        r2 = np.sum(np.array([self.r2_data[i, 1] * np.cos(self.r2_data[i, 2] + self.r2_data[i, 3] * jme) \
            for i in range(self.r2_data.shape[0])]))
        r3 = np.sum(np.array([self.r3_data[i, 1] * np.cos(self.r3_data[i, 2] + self.r3_data[i, 3] * jme) \
            for i in range(self.r3_data.shape[0])]))
        r4 = np.sum(np.array([self.r4_data[i, 1] * np.cos(self.r4_data[i, 2] + self.r4_data[i, 3] * jme) \
            for i in range(self.r4_data.shape[0])]))
        
        radius = (180.0/np.pi) * 1e-8 * \
            (r0 + r1*jme + r2*power(jme,2) + r3*power(jme,3) + r4*power(jme,4))
        
        if radius > 0.0:
            return radius % 360.0
        elif radius < 0.0:
            return 360.0 - radius % 360.0 
        return 0.0


    # 3.3: Calculate the geocentric long and lat (theta, beta)
    def theta_gc_long(self, jme):
        theta = self.earth_hc_long(jme) + 180.0
        if theta > 0.0:
            return theta % 360.0
        elif theta < 0.0:
            return 360.0 - theta % 360.0 
        return 0.0

    def beta_gc_lat(self, jme):
        return -self.earth_hc_lat(jme)


    # 3.4: Calculate the nutation in longitude and obliquity (DeltaPsi and DeltaEpsilon):

    def x0(self, jce):
        # Mean elongation
        return 297.85036 + 445267.111480*jce - 0.0019142*power(jce,2) + power(jce,3)/189474

    def x1(self, jce):
        # Mean anomaly sun (earth)
        return 357.52772 + 35999.050340*jce - 0.0001603*power(jce,2) - power(jce,3)/300000

    def x2(self, jce):
        # mean anomaly of the moon
        return 134.96298 + 477198.867398*jce - 0.0086972*power(jce,2) + power(jce,3)/56250

    def x3(self, jce):
        # moon's argument of latitude
        return 93.27191 + 483202.017538*jce - 0.0036825*power(jce,2) + power(jce,3)/327270

    def x4(self, jce):
        # longitude of ascending node of moon's orbit on the eliptic, measured from mean equinox
        return 125.04452 + 1934.136261*jce - 0.0020708*power(jce,2) + power(jce,3)/450000

    def delta_psi_i(self, jce, i):
        # nutation longitude
        return (self.ai[i] + self.bi[i]*jce) * sin(self.x0(jce) * self.yi0[i] + self.x1(jce) * self.yi1[i] \
                                                + self.x2(jce) * self.yi2[i] + self.x3(jce) * self.yi3[i] \
                                                + self.x4(jce) * self.yi4[i])

    def delta_epsilon_i(self, jce, i):
        # nutation obliquity
        return (self.ci[i] + self.di[i]*jce) * cos(self.x0(jce) * self.yi0[i] + self.x1(jce) * self.yi1[i] \
                                                    + self.x2(jce) * self.yi2[i] + self.x3(jce) * self.yi3[i] \
                                                    + self.x4(jce) * self.yi4[i])

    def delta_psi(self, jce):
        return np.sum([self.delta_psi_i(jce, i) for i in range(63)]) / 36000000

    def delta_epsilon(self, jce):
        return np.sum([self.delta_epsilon_i(jce, i) for i in range(63)]) / 36000000


    # 3.5: Calculate the true obliquity of the ecliptic, epsilon (in degrees): 

    def epsilon(self, jme):
        u = jme/10
        eps0 = 84381.448 - 4680.93*u - 1.55*power(u,2) + 1999.25*power(u,3) \
            - 51.38*power(u,4) - 249.67*power(u,5) - 39.05*power(u,6) \
                + 7.12*power(u,7) + 27.87*power(u,8) + 5.79*power(u,9) \
                    + 2.45*power(u,10)
        
        return (eps0/3600) + self.delta_epsilon(10*jme)


    # 3.6: Calculate the aberration correction, delta T (in degrees):

    def delta_tau(self, jme):
        return -2048.98 / (3600 * self.earth_hc_radius(jme))
    

    # 3.7: Calculate the apparent sun longitude, lambda (degrees):

    def lambda_sun_long(self, jme):
        return self.theta_gc_long(jme) + self.delta_psi(10*jme) + self.delta_tau(jme)
    

    # 3.8: Calculate the apparent sidereal time at Greenwich at any given time (in degrees)

    def v0(self):
        pass