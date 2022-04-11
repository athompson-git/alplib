# Classes and methods to get solar radiation angle of incidence on a surface
# Based on "Solar Position Algorithm for Solar Radiation Applications"
# by Ibrahim Reda and Afshin Andreas

import pkg_resources

import numpy as np
from numpy import pi, power, sin, cos, tan, arccos, arctan, arctan2, arcsin, heaviside
from numpy import deg2rad as d2r


class SPA:
    """
    Solar Position Algorithm for Solar Radiation Applications
    """
    def __init__(self):
        self.path_prefix = "data/solar/"
        self.file_extension = ".txt"

        self.r0_data = np.genfromtxt(pkg_resources.resource_filename(__name__, self.path_prefix \
            + "periodic_earth_terms_R0" + self.file_extension), skip_header=1)
        self.r1_data = np.genfromtxt(pkg_resources.resource_filename(__name__, self.path_prefix \
            + "periodic_earth_terms_R1" + self.file_extension), skip_header=1)
        self.r2_data = np.genfromtxt(pkg_resources.resource_filename(__name__, self.path_prefix \
            + "periodic_earth_terms_R2" + self.file_extension), skip_header=1)
        self.r3_data = np.genfromtxt(pkg_resources.resource_filename(__name__, self.path_prefix \
            + "periodic_earth_terms_R3" + self.file_extension), skip_header=1)
        self.r4_data = np.genfromtxt(pkg_resources.resource_filename(__name__, self.path_prefix \
            + "periodic_earth_terms_R4" + self.file_extension), skip_header=1)

        self.b0_data = np.genfromtxt(pkg_resources.resource_filename(__name__, self.path_prefix \
            + "periodic_earth_terms_B0" + self.file_extension), skip_header=1)
        self.b1_data = np.genfromtxt(pkg_resources.resource_filename(__name__, self.path_prefix \
            + "periodic_earth_terms_B1" + self.file_extension), skip_header=1)

        self.l0_data = np.genfromtxt(pkg_resources.resource_filename(__name__, self.path_prefix \
            + "periodic_earth_terms_L0" + self.file_extension), skip_header=1)
        self.l1_data = np.genfromtxt(pkg_resources.resource_filename(__name__, self.path_prefix \
            + "periodic_earth_terms_L1" + self.file_extension), skip_header=1)
        self.l2_data = np.genfromtxt(pkg_resources.resource_filename(__name__, self.path_prefix \
            + "periodic_earth_terms_L2" + self.file_extension), skip_header=1)
        self.l3_data = np.genfromtxt(pkg_resources.resource_filename(__name__, self.path_prefix \
            + "periodic_earth_terms_L3" + self.file_extension), skip_header=1)
        self.l4_data = np.genfromtxt(pkg_resources.resource_filename(__name__, self.path_prefix \
            + "periodic_earth_terms_L4" + self.file_extension), skip_header=1)
        self.l5_data = np.genfromtxt(pkg_resources.resource_filename(__name__, self.path_prefix \
            + "periodic_earth_terms_L5" + self.file_extension), skip_header=1)

        nuta_data = np.genfromtxt(pkg_resources.resource_filename(__name__, self.path_prefix \
            + "nutation_longitude_obliquity" + self.file_extension), skip_header=1)
        self.ai = nuta_data[:,5]
        self.bi = nuta_data[:,6]
        self.ci = nuta_data[:,7]
        self.di = nuta_data[:,8]
        self.yi0 = nuta_data[:,0]
        self.yi1 = nuta_data[:,1]
        self.yi2 = nuta_data[:,2]
        self.yi3 = nuta_data[:,3]
        self.yi4 = nuta_data[:,4]
    

    # Helper functions

    def deltaT(self, year):
        t = (year - 1820)/100
        return -20 + 32 * t**2  # seconds


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
    
    def date_to_jme(self, y, m, d, b=0.0):
        return self.jme(self.jce(self.jde(self.jd(y,m,d), self.deltaT(y))))


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
        l5 = self.l5_data[1] * np.cos(self.l5_data[2] + self.l5_data[3] * jme)
        
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
        r4 = self.r4_data[1] * np.cos(self.r4_data[2] + self.r4_data[3] * jme)
        
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

    def v0(self, jd):
        v0 = 280.46061837 + 360.98564736629 * (jd - 2451545) \
            + 0.000387933*power(self.jc(jd),2) - power(self.jc(jd),3) / 38710000
        if v0 > 0.0:
            return v0 % 360.0
        elif v0 < 0.0:
            return 360.0 - v0 % 360.0 
        return 0.0
    
    def v(self, jd, year=2022):
        jde = self.jde(jd, self.deltaT(year))
        return self.v0(jd) + self.delta_psi(self.jce(jde)) * cos(d2r(self.epsilon(self.jme(self.jce(jde)))))
    

    # 3.9: Calculate the geocentric sun right ascension, alpha (degrees)

    def alpha(self, jme):
        alpha = (180.0 / pi)*arctan2((sin(d2r(self.lambda_sun_long(jme)))*cos(d2r(self.epsilon(jme))) \
            - tan(d2r(self.beta_gc_lat(jme)))*sin(d2r(self.epsilon(jme)))) / cos(d2r(self.lambda_sun_long(jme))), 1.0)
        if alpha > 0.0:
            return alpha % 360.0
        elif alpha < 0.0:
            return 360.0 - alpha % 360.0 
        return 0.0
    

    # 3.10: Calculate the geocentric sun declination

    def delta(self, jme):
        return (180.0/pi)*arcsin(sin(d2r(self.beta_gc_lat(jme)))*cos(d2r(self.epsilon(jme))) \
            + cos(d2r(self.beta_gc_lat(jme)))*sin(d2r(self.epsilon(jme)))*sin(d2r(self.lambda_sun_long(jme))))


    # 3.11: Calculate the observer local hour anlge H

    def h_hour_angle(self, jd, year, lon):
        jme = self.jme(self.jce(self.jde(jd, self.deltaT(year))))
        return self.v(jd, year) + lon - self.alpha(jme)


    # 3.12 Calculate the topocentric sun right ascension alphaPrime (radians)

    def delta_prime(self, y, m, d, lat, lon, elev):
        jme = self.date_to_jme(y, m, d)
        xi = d2r(8.794 / (3600 * self.earth_hc_radius(jme)))
        u = arctan(0.99664719 * tan(d2r(lat)))
        x = cos(u) + (elev / 6378140) * cos(d2r(lat))
        y = 0.99664719*sin(u) + (elev / 6378140) * sin(d2r(lat))
        H = d2r(self.h_hour_angle(self.jd(y, m, d), y, lon))

        delta_alpha = arctan2(-x*sin(xi)*sin(H) / (cos(d2r(self.delta(jme))) - x*sin(xi)*cos(H)), 1.0)
        #alpha_prime = d2r(self.alpha(jme)) + delta_alpha

        return arctan2(((sin(d2r(self.delta(jme))) - y*sin(xi))*cos(delta_alpha)) \
                            / (cos(d2r(self.delta(jme))) - x*sin(xi)*cos(H)), 1.0)
    

    # 3.13: Calculate the topocentric local hour anlge, Hprime (radians)

    def h_prime(self, y, m, d, lat, lon, elev):
        jme = self.date_to_jme(y, m, d)
        xi = d2r(8.794 / (3600 * self.earth_hc_radius(jme)))
        u = arctan(0.99664719 * tan(d2r(lat)))
        x = cos(u) + (elev / 6378140) * cos(d2r(lat))
        y = 0.99664719*sin(u) + (elev / 6378140) * sin(d2r(lat))
        H = d2r(self.h_hour_angle(self.jd(y, m, d), y, lon))

        delta_alpha = arctan2(-x*sin(xi)*sin(H) / (cos(d2r(self.delta(jme))) - x*sin(xi)*cos(H)), 1.0)
    
        return H - delta_alpha
    

    # 3.14: Calculate the topocentric zenith angle (degrees)

    def theta_topo_elev(self, y, m, d, lat, lon, elev, pres=1013.25, temp=20.0):
        # pressure in millibars
        # temperature in celcius
        delta_prime = self.delta_prime(y, m, d, lat, lon, elev)
        h_prime = self.h_prime(y, m, d, lat, lon, elev)

        e0 = (180/pi)*arcsin(sin(d2r(lat))*sin(delta_prime) + cos(d2r(lat))*cos(delta_prime)*cos(h_prime))  
        delta_e = (pres/1010)*(283/(273+temp))*(1.02/60.0/tan(d2r(e0 + 10.3/(e0+5.11))))
    
        return 90 - (e0 + delta_e)
    

    # 3.15: Calculate the topocentric azimuth angle

    def gamma_topo_azimuth(self, y, m, d, lat, lon, elev):
        delta_prime = self.delta_prime(y, m, d, lat, lon, elev)
        h_prime = self.h_prime(y, m, d, lat, lon, elev)

        gamma = (180/pi)*arctan2(sin(h_prime)/(cos(h_prime)*sin(d2r(lat)) - tan(delta_prime)*cos(d2r(lat))), 1.0)
        gamma = heaviside(gamma, 0.0) * (gamma % 360.0) + heaviside(-gamma, 0.0) * (360.0 - gamma % 360.0)
        return gamma


    # 3.16: Calculate the incidence angle for a surface oriented in any direction

    def incidence_angle(self, omega, gamma, y, m, d, lat, lon, elev, pres=1013.25, temp=20.0):
        theta = d2r(self.theta_topo_elev(y, m, d, lat, lon, elev, pres, temp))
        big_gamma = d2r(self.gamma_topo_azimuth(y, m, d, lat, lon, elev))

        return arccos(cos(theta)*cos(d2r(omega))+ sin(d2r(omega))*sin(theta)*cos(big_gamma - d2r(gamma)))