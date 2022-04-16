# Provide interpolating functions and integrals of the atomic response functions from DarkARC

import numpy as np
from numpy import sqrt, exp, log10, log, pi, interp
from scipy.interpolate import interp2d
from scipy.integrate import dblquad


# Flux is arranged in TSV format, (T, q) = (row, col)


class XeResponse:
    def __init__(self, shell, keV=1.0):
        if shell=="5p":
            self.data = np.genfromtxt("data/Xe_5p.txt")
        if shell=="5s":
            self.data = np.genfromtxt("data/Xe_5p.txt")
        if shell=="4d":
            self.data = np.genfromtxt("data/Xe_4d.txt")
        if shell=="4p":
            self.data = np.genfromtxt("data/Xe_4p.txt")
        if shell=="4s":
            self.data = np.genfromtxt("data/Xe_4s.txt")
        if shell=="3d":
            self.data = np.genfromtxt("data/Xe_3d.txt")
        if shell=="3p":
            self.data = np.genfromtxt("data/Xe_3p.txt")
        if shell=="3s":
            self.data = np.genfromtxt("data/Xe_3s.txt")
        if shell=="2p":
            self.data = np.genfromtxt("data/Xe_2p.txt")
        if shell=="2s":
            self.data = np.genfromtxt("data/Xe_2s.txt")
        if shell=="1s":
            self.data = np.genfromtxt("data/Xe_1s.txt")
        if shell=="total":
            self.data = np.genfromtxt("data/Xe_total.txt")
        
        self.keV = keV
        self.me = 511*self.keV
        self.qMin = 1*self.keV
        self.qMax = 1000*self.keV
        self.kMin = 0.1*self.keV
        self.kMax = 1000*self.keV
        self.TMin = sqrt(2*self.me*self.kMin)
        self.TMax = sqrt(2*self.me*self.kMax)
        self.gridsize = 100
        self.qGrid = np.logspace(np.log10(self.qMin),np.log10(self.qMax),self.gridsize)
        self.kGrid = np.logspace(np.log10(self.kMin),np.log10(self.kMax),self.gridsize)
        self.TGrid = self.kGrid**2 / (2 * self.me)
        self.dlnq = log10(self.qGrid[1]) - log10(self.qGrid[0])
        self.dlnk = log10(self.kGrid[1]) - log10(self.kGrid[0])
        self.qGrid_edges = np.logspace(log10(self.qMin) - self.dlnq/2, log10(self.qMax) + self.dlnq/2, 101)
        self.kGrid_edges = np.logspace(log10(self.kMin) - self.dlnk/2, log10(self.kMax) + self.dlnk/2, 101)
        self.delq = self.qGrid_edges[1:] - self.qGrid_edges[:-1]
        self.delk = self.kGrid_edges[1:] - self.kGrid_edges[:-1]

    def W1(self, T, q):
        k = sqrt(2*self.me*T)
        f = interp2d(self.qGrid, self.kGrid, self.data)
        if q < self.qMin:
            return (q / self.qMin)**2 * f(self.qMin, k)
        return f(q, k)
    
    def DblIntegrate(self, wgtfunc, q_low, q_high, T_low, T_high, nsamples=1000):
        # We use logarithmic MC integration here
        def f(T, q):
            return T * q * self.W1(T, q) * wgtfunc(T, q) 

        integrand = np.vectorize(f)

        dq_grid = 10**np.random.uniform(log10(q_low), log10(q_high), nsamples)
        dT_grid = 10**np.random.uniform(log10(T_low), log10(T_high), nsamples)

        volume = (log10(q_high) - log10(q_low))*(log10(T_high) - log10(T_low)) * log(10)**2

        return volume * np.sum(integrand(dT_grid, dq_grid)) / nsamples
    
    def TIntegrate(self, q, wgtfunc, T_low, T_high, nsamples=1000):
        # We use logarithmic MC integration here
        def f(T):
            return T * self.W1(T, q) * (1 / (8*T)) * wgtfunc(T)  # factor of T for log-MC

        integrand = np.vectorize(f)

        dT_grid = 10**np.random.uniform(log10(T_low), log10(T_high), nsamples)

        volume = (log10(T_high) - log10(T_low)) * log(10)

        return np.sum(volume * integrand(dT_grid) / nsamples)
    
    def QIntegrate(self, T, wgtfunc, q_low, q_high, nsamples=1000):
        # We use logarithmic MC integration here
        def f(q):
            return q * self.W1(T, q) * (1 / (8*T)) * wgtfunc(q)  # factor of q for log-MC

        integrand = np.vectorize(f)

        dq_grid = 10**np.random.uniform(log10(q_low), log10(q_high), nsamples)

        volume = (log10(q_high) - log10(q_low)) * log(10)

        return np.sum(volume * integrand(dq_grid) / nsamples)




# Deprecated
"""
    def TGridIntegrate(self, wgtfunc, T_low, T_high):
        k_low = sqrt(2*self.me*T_low)
        k_high = sqrt(2*self.me*T_high)
        idx_k_low = (np.abs(self.kGrid-k_low)).argmin()
        idx_k_high = (np.abs(self.kGrid-k_high)).argmin()

        QQ, KK = np.meshgrid(self.qGrid, self.kGrid[idx_k_low:idx_k_high])
        dQQ, dKK = np.meshgrid(self.delq, self.delk[idx_k_low:idx_k_high])
        wgts = wgtfunc(KK, QQ)
        resp = self.data[idx_k_low:idx_k_high,:]

        return np.sum(wgts * resp * (KK * dKK / self.me), axis=0)
    
    def DblGridIntegrate(self, wgtfunc, q_low, q_high, T_low, T_high):
        k_low = sqrt(2*self.me*T_low)
        k_high = sqrt(2*self.me*T_high)
        idx_k_low = (np.abs(self.kGrid-k_low)).argmin()
        idx_k_high = (np.abs(self.kGrid-k_high)).argmin()
        idx_q_low = (np.abs(self.qGrid-q_low)).argmin()
        idx_q_high = (np.abs(self.qGrid-q_high)).argmin()

        QQ, KK = np.meshgrid(self.qGrid[idx_q_low:idx_q_high], self.kGrid[idx_k_low:idx_k_high])
        dQQ, dKK = np.meshgrid(self.delq[idx_q_low:idx_q_high], self.delk[idx_k_low:idx_k_high])
        wgts = wgtfunc(KK, QQ)
        resp = self.data[idx_k_low:idx_k_high,idx_q_low:idx_q_high]

        return np.sum(wgts * resp * dQQ * (KK * dKK / self.me))
"""

