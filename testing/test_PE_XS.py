import sys
sys.path.append("../src/")

from numpy import logspace
import matplotlib.pyplot as plt

from photoelectric_xs import PECrossSection

H_xs = PECrossSection("H")
Be_xs = PECrossSection("Be")
Ge_hxs = PECrossSection("Ge")
Si_hxs = PECrossSection("Si")
CsI_hxs = PECrossSection("CsI")
NaI_hxs = PECrossSection("NaI")
C_hxs = PECrossSection("C")
Xe_hxs = PECrossSection("Xe")
Ar_hxs = PECrossSection("Ar")

#print(Ge_hxs.sigma(1.09*8.75071e-3))

energies = logspace(-3, 4, 1000)

plt.plot(energies, H_xs.sigma(energies), label="H")
plt.plot(energies, Be_xs.sigma(energies), label="Be")
plt.plot(energies, Ge_hxs.sigma(energies), label="Ge")
plt.plot(energies, Si_hxs.sigma(energies), label="Si")
plt.plot(energies, CsI_hxs.sigma(energies), label="CsI")
plt.plot(energies, NaI_hxs.sigma(energies), label="NaI")
plt.plot(energies, Xe_hxs.sigma(energies), label="Xe")
plt.plot(energies, Ar_hxs.sigma(energies), label="Ar")
plt.plot(energies, C_hxs.sigma(energies), label="C")
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r"$E$ [MeV]")
plt.ylabel(r"$\sigma$ [cm$^2$]")
plt.legend()
plt.show()
plt.close()

