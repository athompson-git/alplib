import sys
sys.path.append("../src/")

import numpy as np
import matplotlib.pyplot as plt

from borrmann import Borrmann




bcalc = Borrmann("Ge", 4.41348e22/2, 370)

print(bcalc.epsilon(2,2,0))
print(bcalc.epsilon(2,4,0))
print(bcalc.epsilon(2,4,2))
print(bcalc.epsilon(3,3,1))
print(bcalc.epsilon(1,1,1))
print(bcalc.epsilon(2,4,1))

energies = np.logspace(-3, -1, 1000)
imff = bcalc.imffe(energies)

plt.plot(energies, imff)
plt.xlabel(r"$E$ [MeV]")
plt.ylabel(r"f")
plt.xscale('log')
plt.yscale('log')
plt.show()
plt.close()
