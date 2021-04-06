import sys
sys.path.append("../src/")

from numpy import logspace
import matplotlib.pyplot as plt

from borrmann import Borrmann




bcalc = Borrmann("Ge", 7e21, 304)

print(bcalc.epsilon(2,2,0))