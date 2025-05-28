# Pareto
import numpy
import matplotlib.pyplot as plt
from scipy.stats import pareto, uniform
import matplotlib.pyplot as plt
a, b = 2, 3
x = numpy.linspace(0, 4, 1000)
# Funcion de densidad
plt.subplot(131); plt.plot(x, uniform.pdf(x, b-a))
# Funcion de distribucion
plt.subplot(132); plt.plot(x, uniform.cdf(x,b-a))
# Generador aleatorio
plt.subplot(133); plt.plot(uniform.rvs(b-a,size=1000))
plt.show()