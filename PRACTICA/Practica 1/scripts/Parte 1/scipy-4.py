import numpy
import matplotlib.pyplot as plt
import scipy.stats as ss
data = numpy.array([[113,105,130,101,138,118,87,116,75,96, 
	122,103,116,107,118,103,111,104,111,89,78,100,89,85,88], 
	[137,105,133,108,115,170,103,145,78,107, \
	84,148,147,87,166,146,123,135,112,93,76,116,78,101,123]])
dataDiff = data[1,:]-data[0,:]
# Ajuste de una normal a los datos
mean,std=ss.norm.fit(dataDiff)
params_cauchy=ss.cauchy.fit(dataDiff)
plt.hist(dataDiff,density=1)
x=numpy.linspace(dataDiff.min(),dataDiff.max(),1000)
pdf=ss.norm.pdf(x,mean,std)
kde=ss.gaussian_kde(dataDiff)
cauchy=ss.cauchy.pdf(x,*params_cauchy)
plt.plot(x,pdf)
plt.plot(x,kde(x))
plt.plot(x,cauchy)
plt.show()
