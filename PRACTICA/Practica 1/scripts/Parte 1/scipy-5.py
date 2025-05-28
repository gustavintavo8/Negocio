import numpy
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski, euclidean, chebyshev, cityblock
Square=numpy.meshgrid(numpy.linspace(-1.1,1.1,512),numpy.linspace(-1.1,1.1,512),indexing='ij')
X=Square[0]; Y=Square[1]
f=lambda x,y: cityblock([x,y],[0.0,0.0])<=1.0
Ball=numpy.vectorize(f)(X,Y)
plt.imshow(Ball); plt.axis('off'); plt.show()
