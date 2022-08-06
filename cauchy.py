import numpy as np
import matplotlib.pyplot as plt

def func(X,d):
	x,n = X
	r = ( (n - 1) / (n + 1) ) ** 2
	return -np.log10( ((1 - r)**2)  / (1 + (r**2) - (2*r*np.cos(4*np.pi*n*d/x)))) 
 
def n(polymer,wavelength):
        ## Use cauchy formula and data from literature to extract index of refraction
        if polymer == "PVC":
                return np.ones(wavelength.shape[0]) * 1.531
        if polymer == "PS":
                return 1.5718 + (8412 / (wavelength ** 2)) + ((2.35*(10**8)) / (wavelength ** 4)) 


x = np.linspace(200,800,1000)
y = func((x,n("PS",x)),80)
plt.plot(x,y)
plt.tight_layout()
plt.show()
