import numpy as np
import matplotlib.pyplot as py

x = 11
d = np.linspace(0.25, 10, 200)
w = 2*np.pi/50
k = w

f = np.sin(-w*x)
j = 1/(2*210*d*w) * (64*np.cos(-w*x-5*k*d) - 175*np.cos(-w*x-4*k*d) + 350*np.cos(-w*x-2*k*d) - 189*np.cos(-w*x) - 50*np.cos(-w*x+2*k*d))

py.plot(d, np.abs(j-f)**(1/4))
py.show()