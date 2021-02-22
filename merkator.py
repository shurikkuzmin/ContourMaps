import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


x = np.linspace(35.0,40.0, 101) / 180.0 * np.pi
y = np.linspace(50.0, 55.0, 101) / 180.0 * np.pi
y2 = np.log(np.tan(np.pi / 4.0 + y / 2.0))

X, Y =np.array(np.meshgrid(x,y2)) / np.pi * 180.0

plt.scatter(X,Y)

#Z = np.ones_like(X)

#fig, ax = plt.subplots(subplot_kw = {"projection": "3d"})

#surf = ax.plot_surface(X, Y, Z)

plt.show()
