# -*- coding: utf-8 -*-
# 
# Campos escalares. Ejercicio # 09.
# ------------------------------------------------------------------------------
# Hecho por : Michael Heredia Pérez
# Fecha     : Mar/2018
# e-mail    : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------

# Las librerías.
import numpy as np 
import matplotlib.pyplot as plt 
#from mpl_toolkits.mplot3d import Axes3D

# Defino los espacios.
xx = np.linspace(-10, 10, 100)
yy = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(xx, yy)      # Crea una malla de coordenadas con los puntos 
                                # anteriores.

# Defino la función y calculo valores.
funcion = lambda x, y : np.sin(x) + np.sin(y) - x
values = funcion(X, Y)

# Grafico como una superficie.
fig = plt.figure()
ax  = fig.add_subplot(111, projection = '3d')            
ax.plot_surface(X, Y, values)                   # El espacio (x, y, z)
ax.set_xlabel(r'$x$', fontsize = 15)
ax.set_ylabel(r'$y$', fontsize = 15)
ax.set_zlabel(r'$z$', fontsize = 15)
plt.show()

# Grafico como una superficie
fig  = plt.figure()
ax   = fig.add_subplot(111)
mesh = ax.pcolormesh(X, Y, values, vmin = -2, vmax = 2.0, cmap = 'hot')
cbar = plt.colorbar(mesh)
cbar.set_label(r'$f$', fontsize = 15)
ax.set_aspect('equal')
ax.set_xlabel(r'$xy$', fontsize = 15)
ax.set_ylabel(r'$y$', fontsize = 15)
plt.show()

# Fin del código.
'''
PENDIENTE ROTAR LAS LETRAS DE LABELS
'''