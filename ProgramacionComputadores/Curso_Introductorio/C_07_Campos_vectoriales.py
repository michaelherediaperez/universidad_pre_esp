# -*- coding: utf-8 -*-
# 
# Campos vectoriales. Ejercicio # 07.
# ------------------------------------------------------------------------------
# Hecho por : Michael Heredia Pérez
# Fecha     : Mar/2018
# e-mail    : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------

# Las librerías.
import numpy as np 
import matplotlib.pyplot as plt 
from itertools import product 

N = 20     # Cantidad de puntos.

# Defino la región del campo vectorial
region = np.linspace(-1, 1, N)
XX, YY = np.array(list(product(region, region))).T 

# Defino las componentes del campo vectorial y el campo.
Fx = np.cos(np.pi * YY - np.pi/2)
Fy = np.sin(np.pi * XX)
F  = np.sqrt(Fx**2 + Fy**2)

# Grafico el campo vectorial
fig = plt.figure()
ax = fig.add_subplot(111)
quiver = ax.quiver(XX, YY, Fx, Fy, F, pivot = 'middle', scale = 15, 
                   width = 0.01, cmap = 'seismic', clim = (0, 1.5))

plt.xlabel(r'$x$', fontsize = 15)
plt.ylabel(r'$y$', fontsize = 15)
cbar = plt.colorbar(quiver)
cbar.set_label(r'$F$', fontsize = 15, rotation = 0)

ax.set_aspect('equal')
plt.show()

# Fin del código.