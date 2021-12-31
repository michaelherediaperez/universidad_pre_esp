# -*- coding: utf-8 -*-
# 
# Generación aleatoria de puntos siguiendo una distribución. Ejercicio # 06.
# ------------------------------------------------------------------------------
# Hecho por : Michael Heredia Pérez
# Fecha     : Mar/2018
# e-mail    : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------

# En el el link se encuentran todas las distribuciones disponibles:
# https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html

# Las librerías
import numpy as np 
import matplotlib.pyplot as plt 

N = 1000000 # Cantidad de números aleatorios.

# Defino varias distribuciones:
normal = np.random.normal(size = N)                 # Dist. Normal.
expo   = np.random.exponential(scale=1, size = N)   # Dist. Exponencial.
gamma  = np.random.gamma(2., 2., N)                 # Dist. Gamma,  mean=4, 
#                                                                   std=2*sqrt(2)


# Grafico las distribuciones.
plt.figure()            
plt.hist(normal, bins = 500, density = True)
plt.grid()
plt.xlabel('Números aleatorios', fontsize = 15)
plt.ylabel('Frecuencia', fontsize = 15)
plt.title('Distribución normal')
plt.show()

plt.figure()            
plt.hist(expo, bins = 500, density = True)
plt.grid()
plt.xlabel('Números aleatorios', fontsize = 15)
plt.ylabel('Frecuencia', fontsize = 15)
plt.title('Distribución exponencial')
plt.show()

plt.figure()            
plt.hist(gamma, bins = 500, density = True)
plt.grid()
plt.xlabel('Números aleatorios', fontsize = 15)
plt.ylabel('Frecuencia', fontsize = 15)
plt.title('Distribución gamma')
plt.show()

# Fin del código.