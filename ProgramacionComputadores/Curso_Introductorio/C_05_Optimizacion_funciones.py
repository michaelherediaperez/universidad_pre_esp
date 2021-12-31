# -*- coding: utf-8 -*-
# 
# Optimización de una función. Ejercicio # 05.
# ------------------------------------------------------------------------------
# Hecho por : Michael Heredia Pérez
# Fecha     : Mar/2018
# e-mail    : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------

# Las librerías.
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.optimize as opt 

# Cargo la información de .dat.
XX, YY = np.loadtxt('data_ej_05.dat', unpack = True)

# Grafico la información.
plt.figure()
plt.plot(XX, YY, 'o', label = 'data')
plt.legend(loc = 'best')
plt.grid()
plt.xlabel(r'$x$', fontsize = 15)
plt.ylabel(r'$y$', fontsize = 15)
plt.show()

# Tiene forma de la suma de dos funciones de distribución normal (aproximado) 
# entonces supongo una funciónde ese tipo.
def function(x, A1, B1, C1, A2, B2, C2):
    return A1*np.exp(-B1*(x-C1)**2) + A2 * np.exp(-B2 * (x-C2) **2)

# Calculo los parámetros para function que se ajusten a los datos.
params, pcov = opt.curve_fit(function, XX, YY)
print('Los parámetros son:', params)

# Calculo errores
errors = np.sqrt(np.diag(pcov))
print('Los errores del ajuste son:', errors)

fig = function(XX, *params)

plt.figure()
plt.plot(XX, YY,    'o', label = 'data')
plt.plot(XX, fig, '--r', label = 'fit')
plt.legend(loc = 'best')
plt.grid()
plt.xlabel(r'$x$', fontsize = 15)
plt.ylabel(r'$y$', fontsize = 15)
plt.show()

# Fin del código