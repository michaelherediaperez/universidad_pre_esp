# -*- coding: utf-8 -*-
#  
# Ajuste de funciones. Ejercicio # 11.
# ------------------------------------------------------------------------------
# Hecho por : Michael Heredia Pérez
# Fecha     : Mar/2018
# e-mail    : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------

# Las librerías
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import quad
from scipy.optimize import curve_fit

# Cargo el archivo .dat.
XX, YY = np.loadtxt('data_ej_11.txt', unpack = True)

# Grafico los puntos.
plt.figure()
plt.scatter(XX, YY, s = 1)
plt.grid()
plt.xlabel(r'$x$', fontsize = 15)
plt.ylabel(r'$y$', fontsize = 15)
plt.show()

# Defino la función de ajuste.
def funcion(x, A, B, C):
    return np.exp(A*x)*np.sin(B*x + C)

# Calculo los parámetros
params, _ = curve_fit(funcion, XX, YY)
print('Los parámetros de juste son: ', params)

# Hago el ajuste
ajuste = funcion(XX, *params)        # *params despliega todos los elementos del
                                    # arreglo.

# Grafico la función 
plt.figure()
plt.scatter(XX, YY, s = 10, label = 'Data')
plt.plot(XX, ajuste, '--r', label = 'Ajuste')
plt.grid()
plt.legend(loc = 'best', fontsize = 15)
plt.xlabel(r'$x$', fontsize = 15)
plt.ylabel(r'$y$', fontsize = 15)
plt.show()

# Calculo el área con la función quad de scipy.
A, B, C = params
A_quad = quad(funcion, 0, 50, args=(A, B, C))
print('El valor del área bajo la curva con quad es: ', A_quad)

# Fin del código.