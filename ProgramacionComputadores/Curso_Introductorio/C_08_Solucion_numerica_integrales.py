# -*- coding: utf-8 -*-
# 
# Calculo de integrales. Ejercicio # 08.
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

# Defino la función del ejercicio.

funcion = lambda x : np.exp(-x**2)

# Defino los límites de integración.
x1 = -np.inf;   x2 = np.inf

# Calculo la integral con una cuadratura.
val, error = quad(funcion, x1, x2)

print('El valor de la integral es: ', val)
print('El error en la integración es de: ', error)

# Fin del código.
'''
PENDIENTE AHCERLO CON CUADRATURAS DE GAUSS-LEGENDRE
'''