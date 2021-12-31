# -*- coding: utf-8 -*-
# 
# Calculo de integrales. Ejercicio # 10.
# ------------------------------------------------------------------------------
# Hecho por : Michael Heredia Pérez
# Fecha     : Mar/2018
# e-mail    : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------

# Las librerías.
import numpy as np 
from scipy.integrate import quad 

# Defino ambas funciones.
f1 = lambda x : np.exp(-x**2)
f2 = lambda x : x*np.exp(-x**2)

# Calculo las integrales
int1, error1 = quad(f1, -np.inf, np.inf)
int2, error2 = quad(f2, -np.inf, np.inf)

# Doy resultados.
print('La primera integral vale: ', int1)
print('El error de la primera integral es: ', error1)
print('La segunda integral vale: ', int2)
print('El error de la segunda integral es: ', error2)

# Fin del código.