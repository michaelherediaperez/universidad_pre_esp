# -*- coding: utf-8 -*-
# 
# Solución de sistema de ecuaciones. Ejercicio # 02.
# ------------------------------------------------------------------------------
# Hecho por : Michael Heredia Pérez
# Fecha     : Mar/2018
# e-mail    : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------

# Librerías
import numpy as np 

# Defino la matriz de coeficientes.
AA = np.array([[ 2, -1,  4,  1, -1],
              [-1,  3, -2, -1,  2],
              [ 5,  1,  3, -4,  1],
              [ 3, -2, -2, -2,  3],
              [-4, -1, -5,  3, -4]])

# Defino el vector de resultados.
bb = np.array([7, 1, 33, 24, -49])

# Calculo la inversa de la matriz A.
inv_AA = np.linalg.inv(AA)

# La solución AX = b -> X = A^(-1) * b
XX = np.matmul(inv_AA, bb)  

# Imprimo la respuesta.
x, y, z, t, u = XX
print('x = {:.1f}'.format(x))
print('y = {:.1f}'.format(y))
print('z = {:.1f}'.format(z))
print('t = {:.1f}'.format(t))
print('u = {:.1f}'.format(u))

# Fin del código.