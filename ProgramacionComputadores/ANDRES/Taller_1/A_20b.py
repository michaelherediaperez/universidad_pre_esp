# -*- coding: utf-8 -*-
# 
# # Ejercicio # 20b.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, Manizales field.
# ------------------------------------------------------------------------------
# Este programa toma una matriz y la rota 90° en sentido antihorario.
# SE ESTÁ UTILIZANDO la librería numpy.
# ------------------------------------------------------------------------------

# La librería numpy me permite trabajar con matrices.
import numpy as np 

print('ESTE PROGRAMA LEERÁ UNA MATRIZ Y LA ROTARÁ 90° EN SENTIDO ANTIHORARIO.')

F = int(input('Ingrese el número de filas de la matriz: ')) 
C = int(input('Ingrese el número de columnas de la matriz: ')) 
  
print('\nIngrese los valores fila por fila, de derecha a izquierda separados'
      + ' por un espacio.') 
  
# map(fun, iterables) retorna una lista con los resultados de aplicar una
# función 'fun' sobre una serie de valores iterables 'iterables'. En este caso, 
# se castea a flotante cada valor que introduce el usuario.

# .split() separa los valores ingresados cada que hay un espacio ' '.

# reshape(a, b) reorganiza el array en un array de a filas y b columnas.

while True:
    try:
        entradas = list(map(int, input().split())) 
        matriz = np.array(entradas).reshape(F, C) 
        break
    except ValueError:
        print('\nEl programa esá limitado a trabajar con número enteros. Por'
              + ' favor, ingrese los valores nuevamente.')
        
# np.rot90() gira la matriz que ingreso como argumento 90° en sentido 
# antihorario 
        
print(np.rot90(matriz))

# Fin del código.