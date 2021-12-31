# -*- coding: utf-8 -*-
# 
# # Ejercicio # 20a.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, Manizales field.
# ------------------------------------------------------------------------------
# Este programa toma una matriz y la rota 90° en sentido antihorario.
# NO SE ESTÁ UTILIZANDO la librería numpy.
# ------------------------------------------------------------------------------

print('ESTE PROGRAMA LEERA UNA MATRIZ Y LA ROTARÁ 90° EN SENTIDO ANTIHORARIO.')

# Pido las dimensiones de la matriz, que cumplan con los requerimientos.
while True:
    try:
        F = int(input('\nIngrese el número de filas de la matriz: '))
        C = int(input('Ingrese el número de columnas de la matriz: '))
        if (F>=0) and (C>=0): break
        else: print('\nLas dimensiones de la matriz son positivas, ingrese'
                    + ' de nuevo los valores.') 
    except ValueError:
        print('\nLas dimensiones de la matriz son números enteros positivos.')

# Se piden las entradas de la matriz.
print('\nA continuación ingrese los valores de la matriz, fila por fila,'
      + ' de izqueirda a derecha')

matriz = []         # Almacena los datos suministrados por el usuario. 
rot = []            # Almacena la matriz rotada.

# Para cada fila se crea una lista, y por columna se pide un valor que se castea
# como flotante, cuando se llena toda una se añade a la matriz, así hasta que se
# llenen las filas. Si no se cumple con el requisito de ingresar un número se 
# pide volver a comenzar.
while True:
    try:
        matriz = [[float(input()) for m in range(C)] for n in range(F)]
        break
    except ValueError:
        print('\nEl programa esá limitado a trabajar con número enteros. Por'
              + ' favor, ingrese los valores nuevamente.')

# De cada fila se toma la última entrada [-1] y se añade a fila, luego esa fila 
# se añade a rot, después a cada fila se le toma la penúltima entrada [-2] y se 
# añade a fila, luego fila se añade a  rot, y así sucesivamente hasta terminar 
# con todas las filas de matriz. En cada paso fila se vuelve cero al pasar de 
# indice.
for i in range(1, C+1):
    fila = []
    for j in range(F):
        fila.append(matriz[j][-i])
    rot.append(fila)

# Imprimo rot, y cada entrada la separa con doble espacio '  '.
for i in range(len(rot)):
    for j in range(len(rot[1])):
        print(rot[i][j], end = '  ')
    print()

# Fin del código.