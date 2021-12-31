# -*- coding: utf-8 -*-
# 
# # Ejercicio # 18.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, Manizales field.
# ------------------------------------------------------------------------------
# Este programa lee una matriz cuadrada A, calcula su matriz triangular superior
# U y determina la fila con mayor cantidad de unos en U.
# ------------------------------------------------------------------------------

# Pido las dimensiones de la matriz, que cumplan con los requerimientos.
while True:
    try:
        N = int(input('\nIngrese la dimensión de la matriz: '))
        if (N>=0): break
        else: print('\nLas dimensiones de la matriz son positivas, ingrese'
                    + ' de nuevo los valores.') 
    except ValueError:
        print('\nLas dimensiones de la matriz son números enteros positivos.')

matriz = []             # Almacena las filas(listas) que se ingresan.
U = []                  # La matriz triangular
unos = []               # Almacena los unos en cada fila de U

# Se piden las entradas de la matriz.
print('\nA continuación ingrese los valores de la matriz, fila por fila,'
      + ' de izqueirda a derecha, y confirme con ENTER:')

# Para cada fila se crea una lista, y por columna se pide un valor que se cast.
# como entero, cuando se llena toda una se añade a la matriz, así hasta que se
# llenen las filas. Si no se cumple con el requisito de ingresar un número se 
# pide volver a comenzar.
while True:
    try:
        matriz = [[int(input()) for m in range(N)] for n in range(N)]
        break
    except ValueError:
        print('\nEl programa esá limitado a trabajar con número enteros. Por'
              + ' favor, ingrese los valores nuevamente.')

# Calculo la matriz triangular superior asociada U, si al estar en la posición
# de matriz (estética) el elemento está debajo de la diagonal, este se vuelve 
# cero, sino, queda igual
for i in range(N):
    fila = []
    for j in range(N):
        if (j<i):
            fila.append(0)           
        else:
            fila.append(matriz[i][j])
    U.append(fila)

# Imprimo la matriz resultante
print('La matriz triangular asociada U es: \n')
for fila in U: print(fila)

# Obtengo una lista con la cant. de unos que hay en cada fila de la matriz U
for i in range(N):
    cant_1 = ''
    for j in U[i]:
        if j == 1: cant_1 += str(j)
        else: cant_1 += ''
    unos.append(cant_1)

# Encuentro cuál fila tiene la mayor cantidad de unos
for i in unos:
    if i == max(unos):
        print('\nLa fila con la mayor cantidad de unos (1) es la '
              + 'fila ', str(unos.index(i) + 1))

# Fin del código.