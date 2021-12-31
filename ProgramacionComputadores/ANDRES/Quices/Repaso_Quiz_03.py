# -*- coding: utf-8 -*-
#
# Repaso Quiz #03
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Ago/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------


# %% Ejercicio 1.
#    Este programa imprime una matriz de 9x7 con determinadas características:   
#    * Las primeras 3 columnas son cero (0).
#    * La cuerta columna vale 0.5, pero en la última fila val 0.7.
#    * Las otras tres columnas valen 1.
#    También el promedio de la última fila

# Importo librerias
import numpy as np

renglon = [0, 0, 0, 0.5, 1, 1, 1]       # Defino un renglón guía
fila = renglon * 9                      # Repito filas veces el renglón
matriz = np.array(fila).reshape(9, 7)   # Creo la matriz con las dimensiones

# Pongo el dato que debe ser diferente en la última entrada de la 4 columna
matriz[8][3] = 0.7

promedio = np.mean(matriz[-1])

# Doy el resultado
print(matriz)
print('El promedio de la última fila es :', round(promedio, 3))

# %% Ejercicio 2.
#    Este programa toma una matriz aleatoria de 5x5 con valores entre 0 y 1, y 
#    selecciona las posiciones de los elementos mayores que 0.5.

# Importo la librería
import numpy as np

# Se crea la matriz aleatoria y redondeo a tres decimales
matriz_aleatoria = np.random.rand(5, 5).round(3)
print(matriz_aleatoria)

print()

# Determino los elementos que son mayores a 0.5
elemento = matriz_aleatoria > 0.5

# Hago la selección e imprimo la respuesta
for i in range(5):
    for j in range(5):
        if elemento[i, j] == True:
            print(matriz_aleatoria[i][j], 'es mayor a 0.5'
                  + ' y su posición es {} '.format((i+1, j+1)))

# %% Ejercicio 3.
#    Este programa compara dos arrays de 4 entradas y evalúa si son iguales 
#    tomando una tolerancia absoluta en cada posición inferior a 0.16.

# Importo la librería
import numpy as np

# Defino los arrays
a = np.array([0, 1, 2, 3])
b = np.array([-0.1, 1.01, 1.98, 3.15])

print('a = {0} y b = {1}'.format(a, b))

# Creo un array que determina si son iguales bajo la tolerancia dada
diferencia = np.absolute(a - b) < 0.16

# Doy la respuesta
if diferencia.all():    
    print('a y b son iguales bajo una tolerancia absoluta de 0.16.')     
else:
    print('a y b no son iguales bajo una tolerancia absoluta de 0.16.')

# %% Ejercicio 4
#    Haciendo uso de la liibrería Pandas, se lee el fichero NucleosPoblacion.csv
#    y se responde por:
#    q1: ¿Cuántos Municipios tienen más de 100000 habitantes? 
#    q2: ¿Cuál es la segunda ciudad más poblada?
#    q3: ¿Qué posición ocupa Granada en el ranking de las más pobladas?
#    q4: Escriba los nombres de los 10 municipios con menos población.
#    q5: ¿Cuántos municipios de Extremadura tienen más de 5000 habitantes?
#    q6: ¿Cuál es el municipio situado más al Norte?
#       (Usar el valor de la coordenada "Y" que representa la latitud en grados). 
#    q7: Proporcione también la provincia a la que pertenece y su población. 
#    Se exporta el fichero y se entrega como un formato .xlsx llamado 
#    'NucleosPoblacionSinComas'.

# Importo las librerías
import pandas as pd
import numpy as np

# Leo el documento con la info
datos = pd.read_csv("NucleosPoblacion.csv")

# Se cuentan cuantos municipios tiene más de 100000 habitantes
q1 = datos['Poblacion'] > 100000
q1_ans = np.sum(q1)
print('Número de municipios con más de 100000 habitantes: {0}'.format(q1_ans))

# Se mira cuál es la segunda ciudad más poblada.
q2 = datos[Población['Poblacion']]
q2_sin_max = np.delete(q2, np.max(q2))
q2_ans = np.max(q2)
print(')

# Se cuentan los municipios de Extremadura con más de 5000 habitantes
q5 = datos['Población']