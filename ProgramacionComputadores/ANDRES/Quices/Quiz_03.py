# -*- coding: utf-8 -*-
#
# Quiz #03
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Ago/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------

# %% EJERCICIO 1

# Este programa imprime una matriz de 9x7 con determinadas características:
#   * Las primeras 3 columnas son cero (0).
#   * La cuerta columna vale 0.5, pero en la última fila val 0.7.
#   * Las otras tres columnas valen 1.
# También el promedio de la última fila

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

# %% EJERCICIO 2. 

# Haciendo uso de la liibrería Pandas, se lee el fichero NucleosPoblacion.csv y 
# se responde por:
#   q1: ¿Cuántos Municipios tienen más de 100000 habitantes? 
#   q2: ¿Cuál es la segunda ciudad más poblada?
#   q3: ¿Qué posición ocupa Granada en el ranking de las más pobladas?
# Se exporta el fichero y se entrega como un formato .xlsx llamado 
# 'NucleosPoblacionSinComas'.

# Importo las librerías
import pandas as pd
import numpy as np

# Leo el documento con los nucleos poblacionales
datos = pd.read_csv("NucleosPoblacion.csv")

# Se cuentan cuantos municipios tiene más de 100000 habitantes
q1 = datos['Poblacion'] > 100000
q1_ans = np.sum(q1)

print('Número de municipios con más de 100000 habitantes: {0}'.format(q1_ans))
print()

# Creo un nuevo arreglo que esté ordenado en base a la población, para así 
# contar el penúltimo.
datos_ord = datos.sort_values(by='Poblacion')

# Se toma el último elemento del arreglo
q2 = datos_ord.iloc[len(datos)-2]

print(f"El segundo municipio más poblado es {q2.loc['Municipio']}, "
      f"con una población de {int(q2.loc['Poblacion'])} habitantes.")
print()

# Creo una nueva columna pob_rank en base al ranking de las más pobladas.
datos['pob_rank'] = datos['Poblacion'].rank(ascending=0)

# Se saca la informacion para la fila de Granada
q3 = datos[datos['Texto'] == 'Granada']

# Cambio el indice para corresponder con Granada.
q3.set_index('Texto',inplace=True)

# Se imprime la sentencia seleccionando sobre q3 la columna de pob_rank
print(f"La posición de Granada en el ranking de los más "
      f"poblados es {int(q3.at['Granada', 'pob_rank'])}.")

# Exporto el .csv a .xlsx 
datos.to_excel('NucleosPoblacionSinCmas.xlsx', sheet_name='a_excel_quiz3')