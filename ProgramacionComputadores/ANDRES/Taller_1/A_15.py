# -*- coding: utf-8 -*-
# 
# # Ejercicio # 15.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------
# Este programa recibe una lista de números enteros e identifica aquellos que se
# repiten exactamente 2 veces.
# ------------------------------------------------------------------------------

# Almaceno los números ingresados.
numeros = eval(input('Ingrese una lista de números separados por comas, '
                      + 'comenzando con un corchete para abrir "[" y terminando'
                      + ' con otro para cerrar \ "]": '))

repetidos = []      # Almacena aquellos que se repiten 2 veces

# Hago el conteo de los números repetidos 2 veces.
for i in numeros:
    if numeros.count(i) == 2: repetidos.append(i)

# Descarto los elementos repetidos en la lista repetidos:
for i in repetidos:
    if repetidos.count(i) > 1: repetidos.remove(i)

print('Los números que se repiten exactamente dos veces son: ', repetidos)

# Fin del código.