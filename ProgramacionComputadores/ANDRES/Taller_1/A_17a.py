# -*- coding: utf-8 -*-
# 
# # Ejercicio # 17a.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------
# Este programa lee una lista y determina cuál es el segundo elemento más grande 
# de dicha lista según el número de espacios que ocupa cada entrada.
# COMPARANDO TODAS LAS POSIBLES ENTRADAS
# ------------------------------------------------------------------------------


# Doy instrucciones sobre cómo ingresar los datos
print('A continuación ingresará los valores a los que les quiera analizar su '
      + 'longitud, para ello tenga cuenta las siguientes consideraciones: '
      + '\n'
      + '* Comience con un corchete abierto "[" y termine con uno cerrado "]".'
      + '\n* Ingrese los elementos separados por comas.'
      + '\n* Si ingresa una palabra, letra o caracter que no sea un dígito, '
      + 'colóquelo entre comillas "".')

while True:
    try:
        lista = eval(input('Ingrese el contenido de la lista: '))
        break
    except SyntaxError:
        print('\nPor favor, ríjase por las consideraciones anteriores.')

tam_1 = []      # El tamaño de los datos ingresados por el usuario
tam_2 = []      # El tamaño de los datos ingresados sin el mayor
ind_1 = 0       # Indice de dónde está el mayor de los datos sin modificar
ind_2 = 0       # Indice de dónde está el mayor de los datos modificados.

# casteo a string para poder comparar.
for i in lista:
    tam_1.append(len(str(i)))

# Elimino el mayor y creo una nueva lista en string con los que quedan.
for i in tam_1:
    if i is max(tam_1):
        ind_1 = tam_1.index(i)
        del lista[ind_1]

# El procedimiento anterior lo repito con los elementos restantes
for i in lista:
    tam_2.append(len(str(i))) 

# De los elementos restantes busco el máximo, su indice y lo comparo con el de 
# la lista ingresada que fue modificada.
for i in tam_2:
    if i is max(tam_2):      
        ind_2 = tam_2.index(i)
      
# el segundo mayor es el mayor de la lista donde no está el mayor de la lista 
# inicial que el usuario ingresa
print('\nEl segundo elemento más grande es: ', lista[ind_2])
print('Puede que hayan más elementos repetidos con el mismo tamaño.')

# Fin del código.