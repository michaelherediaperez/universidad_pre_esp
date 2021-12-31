# -*- coding: utf-8 -*-
# 
# # Ejercicio # 16.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------
# Este programa recibe una lista de números organizados y determina si están en 
# orden ascendente. Luego, lee un número y lo inserta en la lista en la posición
# adecuada.
# ------------------------------------------------------------------------------

# Leo los números que ingresa el usuario, considero que sólo puede ingresar 
# números enteros con el formato de una lista
while True:
    try:
        numeros = eval(input('Ingrese una serie de números organizados, '
                             + 'comience la lista con un corchete para abrir '
                             + '"[", ingrese los números separador por comas, '
                             + 'finalice con un corchete para cerrar "]".\n'))
        if type(numeros) is list:
            for numero in numeros:
                if type(numero) is int:
                    continue
                else: raise Exception
            break
        else: raise Exception
    except Exception:
        print('\nEste programa sólo soporta números enteros, rijase a las '
              + 'condiciones iniciales.')

ordenados = sorted(numeros)   # Lista con los números organizados

# Informo si la lista está organizada ascendentemente o no
if numeros != ordenados:
    print('\nLos números organizados ascendentemente quedan así:\n', ordenados)

# Leo el número que el usuario quiere ingresar en la lista
while True:
    try:
        num = int(input('Ingrese un número entero: '))
        break
    except ValueError:
        print('Debe ingresar un número entero.')

# Busco en cuál posición encajaría num, y luego lo añado en la lista ordenada
for i in ordenados:
    if num <= i: 
        ordenados.insert(ordenados.index(i), num)
        break

print('La lista con el número suministrado es: ', ordenados)

# Fin del código.