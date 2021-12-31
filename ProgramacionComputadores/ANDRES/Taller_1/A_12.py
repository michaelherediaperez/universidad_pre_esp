# -*- coding: utf-8 -*-
# 
# # Ejercicio # 12.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------
# Este programa lee dos palabras y determina cuál se encuentra primera en el
# diccionario.
# ------------------------------------------------------------------------------

# Leo dos palabras
palabra_1 = input('Ingrese una palabra: ')
palabra_2 = input('Ingrese otra palabra: ')

# Evalúo cuál está primera o si son iguales
if palabra_1.lower() < palabra_2.lower():
    print(palabra_1, 'está primero en el diccionario.')
elif palabra_1.lower() == palabra_2.lower():
    print('Son la misma palabra.')
else:
    print(palabra_2, 'está de primera en el diccionario.')

# Fin del código.

