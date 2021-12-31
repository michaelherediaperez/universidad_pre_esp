# -*- coding: utf-8 -*-
# 
# # Ejercicio # 11.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------
# Este programa lee una cadena de texto y reporta lo siguiente:
# a) Cuántas letras vocales en mayúscula se entraron
# b) Cuántas letras con tilde se entraron
# c) Cuántos digitos se entraron
# d) cuántos espacios se entraron
# e) Cuántas palabras reservadas se entraron
# ------------------------------------------------------------------------------

cadena = input('Ingrese una frase u oración que desee analizar: ')

# Saco una lista con las letras y espacios, y otra solo con las palabras
cadena_lis = list(cadena)
para_key = cadena.split()

# La librería keyword me da la lista con las palabras reservadas
import keyword
key = keyword.kwlist

# Creo un banco de datos para comparar
vocal_mayus = [vocal for vocal in 'AEIOUÁÉÍÓÚÜ']
tildes = [letra for letra in 'áéíóúÁÉÍÓÚ']

cuenta_mayus = 0           # Cantidad de vocales en mayúscula
cuenta_tildes = 0          # Cantidad de vocales con tildes en mayus y en minus
cuenta_digitos = 0         # Cantidad de dígitos
cuenta_espacios = 0        # Cantidad de espacios que hay
cuenta_key = 0             # Cantidad de palabras reservadas

# Itero sobre los caracteres de la frase para responder por a), b), c) y d)
for i in cadena_lis:
    if i in vocal_mayus: cuenta_mayus += 1
    if i in tildes: cuenta_tildes += 1
    if i.isdigit(): cuenta_digitos += 1
    if i == ' ': cuenta_espacios += 1

# Itero en las palabras para responder por e)
for i in para_key:
    if i in key: cuenta_key += 1
       
print('La cantidad de vocales en mayúscula es: ', cuenta_mayus) 
print('La cantidad de vocales con tilde, mayúsculas y minúsculas, es: ', \
cuenta_tildes)
print('La cantidad de dígitos es: ', cuenta_digitos)
print('La cantidad de espacios es: ', cuenta_espacios)
print('La cantidad de palabras reservadas es: ', cuenta_key)

# Fin del código.