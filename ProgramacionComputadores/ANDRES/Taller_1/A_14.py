# -*- coding: utf-8 -*-
# 
# # Ejercicio # 14.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------
# Este programa lee una frase, la descompone en sus caracteres y los imprime como
# una sola 'palabra' organizadas algabéticamente.
# ------------------------------------------------------------------------------

cadena = list(input('Ingrese una frase u oración que desee: '))

# Remuevo los espacios de la oración
while ' ' in cadena: cadena.remove(' ')

# Organizo los caracteres, y los juntos en una sola sentencia
cadena = sorted(cadena)
cadena = ''.join(cadena)
print(cadena)

# Fin del código.