# -*- coding: utf-8 -*-
# 
# # Ejercicio # 04.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------
# Este programa lee un número entero positivo y determina la suma de sus dígitos 
# pares.
# Un número par x es aquel que x/2 es un valor exacto, lo cual es que el residuo 
# de la división es nulo (vale cero).
# ------------------------------------------------------------------------------

# Evalúo que realmente se ingrese un número, si no es así, se pedirá ingresarlo
# nuevamente.
while True:                                                                                              
    try:                                                                                                   
        num = int(input('Por favor, ingrese un número entero positivo'))
        if num > 0: break                                                                                
    except ValueError:                                                                                     
        print('Cuidado, debe ingresar un número entero de 3 dígitos.')
       
digitos = []     # Para almacenar los dígitos del número.
pares = []       # Para almacenar los dígitos que son pares.
suma = 0         # Para contar la suma de los dígitos pares.

# Separo los dígitos de num y los guardo en una la lista digitos    
while num>0:    
    num, dig = divmod(num, 10)
    digitos.append(dig)

# Si el dígito es par, lo guardo en la lista pares y comienzo a sumarlos en suma
for i in digitos:    
    if i%2 == 0:
        pares.append(i)
        suma += + i

if len(pares) == 1:
    print('Tiene un único dígito par: ', pares, '.')
else:
    print('Sus dígitos pares son ', pares, ' y su suma vale ', suma, '.')

# Fin del código.   