# -*- coding: utf-8 -*-
#
# Ejercicio # 02.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------
# Este programa lee un número entero y lo escribe en reversa.
# ------------------------------------------------------------------------------

# Evalúo que realmente se ingrese un número, si no es así, se pedirá ingresarlo
# nuevamente.

while True:
    try:
        num = int(input('Por favor, ingrese un número entero: '))
        break
    except ValueError:
        print('Cuidado, debe ingresar un número.')

digitos = []            # En esta lista se almacenarán los dígitos.
es_negativo = False     # Me servirá para ponerle el signo al final

# hago una excepción para cuando el num sea negativo
if num<0: 
    num = -num
    es_negativo = True

# Separo los dígitos de num y los guardo en una la lista digitos
while num>0:
    num, dig = divmod(num, 10)
    digitos.append(dig)

# El asterisco en *digitos hace que se impriman los elementos sin formato de 
# lista, y terminar con sep = '' hace que los elementos se impriman sin 
# separación
if es_negativo:
    print('El número escrito al reves sería -', *digitos, sep = '')
else:
    print('El número escrito al reves sería ', *digitos, sep = '')

# Fin del código.