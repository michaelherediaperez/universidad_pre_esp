# -*- coding: utf-8 -*-
# 
# # Ejercicio # 07.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------
# Este programa lee un entero positivo y determina si existen en él dígitos 
# repetidos, además indica cuáles son. 
# ------------------------------------------------------------------------------

# Evalúo que realmente se ingrese un número, si no es así, se pedirá ingresarlo
# nuevamente.
while True:
    try:
        num = int(input('Por favor, ingrese un número entero: '))
        if num>0: break
    except ValueError:
        print('Cuidado, debe ingresar un número.')

digitos = []            # En esta lista se almacenarán los dígitos.
repetidos = []          # En esta lista se almacenan los digitos repetidos
unicos = []             # En esta lista se almacenan los dígitos únicos

# Separo los dígitos de num y los guardo en la lista digitos
while num>0:
    num, dig = divmod(num, 10)
    digitos.append(dig)

# Hago la selección de los repetidos y los únicos. Los únicos son de control
# Todo dígito se guarda primero en la lista unicos, y cuando haya otro dígito
# igual, este ya 'estará en unicos, por lo que se guardará en repetidos.
for i in digitos:
    if i not in unicos: 
        unicos.append(i)
    else:
        if i not in repetidos: repetidos.append(i)

print('La cantidad de dígitos repetidos es: ', len(repetidos))
print('Los dígitios repetidos son: ', repetidos)

# Fin del código.