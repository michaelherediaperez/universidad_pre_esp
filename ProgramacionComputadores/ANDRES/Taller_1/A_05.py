# -*- coding: utf-8 -*-
# 
# # Ejercicio # 05.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------
# Este programa lee un número y determina la cantidad de dígitos que son cinco y 
# que están consecutivos.
# El número 785265557 tiene 3 cincos consecutivos. 
# ------------------------------------------------------------------------------

# Evalúo que realmente se ingrese un número, si no es así, se pedirá ingresarlo
# nuevamente.
while True:                                                                                              
    try:                                                                                                   
        num = int(input('Por favor, ingrese un número entero: '))
        break                                                                                            
    except ValueError:                                                                                     
        print('Cuidado, debe ingresar un número entero.')

digitos = []          # Para almacenar los dígitos del número.
cinco = ''            # Lista con los 5 consecutivos
cinco_L = []          # Almacena los cincos consecutivos
cant_5 = 0            # Número de cincos que están seguidos
 
# Para cuando el número sea negativo
if num < 0: num = -num

# Separo los dígitos de num y los guardo en una la lista digitos    
while num>0:    
    num, dig = divmod(num, 10)
    digitos.append(dig)

# Selecciono los cincos consecutivos como una cadena. OJO CON LOS +=
for i in digitos:
    if i == 5: 
        cinco += str('5')
    else: 
        cinco += ' '

# Parto cinco en las 'palabras' y lo organizo para tener de último el mayor 
# conjunto de cincos seguidos en el número.
cinco_L = sorted(cinco.split())

# Mido cuántos son los cincos seguidos y respondo
for i in cinco_L:
    if len(i) > 1: cant_5 += len(i)
           
if cant_5 == 0:
    print('No hay cincos seguidos en este número.')
else:
    print('Hay ', cant_5, ' cincos seguidos en el número.')

# Fin del código.