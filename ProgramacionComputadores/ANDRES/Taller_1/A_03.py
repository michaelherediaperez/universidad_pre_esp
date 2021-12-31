# -*- coding: utf-8 -*-
# 
# # Ejercicio # 03.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------
# Este programa lee un número de 3 dígitos y determina si el número ocho (8) es 
# un dígito del número ingresado.
# ------------------------------------------------------------------------------

# Evalúo que realmente se ingrese un número, si no es así, se pedirá ingresarlo
# nuevamente.

while True:                                 
    try:                                                                                              
        num = int(input('Por favor, ingrese un número entero de tres dígitos, '
                        + 'por ejemplo el 678: ')) 
        if -999 <= num <= 999: break                
    except ValueError:
        print('Cuidado, debe ingresar un número entero de 3 dígitos.')
     
digitos = []            # Para almacenar los dígitos del número.
hay_un_8 = False        # Para contar cuantas veces el 8 es dígito del número.

# Por si el número es negativo
if num < 0: num = -num

# Separo los dígitos de num y los guardo en una la lista digitos   
while num>0:      
    num, dig = divmod(num, 10)
    digitos.append(dig)

# Cuento las veces que el 8 esté en los dígitos.
if 8 in digitos: hay_un_8 = True

if hay_un_8:           
    print('Ocho (8) es un dígito del número ingresado.')     
else:                                                       
    print('Ocho (8) no es un dígito del número ingresado.')

# Fin del código