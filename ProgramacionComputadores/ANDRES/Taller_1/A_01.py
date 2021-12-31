# -*- coding: utf-8 -*-
#
# Ejercicio # 01.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------
# Este programa lee un número, entero o decimal, e imprime su valor absoluto.
# Por definición, |x| = x cuando x > 0 y |x| = (-x) cuando x < 0 
# ------------------------------------------------------------------------------

# Evalúo que realmente se ingrese un número, si no es así, se pedirá ingresarlo
# nuevamente.

while True:                                                     
    try:                                                         
        num = float(input('Por favor ingrese un número: '))
        break
    except ValueError:
        print('Cuidado, debe ingresar un número.')

# Se aplica la definición 
if num<0:
    abso = -num
    print('El valor albsoluto de ', num, ' es: ', abso)
else:
    print('El valor albsoluto de ', num, ' es: ', num)

# Fin del código 