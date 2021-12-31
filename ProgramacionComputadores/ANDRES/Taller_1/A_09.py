# -*- coding: utf-8 -*-
# 
# # Ejercicio # 09.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------
# Este programa lee dos vectors (x1, y1, r1) y (x2, y2, r2), donde xi es la 
# posición en x del centro de una cirfunferencia, y1 es la posición en y del 
# centro de una circunferencia, y ri es el valor del radio de la circunferencia.

# Analiza si un punto introducido dado (a, b) está:
# a) Dentro del círculo 1
# b) Dentro del círculo 2
# c) Dentro de ambos círculos
# d) Fuera de ambos círculos
# ------------------------------------------------------------------------------

# Evalúo que realmente se ingrese un número, si no es así, se pedirá ingresarlo
# nuevamente.
while True:                                                                                   
    try:                                                                                       
        x1 = float(input('Ingrese la coordenada en x del centro del primer '
                        + 'círculo: '))         
        y1 = float(input('Ingrese la coordenada en y del centro del primer '
                        + 'círculo: '))       
        r1 = float(input('Ingrese el valor del radio del primer círculo: '))                  
        x2 = float(input('Ingrese la coordenada en x del centro del segundo '
                        + 'círculo: '))
        y2 = float(input('Ingrese la coordenada en y del centro del segundo '
                        + 'círculo: '))
        r2 = float(input('Ingrese el valor del radio del segundo círculo: '))
        break
    except ValueError:
        print('Cuidado, debe ingresar valores numéricos. ')

# Recibo las coordenadas del punto (a,b) cerciorándome de que es un número
while True:                                                                 
    try:                                                                   
        a = float(input('Ingrese el valor en x del punto que desea calcular: '))      
        b = float(input('Ingrese el valor en y del punto que desea calcular: '))
        break
    except ValueError:
        print('Cuidado, debe ingresar valores numéricos: ')
        
punto = (a, b)

# Evalúo las condiciones pedidas
if (((a-x1) ** 2 + (b-y1) ** 2)<r1**2):                     
    print('El punto ', punto, ' está dentro de la circunferencia 1.')                        
elif (((a-x2) ** 2 + (b-y2) ** 2)<r2**2):
    print('El punto ', punto, ' está dentro de la circunferencia 2.')
elif (((a-x2) ** 2 + (b-y2) ** 2))<r2**2) 
        and (((a-x1) ** 2 + (b-y1) ** 2)<r1**2):
    print('El punto ', punto, ' está dentro de ambas circunferencias.')
elif (((a-x2) ** 2 + (b-y2) ** 2)>r2**2) 
        and (((a-x1) ** 2 + (b-y1) ** 2)>r1**2):
    print('El punto ', punto, ' está fuera de ambas circunferencias.')
else:
    print('El punto ', punto, ' hace parte de alguna circunferencia.')

# Fin del código.