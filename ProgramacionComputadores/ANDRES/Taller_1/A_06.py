# -*- coding: utf-8 -*-
# 
# # Ejercicio # 06.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------
# Este programa lee un número entero y determina si la suma de sus dígitos es un
# número de Fibonacci.
# ------------------------------------------------------------------------------

# Evalúo que realmente se ingrese un número, si no es así, se pedirá ingresarlo
# nuevamente.
while True:                                                                                              
    try:                                                                                                   
        num = int(input('Por favor, ingrese un número entero: ')
        break                                                                                            
    except ValueError:                                                                                     
        print('Cuidado, debe ingresar un número entero.')

max_fib = num**num    # cantidad de numeros de fibonacci que voy a generar    
digitos = []          # Para almacenar los dígitos del número
fibonacci = []        # Para almacenar los números de Fibonacci
suma = 0              # almaceno la suma de los dígitos
a, b = 0, 1           # Valores inciales para trabajar Fibonacci

# Genero la serie de Fibonacci hasta el max_fib - ésimo número y los almaceno en
# la lista fibonacci.
while a<max_fib:
    fibonacci.append(a)
    a, b = b, a+b

# Para cuadno el número sea negativo
if num < 0: num = -num

# Separo los dígitos de num y los guardo en una la lista digitos 
while num>0:
    num, dig = divmod(num, 10)
    digitos.append(dig)

# Calculo la suma de los dígitos
for i in digitos:
    suma += i

if suma in fibonacci:
    print('La suma de sus dígitos es un número de Fibonacci.')
else:
    print('La suma de sus dígitios no es un número de Fibonacci.')

# Fin del código.