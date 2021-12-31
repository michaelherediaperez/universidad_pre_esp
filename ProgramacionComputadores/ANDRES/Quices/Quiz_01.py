# -*- coding: utf-8 -*-
#
# Quiz #01
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Ago/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------

# %% Primer ejercicio.
#    Este programa lee un DNI y le asocia una letra según el banco de entradas.

# Leo el DNI y lo almaceno
DNI = eval(input('Ingrese su DNI, sin comas, sin puntos, sólo números: '))

#  Creo las relaciones de los números con su respectiva letra
numeros = [i for i in range(0, 23)]
letras = [a for a in 'TRWAGMYFPDXBNJZSQVHLCKE']

residuo = DNI % 23             # Almacena el residuo
la_letra = []                  # Almacena la letra           

# Busco la letra correspondiente
for i in numeros:
    if i == residuo: la_letra.append(letras[i])

# Casteo el DNI a una lista para concatenarlo con la_letra
DNI = list(str(DNI))
DNI_letra = DNI + la_letra    # El resultado

# Imprimmo el resultado juntando las entradas de DNI_letra
print('Su DNI (con letra) es: ', ''.join(DNI_letra))
    

# %% Segunda pregunta. 
#    Este programa lee dos números enteros y devuelve una lista con los números 
#    consecutivos entre número y número.


# Doy la idea dle código y leo los números
print('LISTA DE UNA VALOR A OTRO')
num_1 = int(input('Escriba el número entero inicial: '))
num_2 = int(input('Esciba el número entero final: '))

lista = []  # en esta lista almacenaré la secuencia

# Analizo las posibilidades que hay para generar la lista entre los números
if num_1 < num_2:
    lista = [i for i in range(num_1, num_2 +1)]
    print(lista)
elif num_2 < num_1:
    lista = [i for i in range(num_2, num_1 +1)]
    print(lista)
else:
    print('Son el mismo número')