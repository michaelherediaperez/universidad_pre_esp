# -*- coding: utf-8 -*-
# 
# # Ejercicio # 17b.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------
# Este programa lee una lista y determina cuál es el segundo elemento más grande 
# de dicha lista según el número de espacios que ocupa cada entrada.
# COMPARANDO SOLO NÚMEROS.
# ------------------------------------------------------------------------------

print('A continuación, ingrese los valores que desea comparar separados por '
      + 'un espacio:')

# Recibo los datos y analizo que sean válidos para el formato
while True:
    try:
        entradas = list(map(float, input().split()))
        break
    except ValueError:
        print('Este programa sólo soporta valores numéricos. Por favor, digite'
              + ' los valores nuevamente.')

# Elimino el número mayor de la lista suminstrada por el usuario. 
for i in entradas:
    if i is max(entradas):
        entradas.remove(i)

# Imprimo el máximo que queda luego de la eliminación anterior.
print('El segundo número más grande es: ', max(entradas))

# Fin del código.