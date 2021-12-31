# -*- coding: utf-8 -*-
# 
# # Ejercicio # 19.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, Manizales field.
# ------------------------------------------------------------------------------
# Este programa lee una lista de números e identifica la sublista más larga 
# dentro de ella que se encuentra ordenada asendentemente.
# ------------------------------------------------------------------------------

print('A continuación, ingrese los valores separados por un espacio y termine ' 
      + 'con un cero:')

# Evalúo que el dato ingresado sea un número, sino pido que lo vuelva a 
# ingresar
while True:
    try:
        lista = list(map(float, input().split()))
        break
    except ValueError:
        print('Este programa sólo soporta valores numéricos. Por favor, digite'
              + ' los valores nuevamente.')

sub = ''    # Almacena todas las listas como caracteres
sub_L = []  # Almacena las listas como listas

# Hago comparaciones entre los números y guardo en una cadena aquellos que 
# están sucesivos en orden asendente, los que no, los guardo como un espacio en
# blanco.
for i in range(len(lista)-1):
    if lista[i] < lista[i+1]:
        sub += str(i)
    else:
        sub += ' '

# Parto la cadena con los consecutivos ascendentes, y la organizo de menor a
# mayor en una lista
sub_L = sorted(sub.split())

# Imprimo la última entrada (la sucesión más grande) y la decoro como cadena.
print('La sublista consecutiva más grande es [', ', '.join(sub_L[-1]),']')

# Fin del código.