# -*- coding: utf-8 -*-
# 
# # Ejercicio # 13.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------
# Este programa recibe una cadena de texto y la imrpime en forma de árbol de 
# navidad.
# ------------------------------------------------------------------------------

cadena = list(input('Ingrese la frase u oración que desee: '))

# Quito los espacios del texto
while ' ' in cadena:
    cadena.remove(' ')

ren_en_lista = []      # Aquí guardo cada renglón del arbol en formato lista

# Imprimo una cantidad impar de carcateres de forma centrada, para ello sigo la
# forma del triángulo de pascal (la razón de los límites del slicing) y obtengo 
# cadenas que guardo en ren_en_lista, esto lo convierto a texto seguido con 
# .join y lo guardo en ren_en_txt. Finalmente se imprime centrado a 1/4 de la 
# longitud de cadena, con vacios ' ' a lado y lado.
for i in range(1, len(cadena)+1):
    ren_en_lista = cadena[(i-1)**2: i**2]
    ren_en_txt = ''.join(ren_en_lista)
    print(ren_en_txt.center(85, ' '))
    if ren_en_lista == []: break

# El ciclo for para cuando ya no tiene texto para organizar.
# Fin del código.