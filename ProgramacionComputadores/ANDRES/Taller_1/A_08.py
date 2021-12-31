# -*- coding: utf-8 -*-
# 
# # Ejercicio # 08.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------
# Este programa itera a través de los primeros cien enteros positivos, buscando 
# los múltiplos de tres, imprimiéndolos y almacenándolos en una lista hasta 
# encontrar los primeros 15 de ellos. Luego, continúa iterando en busca de los 
# múltiplos de 4 y los almacena en otra lista.

# Un número x es múltiplo de m o n, cuando en x/m o x/n el residuo es cero, 
# respectivamente.
# ------------------------------------------------------------------------------

cien_num = list(range(1,101))      # Genero los números de 1 a 100
mul_3 = []                         # Aquí contengo los múltiplos de 3 (15)
mul_4 = []                         # Aquí contengo los múltiplos de 4

# Itero en la lista cien_num, y paro cuando tenga 15 números en mul_3
for i in cien_num:                 
    if (i % 3 == 0):        
        mul_3.append(i)
        if len(mul_3) == 15:                
            break

# Itero desde el número siguiente al quinceavo múltiplo de 3, hasta el límite                                          
for j in range(int(mul_3[-1]) + 1, cien_num[-1]):   
    if (j % 4 == 0):                    
        mul_4.append(j)                 


print('Los primeros 15 múltiplos de 3, entre 1 y 100, son:\n\n', mul_3)
print()
print('Los múltiplos de 4, luego de ', mul_3[-1], ' hasta 100, son:\n\n', mul_4)

# Fin del código.