# -*- coding: utf-8 -*-
#
# Sustentación taller 01.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------


# %% Ejercicio 1

'''
Este programa lee una frase, la descompone en sus caracteres y los imprime 
como una sola 'palabra' organizadas algabéticamente, conviertiendo los 
espacios en el carcater '/' adjuntados al principio.

ESTE PROGRAMA NO FUNCIONA CUANDO HAY OTROS SIGNOS DE PUNTUACIÓN
'''

# Leo la cadena del usuario y reemplazo los espacios por el caracter '/'
cadena = input('Ingrese una frase u oración que desee: ').replace(' ', '/')
print()

# Casteo la cadena a una lista para así tener las letras, la organizo e imprimo
# las letras organziadas
letras = sorted(list(cadena))
letras = ''.join(letras)
print(letras)

# %% Ejercicio 2

'''
Este programa lee una frase, la descompone en sus caracteres y los imprime 
como una sola 'palabra' organizadas algabéticamente, conviertiendo los 
espacios en el carcater '/' adjuntados al principio.
'''

# Leo la cadena dada por el usuario y la casteo a una lista
cadena = list(input('Ingrese la frase u oración que desee: '))

# Cuento la cantidad de espacios
espacios = cadena.count(' ')          

# Ahora quito los espacios de la cadena
while ' ' in cadena: cadena.remove(' ')

# Organizo alfabeticamente y las preparo para imprimir
letras = sorted(list(cadena))
letras = ''.join(letras)

# Imprimo la concatenación de los carcateres '/' con la cadena organizada
resultado = espacios*'/' + str(letras)
print(resultado)

# %% Ejercicio 2 (Más específico)

'''
Este programa lee los valores (x1, y1, r1) y (x2, y2, r2), donde xi es la 
posición en x del centro de una cirfunferencia, y1 es la posición en y del 
centro de una circunferencia, y ri es el valor del radio de la circunferencia.

Analiza si un punto introducido dado (a, b) está:
a) Dentro del círculo 1
b) Dentro del círculo 2
c) Dentro de ambos círculos
d) Fuera de ambos círculos

Para cada numcondición imprime además:
a) Una matriz aleatoria d eorden axb
b) Una matriz aleatoria bxa
c) Una matriz cuadrada de orden a
d) Una matriz cuadrada de orden b
'''

# Importo las librerias
import numpy as np 

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
        
print()

# Recibo las coordenadas del punto (a,b) cerciorándome de que es un número
while True:                                                                 
    try:                                                                   
        a = int(input('Ingrese el valor en x del punto que desea calcular: '))      
        b = int(input('Ingrese el valor en y del punto que desea calcular: '))
        if a>0 and b>0 : break
        else:
            print('Para el punto, ingrese solo números enteros positivos.')
    except ValueError:
        print('Cuidado, debe ingresar valores numéricos: ')
        
punto = (a, b)              # Forma cartesiana del punto ingresado
matriz_aleatoria = []       # La matriz que debe imprimir

print()

# Evalúo si está dentro de la circunferencia 1
if (((a-x1) ** 2 + (b-y1) ** 2)<r1**2):                     
    print('El punto ', punto, ' está dentro de la circunferencia 1.')    
    matriz_aleatoria = np.random.rand(a, b).round(2)
    print(matriz_aleatoria)                   

# Evalúo si está dentro de la circunferencia 2
elif (((a-x2) ** 2 + (b-y2) ** 2)<r2**2):
    print('El punto ', punto, ' está dentro de la circunferencia 2.')
    matriz_aleatoria = np.random.rand(b, a).round(2)
    print(matriz_aleatoria) 
    
# Evalúo si está dentro de ambas circunferencias
elif (((a-x2)**2 + (b-y2)**2)<r2**2) and (((a-x1)**2 + (b-y1)**2)<r1**2):
    print('El punto ', punto, ' está dentro de ambas circunferencias.')
    matriz_aleatoria = np.random.rand(a, a).round(2)
    print(matriz_aleatoria) 

# Evalúo si no está dentro de ninguna cirfunferencia
elif (((a-x2)**2 + (b-y2)**2)>r2**2) and (((a-x1)**2 + (b-y1)**2)>r1**2):
    print('El punto ', punto, ' está fuera de ambas circunferencias.')
    matriz_aleatoria = np.random.rand(b, b).round(2)
    print(matriz_aleatoria) 

else:
    print('El punto pertenece a alguna de las circunferencias.')