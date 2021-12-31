# -*- coding: utf-8 -*-
#
# Quiz #02
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Ago/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------

# %% Ejercicio 1
#    Este programa lee el ancho de un triángulo y lo imprime con asteríscos.

# Pido al usuario la dimensión dleanco que desea para el triángulo
while True:
    try:
        ancho = int(input('Ingrese el ancho del triángulo: '))
        if ancho > 0: break
        else: print('Las dimensiones geométricas sdeben de ser positivas.')
    except ValueError:
        print('La dimensión pedida debe ser un dato entero.')
        
# Defino la función que realiza el trabajo
def triangulo(anchura):
    '''
        Se toma como dato de entrada la anchura de un triángulo, y se imprime 
        dicha figura con asteriscos (*). Por ejemplo, si anchura = 3 se imprime
        
        *
        **
        ***
        **
        *
    '''
    for i in range(1, anchura + 1):
        print(i*'*')   
    for i in range(1, anchura):
        print((anchura-i)*'*')

# Imprimo el triángulo
print()
triangulo(ancho)

# %% Ejercicio 2
#    Este programa es un juego, el programa lanza una serie de sumas de números
#    entre 1 y 100 y el usuario deberá acertarlas, una vez completadas 5 sumas 
#    correctamente el programa se cerrará y el juego habrá terminado.
#    Se emplea la librería random.


# Importo la función randint de la librerí­a random
from random import randint

aciertos = 0              # El usuario debe hacer 5 aciertos

# Defino mi función para hacer las sumas aleatorias.
def sumas_aleatorias():
    '''
        Esta función da al usuario una suma y le pide que la resuelva, y 
        advierte cuando se ha acertado y cuando no.
        
        La función retorna True o False según acertado o erroneo el resultado.
    '''
    
    # Genero dos números aleatorios entre 1 y 100.
    x = randint(1, 101)
    y = randint(1, 101)
    
    # Analizo si es valido el dato ingresado.
    while True:
        try:
            # Lanzo al usuario la pregunta y pido su respuesta.
            suma = int(input('{0} + {1} = '.format(x, y)))
            break
        except ValueError:
            print('Ojo, debe ingresar un número entero positivo ya que las '
                  + 'operaciones son con números enteros positivos.')
    
    # Comparo el resultado de la suma con la suma ingresada.
    if x + y == suma:
        return True
    else:
        return False
        
# EL CUERPO DEL PROCEDIMIENTO EMPLEANDO LA FUNCIÓN sumas_aleatorias.
while aciertos != 5:
    if sumas_aleatorias(): 
        aciertos += 1
        print('RESPUESTA CORRECTA. ACIERTOS -> {0} '.format(aciertos))
    else:
        print('RESPUESTA INCORRECTA, VA OTRA.')

print()
print('FIN DEL JUEGO.')