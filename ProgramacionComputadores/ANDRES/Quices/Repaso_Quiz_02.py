# -*- coding: utf-8 -*-
#
# Repaso Quiz #02
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Ago/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------

# %% Ejercicio 1
#    Este programa pide la altura y anchura de un rectángulo, y el caracter a 
#    utilizar para dibujarlo. Lo imprime.
#    Emplea una función de tres parámetros

# Recibo los datos y verifico
while True:
    try:
        alto = int(input('Ingrese la altura del rectángulo: '))
        ancho = int(input('Ingrese el ancho del rectangulo: '))
        parametro = input('Ingrese el carcater con el que quiere imprimir el '+
                          'rectángulo: ')
        if alto>0 and ancho>0: break
    except ValueError:
        print('Las dimensiones deben de ser números enteros. Vuela a '
              + 'ingresarlas')
        
# Defino la función que necesito        
def imprime_rectangulo(altura, anchura, caracter):
    '''Se imprime un rectángulo según las dimensiones y el caracter dado.
    
        Parámetros de entrada:
            altura: la altura del rectángulo.
            anchura: el ancho del rectángulo
            caracter: el caractere con el cual se quiere dar forma al 
                      rectángulo.
     
        Se imprime la figura.
    '''
    # Como matriz: imprimo por filas la cantidad de caracteres que me indica el 
    # ancho, separados por un espacio.
    for i in range(altura):
        print(' '.join(caracter*ancho))

print()   # Le doy un espacio al rectángulo para imprimirse         
imprime_rectangulo(alto, ancho, parametro)

# %% Ejercicio 2
#    Este programa lee el ancho de un triángulo y lo imprime con asteríscos.

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

# %% Ejercicio 3
#    Este programa recibe dos años y escribe cuántos bisiestos hay entre ambas 
#    fechas (incluidos los dos años).

# Recibo los datos de entrada: los dos años
while True:
    try:                
        year1 = int(input('Escriba un año: '))
        year2 = int(input('Escriba otro año posterior a {0}: '.format(year1)))
        if year1<year2: break
        elif year1<0 or year2<0:
            print('Los años deben ser cantidad positivas.')
    except ValueError:
        print('Los años deben de ser números enteros')
    
# DEFINO LAS FUNCIONES NECESARIAS
       
def bisiestos(a1, a2):
    '''Este función recibe dos años a1 y a2, tales que a1 < a2 y determina 
       cuántos años bisiestos hay entre ambos.
       
       Parámetros de entrada:
           a1: el primer año.
           a2: el segundo año
           
       Retorna un entero.
    '''
    bisiestos = 0       # Contador de años bisiestos.
    
    # El cuerpo de la función, un año es bisiesto cuando es divisible entre
    # 4 y si no lo es entre 100, o si lo es entre 400.
    for i in range(a1, a2+1):
        if (i%4 == 0 and i%100 !=0) or i%400 == 0:
            bisiestos += 1
    
    return(bisiestos)

# EL CUERPO PRINCIPAL
    
los_bis = bisiestos(year1, year2)  # Almacena la cantidad de años bisiestos

# Secuencia de ordenes para imprimir la respuesta
if year1 == year2:
    print('Los años ingresados son iguales')
elif bisiestos(year1, year2) == 0:
    print('Entre {0} y {1} no hay bisiestos.'.format(year1, year2))
elif bisiestos(year1, year2) == 1:
    print('Entre {0} y {1} hay un año bisiesto.'.format(year1, year2))
else:
    print('Entre {0} y {1} hay {2} años '.format(year1, year2, los_bis)
          + 'bisiestos.')
     
# %% Ejercicio 4
#    Este programa lee un número, imprime una lista asendente entre 1 y el 
#    número, y en seguida una de 1s y 2s. El número es entero.

# Recibo el número
while True:
    try:
        num = int(input('Ingrese un número entero: '))
        break
    except ValueError:
        print('El número debe de ser un entero.')

def print_list_of_numbers(num):
    '''Se recibe un númeor y devuelve una lista asencendente y otra desendiente
        en frente con los números 1 y 2 intercalados.
        
        El dato de entrada es num, y es número entero.
        
        Se imprimen dos listas.
    '''
    numeros = []    # Almacena los números de 1 al num
    lista = []      # Almacena las parejas
    
    if num>0:
        numeros = [x for x in range(1, num+1)]  
    else:
        numeros = [x for x in range(num, 2)]
                                  
    # Creo las duplas para tener los valores de las dos listas.
    for i in numeros:
        if i%2 == 0:
            lista.append((i, 2))
        else:
            lista.append((i, 1))
    
    # Imprimo las dos filas en forma de cadenas por renglones.
    for i in lista:
        print('{0}    {1}'.format(i[0],i[1]))
       
print()
print_list_of_numbers(num)