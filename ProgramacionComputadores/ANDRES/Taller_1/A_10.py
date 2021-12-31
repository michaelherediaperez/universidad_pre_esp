# -*- coding: utf-8 -*-
# 
# # Ejercicio # 10.
# ------------------------------------------------------------------------------
# Escrito por : Michael Heredia Pérez
# Fecha       : Jun/2019
# e-mail      : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------
# Este programa lee una cadena de texto y la imprime en mayúsculas. Esta apoyado 
# en la codificación UNICODE de cada letra.
# El mejor ejemplo para comprobar el código es ingresar:
# 'Hola, me llamo Michael, ¿Tienes 1 o 456 segundos? ¡Quiero hablarte de "ñames" 
# y de pingüinos!'
# ------------------------------------------------------------------------------

cadena = list(input('Ingrese la oración en minúsculas que desea capitalizar: '))

# Defino un banco de caracteres donde buscar
entra = 'abcdefghijklmnñopqrstuvwxyzáéíóúü ,.¿?¡!:;ABCDEFGHIJKLMNÑOPQRSTUVWXYZH\
0123456789"-_{[]()=#$%&/\^°|}'
sale = 'ABCDEFGHIJKLMNÑOPQRSTUVWXYZÁÉÍÓÚÜ ,.¿?¡!:;ABCDEFGHIJKLMNÑOPQRSTUVWXYZ0\
123456789"-_{[]()=#$%&/\^°|}'

# Defino unas listas donde tenga como entradas los caracteres de lo que puede 
# entrar y lo que debe salir, respetándo su orden.
entrada = [ord(i) for i in entra]
salida = [ord(i) for i in sale]

cad_codif = []      # Guarda el código UNICODE de cada caracter
cad_capital = []    # Guarda cada caracter capitalizado
indice = 0          # Sirve de guía para comparar minusculas y mayusculas

# Tomo cada letra de cadena y la guardo en una lista como su código UNICODE
for letra in cadena:
    cad_codif.append(ord(letra))

# Tomo el UNICODE del caracter que entra en la lista entrada, y busco cuál es su
# homologo en la lista salida, lo convierto de UNICODE a caracter común.
for i in cad_codif:
    if i in entrada:
        indice = entrada.index(i)
        cad_capital.append(chr(salida[indice]))
  
resultado = "".join(cad_capital)
print(resultado)

# Fin dle código.