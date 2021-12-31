# TALLER 1 (20%): CADENAS, LISTAS Y ESTRUCTURAS DE CONTROL

## NOTAS

- No convertir los números a cadenas para poderlos descomponer.
- Cuando se pregunta por ejemplo un número entero de tres dígitos, se está refiriendo a los números enteros -999, -998, ..., -101, -100, 100, 101, ..., 998 y 999.
- Cuando se pregunta por ejemplo un número entero de máximo tres dígitos, se está refiriendo a los números enteros -999, -998, ..., -2, -1, 0, 1, 2, ..., 998 y 999.
- Los números primos son los números naturales 2, 3, 5, 7, 11, 13, 17, 19, 23, ...
- La serie de Fibonacci está dada por: 0, 1, 1, 2, 3, 5, 8, 13, 21, ...
- Tener en cuenta que muchos de los ejercicios solicitan cálculos sobre los números enteros; no sobre los números naturales; por lo tanto, es necesario tener en cuenta los números enteros negativos, cuando sea del caso.
- Se dice que 3, 4, 5 y 6 son los números enteros comprendidos entre el 3 y el 6.
- Del número 349857 se puede decir que el 3 es su primer dígito, el 4 es su segundo dígito, el 5 es su penúltimo dígito y el 7 es su último dígito.
- No utilizar el método count() de listas, a menos que se diga lo contrario.
- Desarrollar algoritmos con el mayor grado de generalidad posible.


## DECISIONES Y CICLOS

1. Sin utilizar el comando abs, escribir un programa que calcule e imprima el valor absoluto de cualquier número (sea entero o decimal).

2. Leer un número entero y escribirlo en reversa. Por ejemplo, 461975 será 579164. No utilizar comandos de ordenamiento.

3. Leer un número entero de 3 dígitos y determinar si contiene el dígito 8.

4. Leer un número entero positivo y determinar la suma de sus dígitos pares. Por ejemplo, en el número 124, los dígitos pares son el 2 y el 4 y su suma vale 6.

5. Leer un número entero y determinar el número de cincos que están consecutivos en un número. Por ejemplo, en 5512355551 hay 6 cincos consecutivos. Ahora, en 234555234512 solo hay 3 cincos consecutivos.

6. Leer un número entero y determinar si la suma de sus dígitos es un número de Fibonacci.

7. Leer un número entero positivo y determine si existen en el número dígitos repetidos. Reporte cuales son los dígitos repetidos.

8. Iterar a través de los primeros cien enteros positivos, buscando los múltiplos de 3 e imprimiendo y almacenándolos en
una lista hasta encontrar los primeros 15 de ellos. Una vez encontrados, continuar iterando en busca de los múltiplos
de 4 y almacenarlos en otra lista.

9. Hacer un programa que lea las coordenadas (x1,y1,r1) y (x2,y2,r2) que corresponden al centro y al radio de dos círculos.
Lea un punto de coordenadas (a,b). Determina si (a,b) está contenido: a) dentro del círculo 1; b) dentro del círculo 2;
c) dentro de ambos círculos; d) fuera de ambos círculos. Recordemos que el círculo es el lugar geométrico de los puntos
del plano cuya distancia a otro punto fijo, llamado centro, es menor o igual que una cantidad constante, llamada radio.


## CADENAS

10. Sin utilizar el método .upper() para cadenas, escribir un programa que lea una cadena de texto y la imprima en
mayúscula. Tenga en cuenta que el programa debe aceptar las letras con tilde, la ñ y la ü.

11. Escribir un programa que reciba una cadena de texto y reporte:
a) cuántas letras vocales en mayúscula se entraron;
b) cuántas letras con tilde se entraron (minúsculas y mayúsculas);
c) cuántos dígitos se entraron,
d) cuántos espacios se entraron;
e) cuántas palabras reservadas se entraron.
Utilizar para ello los métodos de cadenas especializados como .isupper(), .isdigit(), etc. y la palabra reservada in.

12. Hacer un programa que lea dos palabras (podrían estar en mayúscula o minúscula) y determine cual está primero en
el diccionario. El programa debe soportar letras con tilde, la ü y la ñ.

13. Escribir un programa que lea una cadena de texto y la imprima como un triángulo. Por ejemplo, si la cadena entrada
es: "Es una locura odiar a todas las rosas porque una te pinchó: renunciar a todos tus sueños porque uno de ellos no
se realizó - El Principito", el programa debe imprimir:

                 E
                sun
               alocu
              raodiar
             atodaslas
            rosasporque
           unatepinchó:r
          enunciaratodost
         ussueñosporqueuno
        deellosnoserealizó-
            ElPrincipito

Observe que la última cadena queda centrada con respecto a la base del triángulo.

14. Leer una cadena de texto y organice alfabéticamente cada una de las letras que la componen, repitiendo cada una
tantas veces como se encuentra. Por ejemplo, la cadena ‘tarea importante’ será ‘aaaeeimnoprrttt’. (Note que no se
incluyen los espacios).


## LISTAS

15. Leer una lista de números enteros y enumerar cuantos elementos se repiten exactamente dos veces. Utilizar para tal
fin el método count() de listas.

16. Leer una lista de números ya ordenados de forma ascendente y verificar que dicha lista está ordenada. Luego, leer un
número e insertarlo en la lista en la posición que le corresponde a dicho número.

17. Leer una lista y determinar cuál es el segundo elemento más grande de dicha lista. No utilice ningún tipo de algoritmo
de ordenamiento para realizar este ejercicio.

18. Leer una matriz cuadrada A, obtener su matriz triangular superior asociada U y determinar qué fila de U contiene la
mayor cantidad de unos.

19. Leer una lista e identifique la sublista más larga dentro de dicha lista que se encuentra ordenada ascendentemente.
Por ejemplo, en la lista [5, 6, 5, 1, 2, 6, 9, 2, 4, 1] la mayor sublista ordenada es la [1, 2, 6, 9].

20. Hacer un programa que lea una matriz y la rote 90 grados en sentido antihorario. Imprimir la matriz resultante. Por
ejemplo, dada la matriz

        62 42 70 66 10
        35 10 4 92 94
        33 37 5 100 19
        79 44 72 53 91

la matriz solicitada es la:

        10 94 19 91
        66 92 100 53
        70 4 5 72
        42 10 37 44
        62 35 33 79

Repita el mismo ejercicio utilizando el módulo numpy.