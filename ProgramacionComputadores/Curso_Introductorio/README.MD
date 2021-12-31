# Curso Introductorio de programación en PYTHON

La siguiente es la lista de ejercicios que fueron propuestos en el curso, de los cuales se encuentra la solución en esta sección del repositorio.

# C_01

Un polinomio de grado *n* se define como


donde las constantes a0, ..., an se llaman coeficientes del polinomio.

Ajustar un polinomio consiste en encontrar los valores de ai que minimice el error. La función `numpy.polyfit` calcula los coeficientes que minimice el error cuadrado:

El archivo data_ej_01.dat contiene 2 columnas, que corresponden a *x* y *y*, respectivamente.

1. Graficar *y* como función de *x*.
2. Fijar el grado del polinomio a *n*.
3. Encontrar los coeficientes ai, por medio de la función `numpy.polifit`.
4. Graficar en forma de *scatter* los puntos y con una línea punteada el ajuste obtenido.
5. Cambiar el grado del polinomio y volver a realizar los pasos 4 y 5.
6. Computar el error cuadrado *r*.

# C_02

Resolver el siguiente sistema de ecuaciones lineales:


# C_03

Calcular una derivada numérica es una tarea común cuando se trabaja con datos. Uno de los métodos empleados es con el método de diferencias finitas.


Este método se fundamenta bajo la definición de derivada




Numéricamente se puede caclular mediante las diferencias entre los datos en *y* dividida la diferencia entre los datos en *x*. Para esto, existe la función `numpy.diff` que calcula la diferencia consecutiva de un arreglo. Sin embargo, es necesario hacer una corrección para calcular la derivada en el último punto, en cuyo caso se puede emplear diferencias finitas en retroceso para calcular tal valor.

La función `derivative` toma dos arreglos, *x* y *y* de la misma dimensión y devuelve un arreglo con la misma dimensión de *x* y *y*.

1. Definir la función.
2. Definir un arreglo de numpy *x* desde un valor inicial hasta un valor final con una cantidad *N* de puntos.
3. Calcular *y* como `function(x)`.
4. Graficar *y* como función de *x* mediante una línea sólida.
5. Calcular la derivada de *y* y asignarla en una variable *dy*.
6. Computar la derivada analítica de la función definida.
7. Graficar en la misma imagen *y*, *dy* y la derivada analítica como funciones de *x*.
8. Variar el valor de *N* y observar el comportamiento.


