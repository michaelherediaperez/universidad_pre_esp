# QUICES

En esta sección están los quices que la asignatura, junto con algunos ejercicios preparatorios para la presentación de ellos.

## Repaso quiz #01


## Repaso quiz #02

1. Escriba una función de tres parámetros que pide la altura y anchura de un rectángulo, y el carácter a utilizar para dibujarlo.

2. Escriba una función que pida la anchura de un triángulo y lo dibuje con caracteres producto (*)

3. Escriba una función que pida dos años y escriba cuántos años bisiestos hay entre esas dos fechas (incluidos los años):

4. Escriba una función que reciba un número entero e imprima las sigueintes listas.

        Digite un número: 6

        1   1
        2   2
        3   1
        4   2
        5   1
        6   2

## Repaso quiz #03

### Numpy 

1. Generar una matriz de 7 por 9.
    - Las primeras 3 columnas de la matriz deben tener el valor 0.
    - La cuarta columna debe tener el valor 0.5, excepto por el último valor de esa columna, que tiene que ser 0.7.
    - Las otras tres columnas deben tener el valor 1.

    Luego imprimir la matriz y el promedio de la última fila.

2. La siguiente línea crea una matriz aleatoria de 5 por 5 con valores entre 0 y 1.

`matriz_aleatoria = np.random.rand(5,5)`
`print(matriz_aleatoria)`

Imprimir las posiciones (fila y columna) de los elementos de la matriz que son mayores que 0.5.

3. Los arrays a = [0, 1, 2, 3] y b = [-0.1, 1.01, 1.98, 3.15], ¿Son iguales si se considera un valor de tolerancia absoluta (diferencia entre elementos de la misma posición) inferior a 0.16?


### Pandas

Haciendo uso de la librería, lea el fichero `NucleosPoblación.csv`.

- ¿Cuántos Municipios tienen más de 100000 habitantes? ¿Cuál es la segunda ciudad más poblada?
¿Qué posición ocupa Granada en el ranking de las más pobladas?
- Escriba los nombres de los 10 municipios con menos población.
- ¿Cuántos municipios de Extremadura tienen más de 5000 habitantes?
- ¿Cuál es el municipio situado más al Norte? (Usar el valor de la coordenada "Y" que representa la
latitud en grados). Proporcione también la provincia a la que pertenece y su población.

Exporte el fichero y entréguelo como un formato .xlsx llamado 'NucleosPoblacionSinComas'.