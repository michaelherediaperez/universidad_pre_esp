# Implementación de una rutina de análisis no lineal de material en sólidos con el elemento finito Q8.

## Referencia.

- Bhatti, M.A. (2006). Advanced Topics in Finite Element Analysis of Structures: With Mathematica and MATLAB Computations. 8.3 Material Nonlinearity in General Solids ([Ver](https://www.amazon.com/Advanced-Topics-Element-Analysis-Structures/dp/0471648078)).


## Archivos.

Esta carpeta contiene los sigueintes archivos: 

- [Q8_NL.py](Q8_NL.py) es el programa principal, contiene la lectura de información, el ciclo de análisis, presentación de gráficas y exportación de resultados.  
- [rutinas_nl.py](rutinas_nl.py) son las rutinas de análisis no lineal de material, adaptadas del libro de Bhatti, cap 8. *Este archivo siempre debe estar en la misma carpeta de ejecución que el programa principal.* 
- [malla3.xlsx](malla3.xlsx) es un archivo guía hecho en MS EXCEL que funciona como plantilla para la lectura de la geometría del sólido, su malla de elementos finitos, configuración de carga y propiedades geométricas, elásticas y plásticas. *Este archivo siempre debe estar en la misma carpeta de ejecución que el programa principal.*
- Unos breves [Apuntes.pdf](Apuntes.pdf) con anotaciones sobre el tema implementado.


## Consideraciones.

- La función de fluencia empleada es la de von Mises $F(\sigma) = \sqrt(3J_2)-\sigma_y$, por lo que se prefiere usar la rutina de análisis no lineal con materiales metálicos. $J_2$ el segundo invariante de esfuerzos del estado de esfuerzos, $\sigma_y$ el esfuerzo de fluencia del estado previo del material.
- Las rutinas de análisis no lineal SÓLO consideran un modelo de endurecimiento isotrópico, factible en materiales metálicos.  
- Se toma la regla de flujo asociativa (*Associative flow rule*), es decir, la función de fluencia $F$ es la misma función potencial plástico $Q$.
- Se emplea el elemento finito isoparamétrico de 8 nodos.


## Errores

- Hay un error de programación que impide el desarrollo del ciclo de incrementos de carga, una vez solucionado el código debería correr sin problemas.


## Por hacer.

- Animar los incrementos de desplazamientos, deformaciones y esfuerzos en la malla según los incrementos de carga. 
- Animar la elipse de fluencia de von Mises $\sigma_1^2 + \sigma_2^2 - \sigma_1\sigma_2 = \sigma_y^2 $ para ver el fenómeno del endurecimiento isotrópico 


## Se podría hacer.

- El archivo [rutinas_nl.py](rutinas_nl.py) está generalizado para un sólido 3D, pero con algunas variables por defecto se simplifica a tensión plana. Se puede reutilizar en otro programa para hacer el análisis de un sólido 3D.