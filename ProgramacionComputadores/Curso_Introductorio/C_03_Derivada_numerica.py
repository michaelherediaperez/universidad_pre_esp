# -*- coding: utf-8 -*-
# 
# Derivada numérica de una función. Ejercicio # 03.
# ------------------------------------------------------------------------------
# Hecho por : Michael Heredia Pérez
# Fecha     : Mar/2018
# e-mail    : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------

# Librerías
import numpy as np
import matplotlib.pyplot as plt 

# Defino la función.
def function(x):
    return np.exp(-x**2)

# Defino la derivada numérica. 
def derivative(x, y):
    assert x.shape == y.shape   # Verifico que x y y tengan el mismo tamaño.

    diff_x = np.zeros_like(x)   # Del mismo tamaño de x.
    diff_y = np.zeros_like(y)   # Del mismo tamaño que y.
    
    diff_x[0:-1] = np.diff(x)   # Diferencias, queda el último espacio vacío.
    diff_y[0:-1] = np.diff(y)

    diff_x[-1] = x[-1] - x[-2]  # Arreglo el último puntos con diferencias 
    diff_y[-1] = y[-1] - y[-2]  # regresivas.

    return diff_y / diff_x
    
# Defino el arreglo de numpy.
N = np.array([10, 50, 200]) # Cantidad de puntos para evluar el ejercicio.

# Analizo para cada cantidad de puntos.
plt.figure()                    # Inicio el gráfico
for i in range(3):
    xx = np.linspace(-10, 10, N[i])    # Defino el espacio.
    yy = function(xx)                  # Evaluo la función.

    dy_n = derivative(xx, yy)          # Derivada numérica.
    dy_a = -2*xx*np.exp(-xx**2)        # Derivada analítica.

    plt.plot(xx, yy,   'b-', label = 'Función')
    plt.plot(xx, dy_n, 'r*', label = 'Derivada numérica')
    plt.plot(xx, dy_a, 'g*', label = 'Derivada analítica')

    plt.grid()
    plt.xlabel(r'$x$', fontsize = 15)
    plt.ylabel(r'$y$', fontsize = 15)
    plt.legend(loc = 'best', fontsize = 12)
    plt.title('Resultados con {} puntos'.format(N[i]))
    plt.show()

# Fin del código.