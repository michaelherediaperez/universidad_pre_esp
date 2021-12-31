# -*- coding: utf-8 -*-
# 
# Monte Carlo para clacular pi. Ejercicio # 04.
# ------------------------------------------------------------------------------
# Hecho por : Michael Heredia Pérez
# Fecha     : Mar/2018
# e-mail    : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------

# https://www.geogebra.org/m/cF7RwK3H

# Librerías
import numpy as np
import matplotlib.pyplot as plt 

N = 10000           # Número de puntos.

# Creo una distribución uniforme de N puntos entre [-1, 1).
x = np.random.uniform(-1, 1, size = N)
y = np.random.uniform(-1, 1, size = N)

# Grafico dicha nube de puntos.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x, y, s = 1)     # s : marker size. 
ax.set_aspect('equal')
plt.show()

# distancias entre el origen y los puntos, y determino las que son menores 
# o igual a 1.
distances = np.sqrt(x**2 + y**2)  
inCircle = distances <= 1
Ncir = np.sum(inCircle)             # Cuántos puntos están dentro del círculo.

pi = 4*Ncir / N
print('Con una iteración, pi toma el valor de:', pi)

# Defino una gama de colores para inCircle puntos.
colors = np.array(['crimson']*N, dtype = str)
colors[inCircle] = 'blue'

# Ploteo nuevamente con la distinticón de color.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x, y, s = 1, color = colors)
ax.set_title(r'$\pi_{exp} = %f$' % pi)
ax.set_aspect('equal')
plt.show()

# Ahora quiero es calcular pi muchas veces y graficar los valores obtenidos en 
# un histograma.

trials = 100000     # Número de pruebas.

pi_arr = []         # Lista que almacenará los resultados de pi. 
for i in range(trials):
    x = np.random.uniform(-1, 1, size = N)      # Creo valores en x.
    y = np.random.uniform(-1, 1, size = N)      # Croe valores en y.
    distances = np.sqrt(x**2 + y**2)            # Calculo distancias.
    inCircle = distances <= 1                   # Selecciono si están adentro.
    Ncir = np.sum(inCircle)                     # Cuento cupantos son.
    pi = 4 * Ncir / N                           # Calculo pi.
    pi_arr.append(pi)                           # Lo añado a la lista.

# Grafico los resultados en un histograma.
plt.figure()
plt.hist(pi_arr, bins = 50)
plt.show()

pi_mean = np.mean(pi_arr)
print('El promedio, en pi, es: ', pi_mean)

error = abs(np.pi - pi_mean) / np.pi * 100
print('El error de este cálculo es de ', error, '%')

# Fin del código.