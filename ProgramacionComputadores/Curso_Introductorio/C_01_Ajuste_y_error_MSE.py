# -*- coding: utf-8 -*-
# 
# Ajuste y Error de mínimos cuadrados. Ejercicio # 01.
# ------------------------------------------------------------------------------
# Hecho por : Michael Heredia Pérez
# Fecha     : Mar/2018
# e-mail    : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------

# Librerías
import numpy as np 
import matplotlib.pyplot as plt

# Cargo el archivo .dat
X, Y = np.loadtxt("data_ej_01.dat", unpack = True)

# %% Grafico y como función de x.
plt.figure()
plt.plot(X, Y, 'b*')
plt.grid()
plt.xlabel(r'$x$', fontsize = 15)
plt.ylabel(r'$y$', fontsize = 15)
plt.show()

# Según el gráfico, asumo el grado del polinomio n.
n = 3

# Busco los coeficientes.
coeffs_3 = np.polyfit(X, Y, n)
print("The optimal coefficients are:", coeffs_3)

# Busco los valores del polinomio ajustado.
fit_3 = np.polyval(coeffs_3, X)

# Grafico el polinomio ajustado y los puntos.
plt.figure()
plt.plot(X, Y, label = 'Data')
plt.plot(X, fit_3, 'r--', label = 'Fit, N = {}'.format(n))
plt.legend(loc = 'best')
plt.xlabel(r'$x$', fontsize = 15)
plt.ylabel(r'$y$', fontsize = 15)
plt.show()

# Computo el error 'r' para n = 3.
r_3 = np.sum((fit_3 - Y)**2)
print('The error for n = 3 is:', r_3)

# %% Ahora, cambio el grado del polionomio a n = 5.
n = 5

# Busco los coeficientes.
coeffs_5 = np.polyfit(X, Y, n)
print("The optimal coefficients are:", coeffs_5)

# Ajusto los valores del polinomio.
fit_5 = np.polyval(coeffs_5, X)

# Vuelvo a hacer el gráfico con los valores para n = 5.
plt.figure()
plt.plot(X, Y, label = 'Data')
plt.plot(X, fit_5, 'r--', label = 'Fit, N = {}'.format(n))
plt.legend(loc = 'best')
plt.xlabel(r'$x$', fontsize = 15)
plt.ylabel(r'$y$', fontsize = 15)
plt.show()

# Computo el error 'r' para n = 5.
r_5 = np.sum((fit_5 - Y)**2)
print('The error for n = 5 is:', r_5)

# Fin del código.