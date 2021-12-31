# -*- coding: utf-8 -*-
#  
# Ecuación diferencial. Ejercicio # 12.
# ------------------------------------------------------------------------------
# Hecho por : Michael Heredia Pérez
# Fecha     : Mar/2018
# e-mail    : mherediap@unal.edu.co
# Universidad Nacional de Colombia, sede Manizales.
# ------------------------------------------------------------------------------

# Las Librerías.
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint

dy_dx = lambda y, x : x - y     # Defino la ecuación diferencial.
                                # Cuidado con el orden de definir lambda.

xx = np.linspace(0, 2, 100)     # Defino el espacio.
y0 = 1                          # La condició inicial.

y_num = odeint(dy_dx, y0, xx)   # La solución numérica.
y = xx - 1 + 2*np.exp(-xx)      # La solución analítica

# Grafico ambos resultados.
plt.figure()
plt.plot(xx, y_num,  '-o', label = r'$y_{numérica}$')
plt.plot(xx,     y, '--r', label = r'$y_{analítica}$')
plt.grid()
plt.legend(loc = 'best', fontsize = 15)
plt.xlabel(r'$x$', fontsize = 15)
plt.ylabel(r'$y$', fontsize = 15)
plt.show()

# Fin del código.