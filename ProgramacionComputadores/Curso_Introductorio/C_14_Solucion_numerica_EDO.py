# -*- coding: utf-8 -*-
#  
# Ecuación diferencial. Ejercicio # 13.
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
 
dy_dt = lambda y, t : t*np.sqrt(y)  # Defino la ecuación diferencial.
                                    # Cuidado con el orden de definir lambda.

tt = np.linspace(0, 2, 100)         # Defino el espacio.
y0 = 1                              # La condició inicial.

y_num = odeint(dy_dt, y0, tt)       # La solución numérica.
y = (1/16) * (tt**2 + 4) ** 2       # La solución analítica

# Grafico ambos resultados.
plt.figure()
plt.plot(tt, y_num,  '-o', label = r'$y_{numérica}$')
plt.plot(tt,     y, '--r', label = r'$y_{analítica}$')
plt.grid()
plt.legend(loc = 'best', fontsize = 15)
plt.xlabel(r'$x$', fontsize = 15)
plt.ylabel(r'$y$', fontsize = 15)
plt.show()

# Fin del código.