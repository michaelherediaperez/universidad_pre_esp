# -*- coding: utf-8 -*-
#  
# Ecuación diferencial. Ejercicio # 14.
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
 
dx_dt = lambda x, t : 5*x - 3      # Defino la ecuación diferencial.
                                   # Cuidado con el orden de definir lambda.

tt = np.linspace(2, 3, 100)        # Defino el espacio.
x2 = 1                             # La condició inicial.

x_num = odeint(dx_dt, x2, tt)      # La solución numérica.
x = (2/5) * np.exp(5*(tt-2))+(3/5) # La solución analítica

# Grafico ambos resultados.
plt.figure()
plt.plot(tt, x_num,  '-o', label = r'$y_{numérica}$')
plt.plot(tt,     x, '--r', label = r'$y_{analítica}$')
plt.grid()
plt.legend(loc = 'best', fontsize = 15)
plt.xlabel(r'$x$', fontsize = 15)
plt.ylabel(r'$y$', fontsize = 15)
plt.show()

# Fin del código.