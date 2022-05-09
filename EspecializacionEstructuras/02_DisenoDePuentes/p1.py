"""
Diseño de puentes. Examen, punto 1.

By:     Michael Heredia Pérez
email:  mherediap@unal.edu.co
Fecha:  mayo/2022

Universidad Nacional de Colombia Sede Manizales
Especialización en Estructuras.
"""

# Librerías de trabajo.
import numpy as np
import matplotlib.pyplot as plt


# La probabilidad de excedencia del periodo de retoro T en un tiempo t medido 
# en años.
P = lambda T, t : 1 - (1 - (1/T))**t

# Considerando un periodo de retorno de 475 años.
T = 475  

# Se busca cuándo se supera la probabilidad de excedencia del 90%.
Pe = 0
t  = 0
while Pe <= 0.9:
    Pe = P(T, t)    # Recalculo la probabilidad de excedencia.
    t += 1          # Añado un año al contador de tiempo.

print(f"Deben transcurrir {t} años para superar una probabilidad de excedencia del 90%.")

# Creo 1.5t+1 años como dominio para el gráfico.
tt = np.linspace(0, int(1.5*t), int(1.5*t+1))   

# Se grafica la evolución de la probabilidad en función del cálculo anterior.
plt.figure()
#plt.title("Evolución de la probabilidad de excedencia para Tr=475 años.")
plt.plot(tt, P(T, tt), '-k', label="P(475, t)")

# Marco el punto: 10% en 50 años.
plt.plot(50, 0.1, '*r', label="(50 años, 10%)")

# Marco el punto: 90% en t años.
plt.plot(t, 0.9, '*b', label= f"({t} años, 90%)" )

plt.grid()
plt.xlabel("t [años]")
plt.ylabel("Probabilidad de excedencia")
plt.legend(loc=0)
plt.xlim(left=0)
plt.ylim(bottom=0)

plt.show()

# Fin :)