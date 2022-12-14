"""
Graficación animada del ciclo de histéresis
por: Michael Heredia Pérez
email: mherediap@unal.edu.co
fecha: julio 15, 2022

"""


import numpy as np                 # Librería para cálculo algebráico.
import matplotlib.pyplot as plt    # Librería para graficar.
import imageio.v2 as imageio       # Librería para hacer GIFs.
import pandas as pd
import os                                   
from matplotlib.gridspec import GridSpec

# -----------------------------------------------------------------------------
# Datos.

# Leo los resultados del archivo de excel.
resultados = pd.read_excel("resultados_ensayo_muro.xlsx", sheet_name="Hoja1")
envolvente_casera = pd.read_excel("resultados_ensayo_muro.xlsx", sheet_name="Hoja2")

# Extraigo los vectores de tiempo, desplazmaineto y fuerza de los actuadores.
tt = resultados["Tiempo de ejecución"][1:].to_numpy()      # seg.
dd = resultados["Desplazamiento Actuador"][1:].to_numpy()  # mm.
ff = resultados["Fuerza Actuador"][1:].to_numpy()          # kN.

# Extraigo la corrección casera de la envolvente.
dd_env = envolvente_casera["d [mm]"].to_numpy()         # mm.
ff_env = envolvente_casera["f [kN]"].to_numpy()         # kN.

# Dimensiones del elemento y variables.
g = 9.81    # m/s^2    gravedad.
m = 611     # kg       masa del muro.
h = 2100    # mm       altura del muro (tomado de 2020 JP, Daniel y Hurtado)

# Cantidad o número de puntos del registro.
npuntos = tt.shape[0]

# -----------------------------------------------------------------------------
# Gráfico.

# Límite de los gráficos.
lim = 120

# Defino el lienzo para las gráficas y el título.
fig = plt.figure(figsize=(15, 20))
fig.suptitle("Histéresis y ciclos de carga")

# Defino la grilla: 2 filas, 3 columnas, en fig.
grilla = GridSpec(2, 3, figure=fig)

# Defino el espacio para el gráfico de la histéresis: fuerza vs desplazamiento.
ax_his = fig.add_subplot(grilla[:, 0:2])
# Defino el espacio para el gráfico desplazamiento vs tiempo.
ax_des = fig.add_subplot(grilla[0, 2])
# Defino el espacio para el gráfico de fuerza vs tiempo.
ax_fur = fig.add_subplot(grilla[1, 2])

# Con figuro los ejes para el gráfico de la histéresis.
ax_his.plot([0, 0], [-lim, lim], '-k')
ax_his.plot([-lim, lim], [0, 0], '-k')

# Para la circunferencia, defino el timpo que tarda en graficar cada punto.
time = 0.005

# Inicio el modo interactivo de matplotlib. 
plt.ion()

# Grafico el primer punto de la histéresis.
plot_his = ax_his.plot(dd[0], ff[0])[0]

# Grafico el primer punto del desplazamiento aplicado.
plot_des = ax_des.plot(tt[0], dd[0])[0]

# Grafico el primer punto de la carga asimilada.
plot_fur = ax_fur.plot(tt[0], ff[0])[0]

# filename container to create the gif.
#filenames = []

# Por cada punto del análisis (instante de tiempo).
for i in range(len(tt)+1):

    # Defino las líneas a graficar 
    #plot.set_data(ssn_t[0:i], ttn_t[0:i])
    plot_his.set_data(dd[0:i], ff[0:i])
    plot_des.set_data(tt[0:i], dd[0:i])
    plot_fur.set_data(tt[0:i], ff[0:i])    

    # Hago el gráfico
    plt.draw()
    
    # Hago pausas cada "time".
    plt.pause(time)

    # create file name and append it to a list
    #filename = f'{i}.png'
    #filenames.append(filename)
    
    # save frame
    #plt.savefig(filename)
    
# Cierro el modo interactivo
plt.ioff()


# Aseguro que los gráficos sean cuadrados.
#format_axes(fig)
#plt.tight_layout
#plt.savefig(histeresis_animada.gif)

#plt.show()

print("todo en orden")


