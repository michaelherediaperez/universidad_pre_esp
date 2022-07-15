"""
Punto 01, parte 2 trabajo final. Histéresis de un muro.

Por: Michael Heredia Pérez
email: mehrediap@unal.edu.co
fecha: marzo/2022

Universidad Nacional de Colombia sede Manizales
Diseño Sismoresistente de Concreto Reforzado.
"""

# Librerías.
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from scipy import signal, integrate
from matplotlib.gridspec import GridSpec
import os

# -----------------------------------------------------------------------------
# Funciones.

def check(num): 
    """Verifica si un número es par o impar."""
    if num & 1 == 0:
        return "even"   # par
    else:
        return "odd"    # impar

def RegresionLineal(xx, yy):
    """Calcula la regresión lineal (y=a+bx) y coeficiente de pearson (r) de un 
    conjunto de datos."""

    # Número de puntos.
    n = xx.shape[0]

    # Pendiente (m) e intercepto (b).
    
    m = (n * np.sum(xx*yy) - np.sum(xx)*np.sum(yy) ) / (n * np.sum(xx**2) - np.sum(xx)**2)
    b = np.mean(yy) - m*np.mean(xx)

    # Línea de regresión.
    yb = xx*m+b

    # Cálculo del factor de correlación R^2.
    Sr = np.sum( (yy - yb)**2 )
    St = np.sum( (yy - np.mean(yy))**2 )
    R2 = (St-Sr)/St
    R = R2**0.5

    return R2, R, m, b

def format_axes(fig):
    """Formato para los gráficos de fuerza contra desplazamiento."""
    for i, ax in enumerate(fig.axes):
        ax.set_xlabel("d [mm]")
        ax.set_ylabel("F [kN]")
        ax.legend()

def GraficoParejas(xp, yp, xn, yn, xlabel, ylabel, title, n):
    """Para hacer gráficos individuales, dados los arreglos de datos, nombre 
    para sus ejes y el label que tendrá la curva."""
    
    fig, ax = plt.subplots()
    fig.suptitle(title)
    
    ax.plot(xp, yp, '*--r', label="Envolvente positiva")
    ax.plot(xn, yn, '*--b', label="Envolvente negativa")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(f"Resultados_G3_HisteresisMuro/{n}.png")

def GraficoSimple(xp, yp, xlabel, ylabel, title, label, n):
    """Para hacer gráficos individuales, dados los arreglos de datos, nombre 
    para sus ejes y el label que tendrá la curva."""
    
    fig, ax = plt.subplots()
    fig.suptitle(title)
    
    ax.plot(xp, yp, '*--r', label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(f"Resultados_G3_HisteresisMuro/{n}.png")

def ProcesoLinealizacion(puntos, xx, yy, inicio):
    """Automatiza el proceso de linealizar rangos elásticos en una curva física
    de datos."""

    npuntos = puntos.shape[0]

    # Espacio para almacenar el valor de R2 y de m,b de las interpolaciones.
    rr2 = np.zeros(npuntos)
    mm  = np.zeros(npuntos)
    bb  = np.zeros(npuntos)

    # Las muestras comienzan con los primeros 50 puntos.
    for i in puntos[inicio:]:
        
        # Tomo un subconjunto de (dd, ff)
        x_p = xx[:i]; y_p = yy[:i] 
        
        # Calculo el coeficiente de regresión R^2.
        r2, r, m, b = RegresionLineal(x_p, y_p)

        # Almaceno los valores de R2, m, b para después.
        rr2[i] += r2
        mm[i]  += m
        bb[i]  += b

    return rr2, mm, bb

# -----------------------------------------------------------------------------
# Carpeta para resultados.

# Check whether the specified path exists or not and if not create it.
path = "./Resultados_G3_HisteresisMuro"

if not os.path.exists(path):
    os.mkdir(path) 
    
print("\nGRUPO 3 - HISTÉRESIS DEL MURO")
print(f"\nGráficas y excel con resultados se guardarán en: {path}\n")

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
# Suavizado con un butter y filtrado. 

sos = signal.butter(4, 0.125, output='sos')
ff_sos = signal.sosfiltfilt(sos, ff)

# -----------------------------------------------------------------------------
# Cargas máximas y rigideces inelásticas.

index_carga_max = np.argmax(ff_sos)
carga_max = ff_sos[index_carga_max] # Carga máxima positiva alcanzada.
despl_max = dd[index_carga_max]     # Desplazamiento para la carga máxima.

index_carga_min = np.argmin(ff_sos)
carga_min = ff_sos[index_carga_min] # Carga máxima negativa alcanzada.
despl_min = dd[index_carga_min]     # Desplazamiento para la carga máxima negativa.

print("\n--- Picos\n")
print(f"Carga máxima positiva es: {round(carga_max, 2)} kN.")
print(f"Carga máxima negativa es: {round(carga_min, 2)} kN.")
print(f"Desplazamiento máximo es: {round(despl_max, 2)} mm.")
print(f"Desplazamiento mínimo es: {round(despl_min, 2)} mm.")

# La rigidez inelástica final se cosndiera como al relación entre la fuerza 
# última aplicada y el desplazamiento último.

K_inelastica_pos = carga_max/despl_max*1000
K_inelastica_neg = carga_min/despl_min*1000

print("\n--- Rigidez última\n")
print(f"La rigidez inelástica para la curva positiva es: {round(K_inelastica_pos, 2)} kN/m.")
print(f"La rigidez inelástica para la curva negativa es: {round(K_inelastica_neg, 2)} kN/m.")

# -----------------------------------------------------------------------------
# Separación de la envolvente positiva y negativa.

dd_env_pos = dd_env[21:]; dd_env_neg = dd_env[:22]
ff_env_pos = ff_env[21:]; ff_env_neg = ff_env[:22]

# -----------------------------------------------------------------------------
# Análisis completo sobre cada envolvente.
 
# Arreglo de puntos de cada envolvente.
puntos_env_pos = np.arange(dd_env_pos.shape[0])

# Proceso de regresión
rr2_env, mm_env, bb_env = ProcesoLinealizacion(puntos_env_pos, dd_env_pos, ff_env_pos, inicio=1)

#punto_e_env = np.argmax(rr2_env)    # Punto elástico.
#r2_env      = np.max(rr2_env)       # R2 máximo alcanzado, rr2[punto_e]
punto_e_env = 5
r2_env      = rr2_env[punto_e_env] 
d_e_env     = dd_env_pos[punto_e_env]       # Desplazamiento elástico.
f_e_env     = ff_env_pos[punto_e_env]       # Fuerza elástica.

# La regresión lineal es:
yyb_env = dd_env_pos[:punto_e_env+1]*mm_env[punto_e_env] + bb_env[punto_e_env]

# La energía elásticas es:
E_e_env = 1/2 * ( (d_e_env/1000) * f_e_env) *1000   # Joules [N.m]

print("\n--- Análisis sobre la envolvente\n.")
print(f"Desplazamiento de fluencia, d_y = {round(d_e_env, 2)} mm")
print(f"Fuerza de fluencia,         f_y = {round(f_e_env, 2)} kN")
print(f"Punto de fluencia,          p_y = {punto_e_env}.")
print(f"Valor de R2,                r2  = {r2_env}.")
print(f"Energía elástica,           Ee = {round(E_e_env, 2)} Joules.")

# -----------------------------------------------------------------------------
# Grafico de la envolvente positiva con el rango elástico.

fig_env_pos, ax = plt.subplots()
fig_env_pos.suptitle("Rango elástico - Análisis en la envolvnete positiva")
ax.plot(dd_env_pos, ff_env_pos, "*-", color="grey", label="Envolvente positiva")
ax.plot(dd_env_pos[:punto_e_env+1], yyb_env, '--b', markersize=3,  label=f"Regresión lineal, R2={round(r2_env, 4)}")
ax.plot(d_e_env, f_e_env, '*b', label=f"Punto de fluencia {round(d_e_env, 2), round(f_e_env, 2)}")
format_axes(fig_env_pos)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.grid(b=True, which='major', linestyle='-')
fig_env_pos.tight_layout()
plt.savefig("Resultados_G3_HisteresisMuro/RangoElastico_Envolvente.png")

# -----------------------------------------------------------------------------
# Extracción de los ciclos (trabajando con la curva suavizada).

# Agrupo las curvas positivas y negativas en arreglos independientes.
ff_pos = []; dd_pos = []; nciclos = 0
ff_neg = []; dd_neg = []; puntos_cambio = [0]  # Primer punto de interés en 0. 

# Analizo cada punto del ensayo.
for punto in range(npuntos):

    # Separo las Fuerza (cargas) positivas de las negativas, y sus respectivos 
    # desplazamientos.
    if ff_sos[punto] > 0:
        ff_pos.append(ff_sos[punto])
        dd_pos.append(dd[punto])
    else: 
        ff_neg.append(ff_sos[punto])
        dd_neg.append(dd[punto])
    
    # Contabilizo la cantidad de ciclos y guardo los puntos en que ocurre un 
    # cambio de ciclo.
    if punto == npuntos-1:
        break
    else:
        # Si pasa de una carga positiva a una negativa lo contabilizo:
        if (ff_sos[punto]>0 and ff_sos[punto+1]<0):
            nciclos+=1 
            puntos_cambio.append(punto)
        # Si pasa de una carga negativa a una positiva lo contabilizo:
        if (ff_sos[punto]<0 and ff_sos[punto+1]>0):
            nciclos+=1
            puntos_cambio.append(punto)

# Creo una figura con grilla para graficar todos los ciclos de carga.
fig_ciclos = plt.figure(figsize=(15, 14))
fig_ciclos.suptitle("Semiciclos de la histéresis")

# Defino la grilla y la sposiciones para los gráficos.
grilla = GridSpec(4, 11, figure=fig_ciclos)
xx, yy = np.meshgrid(np.arange(4), np.arange(11))
xx = np.ndarray.flatten(xx)
yy = np.ndarray.flatten(yy)

# Creo contenedores para almacenar los picos y los ciclos.
d_peaks_pos = []; d_peaks_neg = []; ciclos = []
f_peaks_pos = []; f_peaks_neg = []; indices_peaks = []

# Análisis sobre cada uno de los ciclos.
for i in range(nciclos):

    # Separo y almaceno cada ciclo.
    f = ff_sos[puntos_cambio[i] : puntos_cambio[i+1]]
    d = dd[puntos_cambio[i] : puntos_cambio[i+1]]
    
    ciclo = (d,f) 
    ciclos.append(ciclo)

    # Busco el índice de los picos del ciclo.
    index_peaks = []

    # Encuentro el pico de fuerza máximo y mínimo de cada ciclo.
    if check(i) == "odd":
        # Si es impar, busco el mínimo.
        index_peaks = np.argmin(f)
    else:
        # Si es par, busco el máximo.
        index_peaks = np.argmax(f) 

    # Guardo el índice para futuros cálculos.
    indices_peaks.append(index_peaks)

    # obtengo los picos de la fuerza y sus respectivos desplazamientos.
    f_peaks_i = f[index_peaks]
    d_peaks_i = d[index_peaks]
    
    # Almaceno los picos según sean negativos o positivos
    if f_peaks_i > 0:
        d_peaks_pos.append(d_peaks_i); f_peaks_pos.append(f_peaks_i)
    else:
        d_peaks_neg.append(d_peaks_i); f_peaks_neg.append(f_peaks_i)
    
    # Grafico el ciclo.
    ax = fig_ciclos.add_subplot(grilla[ xx[i], yy[i] ])
    ax.plot(ciclo[0], ciclo[1], 'k', markersize=3)

    # Grafico los picos en cada ciclo.
    ax.plot(d_peaks_i, f_peaks_i, '*r', markersize=5)

    # Configuración del gráfico. 
    ax.set_title(f"Semiciclo {i+1}", fontsize=10)
    
    if i == 0:
        # Sólo pongo las unidades de los ejes en el primer gráfico.
        ax.set_xlabel("d [mm]", fontsize=10)
        ax.set_ylabel("F [kN]", fontsize=10)
        ax.legend()

fig_ciclos.subplots_adjust(wspace=0.8, hspace=0.4)
plt.savefig("Resultados_G3_HisteresisMuro/Semiciclos.png")

# -----------------------------------------------------------------------------
# Cálculo de la rigidez elástica sobre el primer ciclo (verificación).

# La información dle ciclo 01.
ciclo_01 = ciclos[0]
npuntos_ciclo_01 = ciclo_01[0].shape[0]

dd_ciclo_01 = ciclo_01[0]
ff_ciclo_01 = ciclo_01[1]

# Lista de puntos.
puntos = np.arange(npuntos_ciclo_01)

rr2, mm, bb = ProcesoLinealizacion(puntos, dd_ciclo_01, ff_ciclo_01, inicio=10)

# Obtengo los valores elásticos encontrando el mayor valor de R2.
#punto_e = np.argmax(rr2)    # Punto elástico.
#r2      = np.max(rr2)       # R2 máximo alcanzado, rr2[punto_e]
punto_e = 70
r2      = rr2[punto_e] 
d_e     = dd_ciclo_01[punto_e]       # Desplazamiento elástico.
f_e     = ff_ciclo_01[punto_e]       # Fuerza elástica.

# La regresión lineal es:
yyb = dd_ciclo_01[:punto_e]*mm[punto_e] + bb[punto_e]

# La energía elásticas es:
E_e = 1/2 * ( (d_e/1000) * f_e) *1000   # Joules [N.m] 

print("\n--- Punto de fluencia en el primer ciclo\n.")
print(f"Desplazamiento de fluencia, d_y_c1 = {round(d_e, 2)} mm")
print(f"Fuerza de fluencia,         f_y_c1 = {round(f_e, 2)} kN")
print(f"Punto de fluencia,          p_y_c1 = {punto_e}.")
print(f"Valor de R2,                r2_c1  = {round(r2, 6)}.")
print(f"Energía elástica,           Ee_c1  = {round(E_e, 2)} Joules.")

# -----------------------------------------------------------------------------
# Gráfica de punto elástico y regresión, ciclo 01.

fig, ax = plt.subplots()
fig.suptitle("Rango elástico - Análisis en el ciclo 01")
ax.plot(dd_ciclo_01, ff_ciclo_01, '-', color = "grey", label="Ciclo 01")
ax.plot(d_e, f_e, '*b', label=f"Punto de fluencia {round(d_e, 2), round(f_e, 2)}")
ax.plot(dd_ciclo_01[:punto_e], yyb, '--b', markersize=3,  label=f"Regresión lineal, R2={round(r2, 4)}")
ax.text(d_e, f_e*1.2, f"Punto p_y: {punto_e}")
format_axes(fig)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.grid(b=True, which='major', linestyle='-')
fig.tight_layout()
plt.savefig("Resultados_G3_HisteresisMuro/RangoElastico_Ciclo01.png")

# -----------------------------------------------------------------------------
# La ductilidad total.

miu     = max(np.max(dd_env_pos), np.max(np.abs(dd_env_neg))) / d_e
miu_env = max(np.max(dd_env_pos), np.max(np.abs(dd_env_neg))) / d_e_env

#miu = max(abs(despl_max), abs(despl_min)) / d_e
#miu_env = max(abs(despl_max), abs(despl_min)) / d_e_env

print("\n--- Ductilidad\n")
print(f"Ductilidad total, rango elástico en ciclo 01,   \mu = {round(miu, 2)}.")
print(f"Ductilidad total, rango elástico en envolvente, \mu = {round(miu_env, 2)}.")

# -----------------------------------------------------------------------------
# Energía disipada y amortiguamiento equivalente.

# Energía disipada de la envolvente pos.
E_dis_env_pos = integrate.simps(ff_env_pos, dd_env_pos/1000)*1000  # Joules = N.m

# Amortiguamiento equivalente considerando el rango elástico en el cilo 01.
xi_eq_c_01 = 1/(4*np.pi) * E_dis_env_pos / E_e
# Amortiguamiento equivalente considerando el rango elástico en la envolvente.
xi_eq_env = 1/(4*np.pi) * E_dis_env_pos / E_e_env

print("\n--- Energía disipada y amortiguamiento\n")
print(f"Energía disipada, rango elástico en la envolvente:         E_dis = {round(E_dis_env_pos, 2)} Joules")
print(f"Amortiguamiento equivalente, rango elástico en ciclo 01:   xi_eq ={round(xi_eq_c_01, 2)}")
print(f"Amortiguamiento equivalente, Rango elástico en envolvente: xi_eq = {round(xi_eq_env, 2)}")

# -----------------------------------------------------------------------------
# Análisis contra derivas.

# Creo contenedores para las cantidades solicitadas.
rigidez_pos     = []; rigidez_neg     = []  # K = f/d
resistencia_pos = []; resistencia_neg = []  # f
ductilidad_pos  = []; ductilidad_neg  = []  # miu = d_u/d_e_env
energia_dis_pos = []; energia_dis_neg = []  # E = int(f.dx)
amort_eq_pos    = []; amort_eq_neg    = []  # 1/(4*np.pi) * E_dis / E_e
derivas_pos     = []; derivas_neg     = []  # max(d)_sentido

# Análisis sobre cada uno de los ciclos.
for i in range(nciclos):

    # La integración de la energía falla en el ciclo 40, de momento se desechan
    # los últimos 4 ciclos.
    if i == 40:
        break

    # Tomo la información del ciclo a estudiar.
    dd_c = ciclos[i][0]; ff_c = ciclos[i][1]

    index_deriva = 0
    # Busco el índice para el desplazamiento máximo, según sea positivo o 
    # negativo.
    if check(i) == "odd":
        index_deriva = np.argmin(dd_c)
    else:
        index_deriva = np.argmax(dd_c)
    
    # El desplazamiento pico del ciclo.
    d_pico_c = dd_c[index_deriva]

    # Obtengo la deriva pico del ciclo.
    deriva_pico_c = d_pico_c / h  

    # Obtengo la resistencia (fuerza alcanzada para su deriva).
    resistencia_c = ff_c[index_deriva]

    # Obtengo la rigidez en el ciclo para la deriva dada [kN/m].
    rigidez_c = resistencia_c / d_pico_c * 1000

    # La ductilidad en el ciclo para la deriva dada.
    ductilidad_c = d_pico_c / d_e_env

    # Para la energía disipada divido el ciclo en 2 tramos: antes y después 
    # del punto máximo. Se toman los valores absolutos (la energía siempre es 
    # positiva).
    indice_pico = indices_peaks[i]

    dd_c_1 = np.abs(dd_c[:indice_pico]); dd_c_2 = np.abs(dd_c[indice_pico:])
    ff_c_1 = np.abs(ff_c[:indice_pico]); ff_c_2 = np.abs(ff_c[indice_pico:])

    # La energía disipada será la resta del área desde que inicia hast el tope 
    # (E_dis_1), y del tope hasta que termina (E_dis_2). Ambas energías se toman positivas

    E_dis_1 = integrate.simps(ff_c_1, dd_c_1/1000)*1000  # Joules = N.m 
    E_dis_2 = integrate.simps(ff_c_2, dd_c_2/1000)*1000  # Joules = N.m
    E_dis_c = E_dis_1 + E_dis_2     # La integración en el sentido contrario es negativa

    # El amortiguamiento equivalente del ciclo es.
    xi_eq_c = 1/(4*np.pi) * E_dis_c / E_e_env

    # Almaceno los resultados según sean positivos o negativos.
    if deriva_pico_c > 0:
        derivas_pos.append(deriva_pico_c)
        resistencia_pos.append(resistencia_c)
        rigidez_pos.append(rigidez_c)
        ductilidad_pos.append(ductilidad_c)
        energia_dis_pos.append(E_dis_c)
        amort_eq_pos.append(xi_eq_c)
    else:
        derivas_neg.append(deriva_pico_c) 
        resistencia_neg.append(resistencia_c)
        rigidez_neg.append(rigidez_c)   
        ductilidad_neg.append(ductilidad_c)
        energia_dis_neg.append(E_dis_c)
        amort_eq_neg.append(xi_eq_c)

# Por comodidad, la envolvente negativa se va a considerar con valores positivos 
# en los gráficos siguientes.

rigidez_neg     = np.abs(np.array(rigidez_neg    ))
resistencia_neg = np.abs(np.array(resistencia_neg))
ductilidad_neg  = np.abs(np.array(ductilidad_neg ))
energia_dis_neg = np.abs(np.array(energia_dis_neg))
amort_eq_neg    = np.abs(np.array(amort_eq_neg   ))
derivas_neg     = np.abs(np.array(derivas_neg    ))

# La energía es acumulativa entre los diferentes ciclos, así que debo sumar 
# ambas energías calculadas (la negativa y la positiva), y realizar la suma 
# acumulativa:
energia_dis    = energia_dis_pos + energia_dis_neg
energia_dis_ac = np.cumsum(energia_dis)

# Así mismo funciona el factor de amortiguamiento equivalente.
amortiguamiento_eq    = amort_eq_pos + amort_eq_neg
amortiguamiento_eq_ac = np.cumsum(amortiguamiento_eq)

# -----------------------------------------------------------------------------
# Gráficas contra derivas.

# Gráficos para las derivas poritivas.
GraficoParejas(derivas_pos, resistencia_pos, derivas_neg, resistencia_neg, r"$\delta$", "F [kN]" ,  "Degradación de la resistencia contra la deriva", "DegradacionResistencia")
GraficoParejas(derivas_pos, rigidez_pos,     derivas_neg, rigidez_neg,     r"$\delta$", "K [kN/m]", "Degradación de la rigidez contra la deriva",     "DegradaconRigidez")
GraficoParejas(derivas_pos, ductilidad_pos,  derivas_neg, ductilidad_neg,  r"$\delta$", r"$\mu$",   "Ductilidad contra la deriva",                    "Ductilidad")

GraficoSimple(derivas_pos, energia_dis_ac,        r"$\delta$", "E_dis [Joules]", "Energía disipada contra derivas",            "Energía disipada", "EnergiaDisipada")
GraficoSimple(derivas_pos, amortiguamiento_eq_ac, r"$\delta$", r"$\xi_{eq}$",    "Amortiguamiento equivalente contra derivas", r"$\xi_{eq}$",      "Amortiguamiento")

# -----------------------------------------------------------------------------
# Organización de resultados en tablas para exportar.

tabla_resultados_pos = pd.DataFrame(
    data = np.c_[derivas_pos, resistencia_pos, rigidez_pos, ductilidad_pos, energia_dis_pos, amort_eq_pos],
    columns = ["deriva [mm]", "resistencia [kN]", "rigidez [kN/m]", "ductilidad", "E_dis [J]", "xi_eq"]
    )

tabla_resultados_neg = pd.DataFrame(
    data = np.c_[derivas_neg, resistencia_neg, rigidez_neg, ductilidad_neg, energia_dis_neg, amort_eq_neg],
    columns = ["deriva [mm]", "resistencia [kN]", "rigidez [kN/m]", "ductilidad", "E_dis [J]", "xi_eq"]
    )

tabla_histeresis_sos = pd.DataFrame(
    data = np.c_[tt, dd, ff_sos],
    columns =["t [s]", 'd [mm]', 'f [kN]'] 
    ) 

tabla_envolvente = pd.DataFrame(
    data = np.c_[dd_env, ff_env],
    columns = ['d [mm]', 'f [kN]']
    )

tabla_max = pd.DataFrame(
    data = np.c_[d_peaks_pos, f_peaks_pos],
    columns = ["d [mm]", "f [kN]"]
    )

tabla_min = pd.DataFrame(
    data = np.c_[d_peaks_neg, f_peaks_neg],
    columns = ["d [mm]", "f [kN]"]
    )

# se crea un archivo de MS EXCEL
archivo_resultados = "Resultados_G3_HisteresisMuro/G3_HisteresisMuro_ResultadosAnalisis.xlsx"
writer = pd.ExcelWriter(archivo_resultados, engine = 'xlsxwriter')

# cada tabla hecha previamente es guardada en una hoja del archivo de Excel
tabla_resultados_pos.to_excel(writer, sheet_name="pos")
tabla_resultados_neg.to_excel(writer, sheet_name="neg")
tabla_histeresis_sos.to_excel(writer, sheet_name='histeresis_sos')
tabla_envolvente.to_excel(    writer, sheet_name='envenvolvente')
tabla_max.to_excel(           writer, sheet_name='peak_max')
tabla_min.to_excel(           writer, sheet_name='peak_min')

# Guardo el excel.
writer.save()

print(f"\nResultados guardados en excel. Dir: {archivo_resultados}\n.")

# -----------------------------------------------------------------------------
# Gráfica de la curva histerética.

# Límite de los gráficos.
lim = 120

fig = plt.figure(figsize=(15, 10))
fig.suptitle("Histéresis y envolvente")

# Defino la grilla: 2 filas, 3 columnas, en fig.
grilla = GridSpec(2, 3, figure=fig)

# ax1 : gráfica de la histéresis.
ax1 = fig.add_subplot(grilla[:, 0:2])

# Grafico la histéresis.
ax1.plot(dd, ff, '-', color = "grey", markersize=2, label="Histéresis")
# Grafico el suavizado
ax1.plot(dd, ff_sos, '-y', markersize=2, label="Histéresis sos")
# Resalto los ejes.
ax1.plot([0, 0], [-lim, lim], '-k')
ax1.plot([-lim, lim], [0, 0], '-k')

# Grafico los máximos (puntros de la envolvente).
ax1.plot(dd_env_pos, ff_env_pos, '*-r', markersize=5, label="Envolvente positiva")
ax1.plot(dd_env_neg, ff_env_neg, '*-b', markersize=5, label="Envolvente negativa")

# ax2 : histéresis positiva.
ax2 = fig.add_subplot(grilla[0, 2])
ax2.plot(dd_pos, ff_pos, '-y', markersize=2, label="Histeresis sos (+)")
ax2.plot(dd_env_pos, ff_env_pos, '*-r', markersize=5, label="Envolvente positiva")

# Resalto los ejes.
ax2.plot([0, 0], [0, lim], '-k')
ax2.plot([0, lim], [0, 0], '-k')

# Limito el gráfico
ax2.set_xlim(right=lim, left=0)

# ax3 : histéresis negativa.
ax3 = fig.add_subplot(grilla[1, 2])
ax3.plot(dd_neg, ff_neg, '-y', markersize=2, label="Histeresis sos (-)")
ax3.plot(dd_env_neg, ff_env_neg, '*-b', markersize=5, label="Envolvente negativa")

# Resalto los ejes.
ax3.plot([0, 0], [0, -lim], '-k')
ax3.plot([-lim, 0], [0, 0], '-k')

# Limito el gráfico
ax3.set_xlim(right=0, left=-lim)

# Formateo todos los gráficos.
format_axes(fig)
plt.tight_layout
plt.savefig(f"Resultados_G3_HisteresisMuro/Histeresis.png")

# -----------------------------------------------------------------------------
# Gráfica de desplazamiento y carga contra tiempo.

fig = plt.figure()
fig.suptitle("Desplazamiento y Carga contra tiempo")

axd = fig.add_subplot(2, 1, 1)
axd.plot(tt, dd, '-b', label="Desplazamiento aplicado")
#axd.set_xlabel("t [s]")
axd.set_ylabel("d [mm]")
axd.legend()
axd.grid() 
axf = fig.add_subplot(2, 1, 2)
axf.plot(tt, ff, '-b', label="Carga asimilada")
axf.set_xlabel("t [s]")
axf.set_ylabel("F [kN]")
axf.legend()
axf.grid()

plt.savefig(f"Resultados_G3_HisteresisMuro/Desplazamiento_Carga.png")

# Se mostrarían todas las figuras creadas, pero es mejor verlas en la carpeta 
# (se demora mucho en cargar)
#plt.show()

# Fin :)