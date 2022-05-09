"""
Diseño de puentes. Examen, punto 3.

By:     Michael Heredia Pérez.
CC:     1010134928
email:   mherediap@unal.edu.co

Universidad Nacional de Colombia sede Manizales.
Especialización en Estructuras.
"""

# Librerías de trabajo.
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


# Defino funciones de utilidad.
# -----------------------------------------------------------------------------

def AproximaPunto_xi(dominio, xi):
    """Aproxima un punto de interés a un x dentro del dominio de la LI."""
    posicion = np.argmin(np.abs(dominio - xi))
    x = dominio[posicion]
    return x, posicion
    

def Calcula_LI(dominio, n, L, xi, E, I):
    """Calcula las lineas de influencia para reacciones, fuerza cortante, 
    momento flector, giro y desplazamiento."""

    # Aproximo los puntos de interés a uno que exista dentro de mi dominio.
    x, posicion = AproximaPunto_xi(dominio, xi)

    # Reservo espacio de memoria.
    R = np.zeros((n, 2))            # Reacciones Ra y Rb.
    V = np.zeros(n)                 # Fuerza Cortante.
    M = np.zeros(n)                 # Momento Flector.
    t = np.zeros(n)                 # Girdo dz_dx.
    z = np.zeros(n)                 # Desplazamiento y deflexión z.

    # Cargas puntuales móviles.
    for i in range(n):
            a = dominio[i]
            b = L-a 

            # Cálculo de las reacciones.
            Ra = b/L; R[i, 0] += Ra
            Rb = a/L; R[i, 1] += Rb

            # Condicional para saber qué ecuaciones utilizar según la posición.
            if a >= x:
                    V[i] = -Ra 
                    M[i] = b/L*x;
                    t[i] = b*x**2/(2*L*E*I) +a**2/(2*E*I) -a**3/(6*L*E*I) -a*L/(3*E*I);
                    z[i] = -L*b*x/(6*E*I)*(1 -b**2/L**2 - x**2/L**2);
            else:
                    V[i] = Rb;
                    M[i] = a*(L-x)/L;
                    t[i] = a*x/(E*I) -a*x**2/(2*L*E*I) -a**3/(6*L*E*I) -a*L/(3*E*I);
                    z[i] = -L*a*(L-x)/(6*E*I)*(1 -a**2/L**2 -((L-x)/L)**2);
    
    return R, V, M, t, z, x 


def ObtencionDeOrdenadas(LI, dominio, puntos):
    """Obtiene las ordenadas de una LI de los ejes dados ajustados al 
    dominio."""
    
    nejes = puntos.shape[0]

    ordenadas = np.zeros(nejes)
    puntos_ap = np.zeros(nejes)   # Puntos aproximados.

    # Por cada punto en el arreglo puntos
    for i in range(nejes):
        # Calculo su punto aproximado en el dominio y el índice de este.
        x_li, posicion = AproximaPunto_xi(dominio, puntos[i])
        # Con el índice puedo sacar el valor que le corresponde dentro de la 
        # LI.
        ordenadas[i] += LI[posicion]
        puntos_ap[i] += x_li

    return ordenadas, puntos_ap


def PosicionEjes(xi, d12, d23):
    """Calcula la posición de los ejes del camión en el sentido A-B sin 
    ajustarlos al dominio de las LI."""
    
    # El camión va de A a B, la posición de los ejes es:
    x1, x2, x3 = xi, xi+d23, xi+d23+d12
    p1, p2, p3 = p_eje_3, p_eje_2, p_eje_1
    
    puntos = np.array([x1, x2, x3])
    cargas = np.array([p1, p2, p3])

    return puntos, cargas
        

def CalculaValoresX(llc, LI, dominio, d12, d23):
    """Calcula el valor de una medición para la LI dada, en cada punto posible 
    del camión sin salirse del puente."""
    
    # Defino un contenedor para los resultados. 
    resultados = np.zeros((llc.shape[0], 2))

    # Por cada posición posible del camión.
    for i in range(llc.shape[0]):
        
        # Tomo el punto guía.
        xi = llc[i]
        
        # Calculo la posición de los ejes sin ajuste al dominio.
        puntos, cargas = PosicionEjes(xi, d12, d23)
        
        # Busco las ordenadas correspondientes para la LI deseada.
        ordenadas, puntos_ap = ObtencionDeOrdenadas(LI, dominio, puntos)
        
        # Opero las ordenadas por las cargas correspondinetes y encuentro el 
        # valor:
        valor = np.dot(ordenadas, cargas)

        # Almaceno el valor de la ccarga/desplazamiento calculado y el punto 
        # desde A donde se encuentra el camión, 
        resultados[i, 0] += xi
        resultados[i, 1] += valor
    
    # Encuentro el máximo pedido y su posición.
    val_max = np.max(np.abs(resultados[:, 1]))
    indice = np.argmax(np.abs(resultados[:, 1]))
    distA  = resultados[indice, 0]
        
    return resultados, val_max, distA


def GraficaEjes(ax, distA, d12, d23):
    """Grafica los ejes según el resultado final que maximiza."""
    
    puntos, cargas = PosicionEjes(distA, d12, d23)
    #ordenadas, puntos_ap = ObtencionDeOrdenadas(LI, dominio, puntos)
    
    # Por cada par de (puntos, ordenadas) 
    for i in range(3):
        ax.plot(puntos[i], 0, 'vr', label = "eje")
        ax.plot(puntos[i], 0, '|r')


def Grafica_LI(LI, dominio, concepto, xi, distA =0, d12=0, d23=0):
    """Grafica las lineas de influencia. Si es de reacciones las grafica en un 
    mismo pliego."""

    # Si se grafica LI para reacciones.
    if concepto == "R":

        fig = plt.figure()
        fig.subplots_adjust(wspace=1.5, hspace=1.5) # Ajuste del espacio entre gráficos.

        for i in range (2):
            ax = fig.add_subplot(2, 1, i+1)
            ax.plot(dominio, LI[:, i], '-b')
            ax.plot(dominio, np.zeros(dominio.shape[0]), '--y')
            ax.plot(xi, 0, 'Hg', label=r"$x_i$")
            # Si hay una distancia extra entonces grafica ejes.
            if distA != 0:
                GraficaEjes(ax, distA, d12, d23)
            if i == 0:
                ax.set_title("Línea de influencia de la reacción en A: Ra")
            elif i == 1:
                ax.set_title("Línea de influencia de la reacción en B: Rb")
            else:
                break
            ax.set_xlabel(r'$L [m]$')
            ax.set_ylabel(r'$R[kN]$')
            ax.set_xlim(0, L)
            ax.grid(b=True, which='major', linestyle='-')

        plt.show()
    
    # Si se grafica LI para las demás variables:
    else:
        fig = plt.figure()
        fig.set_size_inches(8, 4)
        fig.subplots_adjust(wspace=1.5, hspace=1.5) # Ajuste del espacio entre gráficos.
        
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(dominio, LI, '-b')
        ax.plot(dominio, np.zeros(dominio.shape[0]), '--y')
        ax.plot(xi, 0, 'Hg', label=r"$x_i$")
        # Si hay una distancia extra entonces grafica ejes.
        if distA != 0:
                GraficaEjes(ax, distA, d12, d23)
        if concepto == "V": 
            ax.set_ylabel(r'$V [kN]$')
            ax.set_title("Línea de influencia para Fuerza Cortante (V)")
        if concepto == "M": 
            ax.set_ylabel(r'$M [kN.m]$')
            ax.set_title("Línea de influencia para Momento Flector (M)")
        if concepto == "t": 
            ax.set_ylabel(r'$\theta [^\degree]$')
            ax.set_title("Línea de influencia para el girdo " + r"$(\frac{dz}{dx})$")
        if concepto == "z": 
            ax.set_ylabel(r'$z [kN]$')
            ax.set_title("Línea de influencia para la deflexión vertical (z)")

        ax.set_xlabel(r'$L [m]$')
        ax.set_xlim(0, L)
        ax.grid(b=True, which='major', linestyle='-')
        
        plt.show()


# Desarrollo del ejercicio.
# -----------------------------------------------------------------------------

# Considerando n = suma de todos los dígitos del documento de identidad CC.
# Calculo n:
CC = 1010134928

n = 0

while CC != 0:          # Mientras haya dígitos en la CC.
    cifra = CC % 10     # Tomo el módulo 10 de CC (último dígito).
    CC //= 10           # Con la división entera eliminio la última cifra.
    n += cifra          # Sumo lacifra eliminada en cada paso.

# Reporto mi valor de n.
print(f"Valor de n para el ejercicio: {n}.")


# Información del ejercicio:
L = 25 + n/4     # m,    Longitud entre apoyos del puente.
print(f"Longitud del puente para el ejercicio: {L} m.")

p_eje_1 = 40     # kN,   Carga del eje 1 (frontal).
p_eje_2 = 160    # kN,   Carga del eje 2 (medio).
p_eje_3 = 160    # kN,   Carga del eje 3 (trasero).
d12     = 4.3    # m,    Distancia entre ejes 1-2.
d23     = 6      # m,    Distancia entre ejes 2-3.

lc = d12+d23     # m, Largo estudiado del camión.

# Datos asrbitrarios para la sección y propiedades del tablero.
E = 200000       # MPa, módulo de elasticidad del material.
bv = 0.7         # m, base de la viga.
hv = 1           # m, alto de la viga.
I = bv*hv**3/12  # m4, Inercia de la viga rectangular.


# Punto de interés para la LI (punto de análisis).
#x_li = 3

# Se tienen 5 puntos de interés, 1 por ada punto del enunciado, y el 
# numeral c) implica dos puntos:
xx_li = [L, L/3, 0, L, L/2]
n_xx_li = 5

# Creo los puntos de análisis, donde irán los ejes del camión, comenzando a 
# medir desde al apoyo A hacia B. 
x1, x2, x3 = 0, 0, 0  
#xi = 3           # VALOR GUÍA PARA INDICAR EL MOVIMIENTO DEL CAMIÓN. VARÍA.

# Discretización del puente.
n = 1000   # Dominio de la línea de influencia.
ll = np.linspace(0, L, n)

# Encuentro el límite en que se puede mover el camión y su índice en el dominio 
# de las LI. 
lim_lc, pos_lim_lc = AproximaPunto_xi(ll, L-lc)
llc = ll[: pos_lim_lc]     # En este espacio se mueve el camión.


# Numeral a) Maximizar la reacción en el apoyo B.
# -----------------------------------------------------------------------------

# Calculo su linea de influencia, xx_li[0] = L
R1, V1, M1, t1, z1, x1a = Calcula_LI(ll, n, L, xx_li[0], E, I)  

# Calculo los resultados desplazando el camión en ambos sentidos para Rb.
resultados_a, val_max_a, distA_a = CalculaValoresX(llc, R1[:, 1], ll, d12, d23)


# Imprimo los resultados.
print("\nNumeral a)")
print(f"Reacción max en B: {val_max_a.round(2)} [kN].")
print(f"Distancia desde el punto A: {distA_a.round(2)} [m].") 

# Grafico la línea de influencia.
Grafica_LI(R1, ll, "R", x1a, distA_a, d12, d23)

# Numeral b) Maximizar el momento en un punto localizado a un tercio de la luz 
# con respecto al apoyo A.
# -----------------------------------------------------------------------------

# Calculo su linea de influencia, xx_li[1] = L/3
R2, V2, M2, t2, z2, x2a = Calcula_LI(ll, n, L, xx_li[1], E, I)  

# Calculo los resultados desplazando el camión en ambos sentidos para M.
resultados_b, val_max_b, distA_b = CalculaValoresX(llc, M2, ll, d12, d23)

# Imprimo los resultados.
print("\nNumeral b)")
print(f"Momento máximo en L/3: {val_max_b.round(2)} [kN.m].")
print(f"Distancia desde el punto A: {distA_b.round(2)} [m].") 

# Grafico la línea de influencia.
Grafica_LI(M2, ll, "M", x2a, distA_b, d12, d23)


# Numeral c) Maximizar el giro en los apoyos A y B.
# -----------------------------------------------------------------------------

# Calculo sus lineas de influencia:
R3, V3, M3, t3, z3, x3a = Calcula_LI(ll, n, L, xx_li[2], E, I)  # xx_li[2] = 0
R4, V4, M4, t4, z4, x4a = Calcula_LI(ll, n, L, xx_li[3], E, I)  # xx_li[3] = L

# Calculo los resultados desplazando el camión en ambos sentidos para M.
resultados_c1, val_max_c1, distA_c1 = CalculaValoresX(llc, t3, ll, d12, d23)
resultados_c2, val_max_c2, distA_c2 = CalculaValoresX(llc, t4, ll, d12, d23)

# Imprimo los resultados.
print("\nNumeral c)")
print(f"Giro máximo en A, en 0: {val_max_c1.round(6)} [°].")
print(f"Distancia desde el punto A: {distA_c1.round(2)} [m].") 

print(f"Giro máximo en B, en L: {val_max_c2.round(6)} [°].")
print(f"Distancia desde el punto A: {distA_c2.round(2)} [m].") 

# Grafico la línea de influencia.
Grafica_LI(t3, ll, "t", x3a, distA_c1, d12, d23)
Grafica_LI(t4, ll, "t", x4a, distA_c2, d12, d23)


# Numeral d) Desplazamiento en el centro de la luz.
# -----------------------------------------------------------------------------

# Calculo su linea de influencia, xx_li[4] = L/2
R5, V5, M5, t5, z5, x5a = Calcula_LI(ll, n, L, xx_li[4], E, I)  

# Calculo los resultados desplazando el camión en ambos sentidos para M.
resultados_d, val_max_d, distA_d = CalculaValoresX(llc, z5, ll, d12, d23)

# Imprimo los resultados.
print("\nNumeral d)")
print(f"Deflexión máxima en L/2: {val_max_d.round(6)} [m].")
print(f"Distancia desde el punto A: {distA_d.round(2)} [m].") 

# Grafico la línea de influencia.
Grafica_LI(z5, ll, "z", x5a, distA_d, d12, d23)

# Fin :)