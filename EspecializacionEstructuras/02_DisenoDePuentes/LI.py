"""
Cálculo de las líneas de influencia para vigas simplemente apoyadas.

By:     Michael Heredia Pérez
email:  mherediap@unal.edu.co
Fecha:  mayo/2022

Universidad Nacional de Colombia Sede Manizales
Especialización en Estructuras.
"""

# Librerías de trabajo.
import numpy as np
import matplotlib.pyplot as plt


# Funciones para simplificación del código:
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

def Grafica_LI(LI, dominio, concepto, xi):
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


# LI para un ejemplo.
# -----------------------------------------------------------------------------

L  = 30           # m,    Longitud entre apoyos del puente.
E  = 200000       # MPa,  módulo de elasticidad del material.
bv = 0.7          # m,    base de la viga.
hv = 1            # m,    alto de la viga.
I  = bv*hv**3/12  # m4,   Inercia de la viga rectangular.

# Punto de interés para la LI (punto de análisis).
x_li = 3

# Discretización del puente.
n = 1000   # Dominio de la línea de influencia.
ll = np.linspace(0, L, n)

# Ejecución.
# -----------------------------------------------------------------------------

# Calculo la LI en x_li
R, V, M, t, z, x_li_a = Calcula_LI(ll, n, L, x_li, E, I)  

# Grafico la línea de influencia.
Grafica_LI(R, ll, "R", x_li_a)
Grafica_LI(V, ll, "V", x_li_a)
Grafica_LI(M, ll, "M", x_li_a)
Grafica_LI(t, ll, "t", x_li_a)
Grafica_LI(z, ll, "z", x_li_a)

# Fin :)