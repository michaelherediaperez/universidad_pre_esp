# -*- coding: utf-8 -*-

"""
Análisis no lineal de un gancho metálico mediante considerando no linealidad 
material en elementos finitos isoparamétricos de 8 nodos (Q8) en tensión 
plana (TP). 

Las rutinas programadas siguen el modelo de fluencia de von Mises para 
materiales metálicos y por consiguiente de un endurecimiento isotrópico.

Metodología de Bhatti, cap 8. 
Referirse al PDFs para mejor comprensión.

Hecho por: Michael Heredia Pérez
Fecha: Diciembre 2021
Correo: mherediap@unal.edu.co
Universidad: Universidad Nacional de Colombia Sede Manizales.
"""

# =============================================================================
#           Librerías
# =============================================================================

# Si al ejecutar el código recibe un mensaje de error del tipo 
# "no module named --", ejecute el comando del comentario "pip install -- " en 
# la consola de comando (cmd) para instalar la librería. 

try:
    import numpy as np              # python -m pip install numpy
    import pandas as pd             # python -m pip install pandas
    import matplotlib.pyplot as plt # python -m pip install matplotlib
    import warnings                 # python -m pip install warnings
    import xlsxwriter               # python -m pip install xlsxwriter
except ImportError:
    raise Exception("Instale las librerías desde cmd. (leer comentario)")

# Funciones del script turinas_nl.py
try:
    from rutinas_nl import CalculaEsfDesviadores
    from rutinas_nl import CalculaDefPlasticasEfectivas
    from rutinas_nl import CalculaPuntoBeta
    from rutinas_nl import dF_dS__vm
    from rutinas_nl import F_vm_tp
    from rutinas_nl import CalculaMatrizConstPlastica
    from rutinas_nl import IncrementoElastoplastico_IH
    from rutinas_nl import DeterminaCt
except ImportError:
    raise Exception("Verifique que el archivo rutinas_nl.py esté en la misma carpeta que Q8_NL.py")


# ==============================================================================
#           Constantes de facil lectura
# ==============================================================================
X, Y = 0, 1
NL1, NL2, NL3, NL4, NL5, NL6, NL7, NL8 = range(8)
g = 9.81 # m/s^2   gravedad.


# ==============================================================================
#           Funciones para graficación y análisis
# ==============================================================================

def ConvierteVector_ft(xnod, lado, carga, espesor):
    """
    Función que convierte las fuerzas superficiales aplicadas a un elemento
    finito rectangular serendípito de 8  nodos a sus correspondientes cargas 
    nodales equivalentes ft    
    
    Datos de entrada:
    - xnod: coordenadas nodales del elemento finito.
    - lado: arista en la que se aplica la carga, puede tomar los siguientes
          valores: 123, 345, 567, 781
    - carga: fuerza distribuida en los nodos
        [ t1x t1y t2x t2y t3x t3y ]; % si carga se aplica sobre lado 123
        [ t3x t3y t4x t4y t5x t5y ]; % si carga se aplica sobre lado 345
        [ t5x t5y t6x t6y t7x t7y ]; % si carga se aplica sobre lado 567
        [ t7x t7y t8x t8y t1x t1y ]; % si carga se aplica sobre lado 781
    - espesor: espesor del elemento.
    
    Datos de salida:
    - ft: vector de fuerzas nodales equivalentes. 
    """
    
    # Las constantes de facil lectura son globales.
    global X, Y, NL1, NL2, NL3, NL4, NL5, NL6, NL7, NL8
    
    # Se definen los indices de los lados.
    if   lado == 123: idx = np.array([ 1, 2, 3 ]) - 1
    elif lado == 345: idx = np.array([ 3, 4, 5 ]) - 1
    elif lado == 567: idx = np.array([ 5, 6, 7 ]) - 1
    elif lado == 781: idx = np.array([ 7, 8, 1 ]) - 1
    else: 
        raise Exception("Únicamente se permiten los lados 123, 345, 567 o 781")

    # Número de nodos.
    nnds = xnod.shape[0]
    if nnds not in (8, 9):
        raise Exception('Solo para elementos rectangulares de 8 o 9 nodos')

    # Se define el número de puntos de la cuadratura y se obtienen los puntos
    # de evaluación y los pesos de la cuadratura.
    n_gl       = 5
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_gl)
    
    # Se definen las funciones de forma unidimensionales y sus derivadas.
    NN      = lambda xi: np.array([ xi*(xi-1)/2, (1+xi)*(1-xi), xi*(1+xi)/2 ])
    dNN_dxi = lambda xi: np.array([ xi - 1/2   , -2*xi        , xi + 1/2    ])
       
    # Se calcula el vector de fuerzas distribuidas en los nodos.
    te = np.zeros(2*nnds)
    te[np.c_[2*idx, 2*idx + 1].ravel()] = carga
    
    # Se asigna el espacio para los contenedores.
    suma   = np.zeros((2*nnds, 2*nnds))
    N      = np.zeros(nnds)
    dN_dxi = np.zeros(nnds)

    # Cálculo de la integral.
    for p in range(n_gl):
        # Se evalúan las funciones de forma
        N[idx] = NN(x_gl[p])
        matN = np.empty((2,2*nnds))
        for i in range(nnds):
            matN[:,[2*i, 2*i+1]] = np.array([[N[i], 0   ],
                                             [0,    N[i]]])

        # se calcula el jacobiano
        dN_dxi[idx] = dNN_dxi(x_gl[p])
        dx_dxi      = np.dot(dN_dxi, xnod[:,X])
        dy_dxi      = np.dot(dN_dxi, xnod[:,Y])
        ds_dxi      = np.hypot(dx_dxi, dy_dxi)

        # y se calcula la sumatoria.
        suma += matN.T @ matN * ds_dxi*w_gl[p]
    
    # Vector de fuerzas nodales equivalentes.
    ft = espesor * (suma @ te)
    
    return ft
 
    
def ExtrapolaEsfDef(deformaciones, esfuerzos, nnds, nef, LaG):
    """
    Realiza la extrapolación de los esfuerzos y las deformaciones medidas en 
    los puntos de GL a los nodos.
    
    Datos de entrada:
    - deformaciones: arreglo de deformaciones globales.
    - esfuerzos: arreglo de esfeurzos globales.
    - nnds: número de nodos en el sistema.
    - nef: número de elementos finitos.
    - LaG: matriz de global a local.
    
    Datos de salida (en este orden):
    - sx, sy, txy: vectores de esfuerzos de tensión plana.
    - ex, ey, gxy: vectores de deformaciones de tensión plana.
    """
    
    # Contenedor para el número de EF adyacentes que tendrá cada nodo.
    num_elem_ady = np.zeros(nnds)
    
    # Contenedores para los esfuerzos y deformaciones a calcular.
    sx  = np.zeros(nnds);    ex  = np.zeros(nnds)
    sy  = np.zeros(nnds);    ey  = np.zeros(nnds)
    txy = np.zeros(nnds);    gxy = np.zeros(nnds)
    
    # Matriz de extrapolación.
    A = np.array([
        [  3**(1/2)/2+1,           -1/2,           -1/2,   1-3**(1/2)/2],
        [3**(1/2)/4+1/4, 1/4-3**(1/2)/4, 3**(1/2)/4+1/4, 1/4-3**(1/2)/4],
        [          -1/2,   1-3**(1/2)/2,   3**(1/2)/2+1,           -1/2],
        [1/4-3**(1/2)/4, 1/4-3**(1/2)/4, 3**(1/2)/4+1/4, 3**(1/2)/4+1/4],
        [  1-3**(1/2)/2,           -1/2,           -1/2,   3**(1/2)/2+1],
        [1/4-3**(1/2)/4, 3**(1/2)/4+1/4, 1/4-3**(1/2)/4, 3**(1/2)/4+1/4],
        [          -1/2,   3**(1/2)/2+1,   1-3**(1/2)/2,           -1/2],
        [3**(1/2)/4+1/4, 3**(1/2)/4+1/4, 1/4-3**(1/2)/4, 1/4-3**(1/2)/4]
    ])
    
    # En cada elemento finito
    for e in range(nef):
        # extrapolación de esfuerzos
        sx[LaG[e]]  += A @ esfuerzos[e,:,:,0].ravel()
        sy[LaG[e]]  += A @ esfuerzos[e,:,:,1].ravel()
        txy[LaG[e]] += A @ esfuerzos[e,:,:,2].ravel()
        # y extrapolación de deformaciones.
        ex[LaG[e]]  += A @ deformaciones[e,:,:,0].ravel()
        ey[LaG[e]]  += A @ deformaciones[e,:,:,1].ravel()
        gxy[LaG[e]] += A @ deformaciones[e,:,:,2].ravel()
        
        # Se miran los nodos del EF (e) globalmente, y por cada vez que estos 
        # aparezcan se suma 1 para contabilizar los EF adyacentes.
        num_elem_ady[LaG[e]] += 1
    
    # Promedio de los esfuerzos y deformaciones de los elementos alisando los 
    # resutlados.
    sx  /= num_elem_ady;   ex  /= num_elem_ady
    sy  /= num_elem_ady;   ey  /= num_elem_ady
    txy /= num_elem_ady;   gxy /= num_elem_ady
    
    return sx, sy, txy, ex, ey, gxy


def GraficarEstructura():
    """
    Grafica la malla de EFs de la esrtuctura y su deformada.  
    """
    
    # Variables y constantes globales.
    global NEF, XNOD, NNDS
    global X, Y, NL1, NL2, NL3, NL4, NL5, NL6, NL7, NL8

    # Inicializo el lienzo.
    fig = plt.figure()
    fig.suptitle("Estructura", fontsize=20)
    #fig.set_size_inches(10, 10)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Lienzo para la malla original.
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Malla de elementos finitos")
    for e in range(NEF):
        nodos = LaG[e, [NL1, NL2, NL3, NL4, NL5, NL6, NL7, NL8, NL1]]
        plt.plot(XNOD[nodos, X], XNOD[nodos, Y], "gray")
    ax1.set_xlabel("$x$ [m]")
    ax1.set_ylabel("$y$ [m]")
    ax1.set_aspect('equal')
    ax1.autoscale(tight=True)    
    plt.tight_layout()
    
    # Lienzo para la malla deformada.
    delta = np.reshape(DESPLAZAMIENTOS, (NNDS,2))
    ESC = 5000                   # factor de escalamiento de la deformada
    xdef = XNOD + ESC*delta      # posición de la deformada

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title(f"Deformada escalada {ESC} veces")
    for e in range(NEF):
        nod_ef = LaG[e, [NL1, NL2, NL3, NL4, NL5, NL6, NL7, NL8, NL1]]
        plt.plot(XNOD[nod_ef, X], XNOD[nod_ef, Y], "gray",
                        label="Posición original"  if e == 0 else "", lw=0.5)
        plt.plot(xdef[nod_ef, X], xdef[nod_ef, Y], "b",
                        label="Posición deformada" if e == 0 else "")
    ax2.set_xlabel("$x$ [m]")
    ax2.set_ylabel("$y$ [m]")
    ax2.set_aspect('equal')
    ax2.autoscale(tight=True)    
    plt.tight_layout()
    
    # Presento el gráfico.
    plt.show()


def GraficarMagnitudesEF(campo, titulo, var_names, LaG, xnod, angulos=None):
    """
    Grafica n varibles de un campo sobre una malla de EFs especificada.
    
    Datos de entrada:
    - campo: campo que se va a graficar (esfuerzos, deformaciones, etc.)
    - titulo: título del gráfico.
    - var_names: arreglo con los nombres de las variables.
    - LaG: matriz de local a global.
    - xnod: coordenadas de los nodos de la malla de EFs.
    - angulos: opcional para magnitudes prinicipales (esfuerzos principales i.e.)
    
    Para propósitos de graficación el EF se divide en 6 triángulos así: 
        
    7 -------6--------5
    |       /|\       |
    | EFT6 / | \ EFT3 |
    |     /  |  \     |
    |    /   |   \    |
    |   /    |    \   |
    |  /     |     \  |
    | /      |      \ |
    8/  EFT5 | EFT4  \4
    |\       |       /|
    | \      |      / |
    |  \     |     /  |
    |   \    |    /   |
    |    \   |   /    |
    |     \  |  /     |
    | EFT1 \ | / EFT2 |
    |       \|/       |
    1--------2--------3
    
    """
    # Las constantes de facil lectura son globales.
    global X, Y, NL1, NL2, NL3, NL4, NL5, NL6, NL7, NL8
    
    # Número de elementos finitos en la mala original.
    nef = LaG.shape[0]
    
    # Matriz de local a global para la submalla triangular.
    LaG_t = np.zeros((6*nef, 3), dtype = int)
    for e in range(nef):
        LaG_t[6*e + 0, :] = LaG[e, [NL1, NL2, NL8]]
        LaG_t[6*e + 1, :] = LaG[e, [NL2, NL3, NL4]]
        LaG_t[6*e + 2, :] = LaG[e, [NL4, NL5, NL6]]
        LaG_t[6*e + 3, :] = LaG[e, [NL2, NL4, NL6]]
        LaG_t[6*e + 4, :] = LaG[e, [NL2, NL6, NL8]]
        LaG_t[6*e + 5, :] = LaG[e, [NL6, NL7, NL8]]
        
    # Número de variables en el campo.
    nvar = len(campo)
      
    # Inicializo el gráfico.
    fig = plt.figure()
    fig.suptitle(titulo, fontsize=20)
    #fig.set_size_inches(10, 10)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    # Por cada variable del campo (var = 0, 1, 2 ...)
    for var in range(nvar):
        
        # Selecciono la variable que se quiere graficar.
        variable = campo[var]
        # Selecciono el nombre de la variable.
        var_name = var_names[var]
                
        # Preparo el espacio para el plot.
        ax = fig.add_subplot(1, nvar, var+1)
                
        # Encuentro el máximo en valor absoluto para ajustar el colorbar()
        val_max = np.max(np.abs(variable))
        
        # Graficación de la malla de EFs.
        for e in range(nef):
        
            # se dibujan las aristas de la malla original.
            nod_ef = LaG[e, [NL1, NL2, NL3, NL4, NL5, NL6, NL7, NL8, NL1]]
            plt.plot(xnod[nod_ef, X], xnod[nod_ef, Y], lw=0.5, color="gray")
            
        # Establezco la escala de colores.
        im = ax.tripcolor(
            xnod[:, X], xnod[:, Y], 
            LaG_t, 
            variable, 
            cmap='bwr',
            shading="gouraud", 
            vmin=-val_max, 
            vmax=val_max
            )
        
        # Curva de nivel cada 20 unidades.
        ax.tricontour(xnod[:, X], xnod[:, Y], LaG_t, variable, 20)
     
        # Ajuste del warning (advertencia) en al curva 0 (no existe).
        warnings.filterwarnings("ignore")
        ax.tricontour(
            xnod[:, X], xnod[:, Y], 
            LaG_t, 
            variable, 
            levels=[0], 
            linewidths=3
            )
        warnings.filterwarnings("default")
        
        # Ajuste del colorbar.
        #if var+1 == nvar:
        #    cax = fig.add_subplot(1, nvar+1, nvar+1)
        #    fig.colorbar(im, cax=cax, ax=ax, format="%6.3g")
        
        # Consideración con magnitudes principales: se grafican las 
        # direcciones prinicipales (opcional).
        # si hay ángulos, selecciono el arreglo correspondiente.
        if angulos is not None:
            var_angulos = angulos[var]
            # Ajuste de ndarray a list.
            if type(var_angulos) is np.ndarray:
                var_angulos = [var_angulos]
            for angulo in var_angulos:
                ax.quiver(
                    xnod[:, X], 
                    xnod[:, Y], 
                    variable*np.cos(angulo), 
                    variable*np.sin(angulo), 
                    headwidth=0, 
                    headlength=0, 
                    headaxislength=0, 
                    pivot="middle"
                    )
        
        # Ejes y títulos.
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')
        ax.set_title(var_name)

        ax.set_aspect('equal')
        ax.autoscale(tight=True)    
        plt.tight_layout()
    
    # Se presenta la gráfica.
    plt.show()
   
   
def CalculaEsfuerzosPrinicipal_TP(sx, sy, txy):
    """
    Cálculo de los esfuerzos prinicipales del campo vectorial de esfueros para 
    el caso de tensión plana (TP).
    
    Datos de entrada:
    - campo: campo a trabajar.
    
    Datos de salida (en este orden):
    - s1: esfuerzo normal máximo.
    - s2: esfuerzo normal mínimo.    
    - tmax: esfuerzo cortante máximo.
    - ang: ángulo de inclinación de s1.
    """
    
    s1   = (sx+sy)/2 + np.sqrt(((sx-sy)/2)**2 + txy**2) 
    s2   = (sx+sy)/2 - np.sqrt(((sx-sy)/2)**2 + txy**2) 
    tmax = (s1 - s2)/2                                  
    ang  = 0.5*np.arctan2(2*txy, sx-sy)                 
    
    return s1, s2, tmax, ang


def GraficarMaximos(titulo, campo, etiquetas, factor_inc):
    """
    Graficar los valores máximos de un campo en diferentes incrementos de 
    carga.
    
    Datos de estrada:
    - titulo: nombre del campo a graficar.
    - campo: arreglo con los valores máximas alcanzados en los incrementos
    - etiquetas: lista de titulos y labels del axis.
    - factor_inc: lista con los valores de lambdas (% de incrementos)
    """

    # Se inicializa el lienzo.
    fig = plt.figure()
    fig.suptitle(f"Valores máximos de {titulo}")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # El número o cantidad de incrementos hechos.
    ninc = factor_inc.shape[0]
    # El número o cantidad de variables a evaluar.
    nvar = campo.shape[1]

    # Por cada variable del campo que se quiere graficar (var=0, 1, 2...)
    for var in range(nvar):

        # Selecciono la variable del campo.
        variable = campo[:,var]
        # Selecciono el nombre de la variable.
        var_name = etiquetas[var]

        # Preparo el espacio axis para el plot.
        ax = fig.add_subplot(1, nvar, var+1)
        plt.plot(factor_inc, variable, 'k-')
    
        # Ejes y títulos.
        ax.set_xlabel("$\lambda$")
        ax.set_ylabel(etiquetas[var])
        
        ax.set_aspect('equal')
        ax.autoscale(tight=True)    
        plt.tight_layout()
    
    # Se presenta la gráfica.
    plt.show()


# ==============================================================================
#           Configuración del sistema
# ==============================================================================

# Archivo con toda la información del sistema.
ARCHIVO = "malla3.xlsx"

# Lectura del archivo en un dataframe para usarse desde el código.
try:
    DF = pd.read_excel(ARCHIVO, sheet_name=None)
except FileNotFoundError:
    raise Exception("Verifique que el ARCHIVO de excel 'malla3.xslx' esté en la carpeta")

# ==============================================================================
#           Propiedades geométricas, plásticas y elásticas
# ==============================================================================

# Posición de los nodos de la malla de EFs.
XNOD = DF["xnod"][["x", "y"]].to_numpy()
# Número o cantidad de nodos.
NNDS = XNOD.shape[0]

# Número o cantidad de grados de libertad (sistema de tensión plana, sin 
# rotaciones).
NGDL = 2*NNDS
# Matriz que asigna los grados de libertad (X, Y) a cada nodo.
GDL = np.reshape(np.arange(NGDL), (NNDS, 2))

# Matriz que relaciona los nodos locales con sus respectivos nodos globales 
# (la matriz de local a global LaG)
LaG = DF["LaG_mat"][["NL1", "NL2", "NL3", "NL4", "NL5", "NL6", "NL7", "NL8"]].to_numpy() - 1
# Número o cantidad de elmentos finitos.
NEF = LaG.shape[0]

# Del archivo se extrae qué material tiene cada elemento.
MAT = DF["LaG_mat"]["material"].to_numpy() - 1

# Se extraen también las propiedades de cada tipo de material en arreglos.
E   = DF["prop_mat"]["E"].to_numpy()       # Pa      módulo de elasticidad
nu  = DF["prop_mat"]["nu"].to_numpy()      # ad      coeficiente de Poisson
rho = DF["prop_mat"]["rho"].to_numpy()     # kg/m^3  densidad
t   = DF["prop_mat"]["espesor"].to_numpy() # m       espesor
H   = DF["prop_mat"]["H"].to_numpy()       # Pa      "strain-hardening parameter"
G   = DF["prop_mat"]["G"].to_numpy()       # Pa      módulo de corte 
fy  = DF["prop_mat"]["fy"].to_numpy()      # Pa      Esfuerzo de fluencia 

# Cada material indica el modelo de endurecimiento que sigue.
# isotrópico (1) o cinemático (2). 
MOD_HARD = DF["prop_mat"]["mod_hard"].to_numpy()

# Número o cantidad de materiales.
NMAT = E.shape[0]

# Contenedores para
C_elas = NMAT*[None]                     # matriz constitutiva elástica TP.  
b = NMAT*[None]                          # y vector de fuerzas másicas.
# Ensamblaje de las matrices constitutivas elásticas por cada material.
for i in range(NMAT):
    C_elas[i] = np.array([
        [E[i]/(1 - nu[i]**2),     E[i]*nu[i]/(1-nu[i]**2), 0                 ],
        [E[i]*nu[i]/(1-nu[i]**2), E[i]/(1-nu[i]**2),       0                 ],
        [0,                       0,                       E[i]/(2*(1+nu[i]))]
        ])
    b[i] = np.array([0, -rho[i]*g])  # kgf/m^3


# ==============================================================================
#           Configuración de cargas
# ==============================================================================

# Del archivo saco la información de las cargas puntuales (CP) y cargas 
# distruibdas (CD) de la estructura.
CP = DF["carga_punt"]
CD = DF['carga_distr']

# Número o cantidad de cargas puntuales.
NCP = CP.shape[0]
# Número o cantidad de lados con cargas distruibuidas.
NLCD = CD.shape[0]

# Creo los ectores de cargas
f = np.zeros(NGDL)  # Vector de fuerzas nodales equivalentes global.
ft = np.zeros(NGDL) # Vector de fuerzas duperficiales global.

# Añado las cargas puntuales al vector de fuerzas nodales equivalentes.
for i in range(NCP):
    f[GDL[CP["nodo"][i]-1, CP["dirección"][i]-1]] = CP["fuerza puntual"][i]

# Por cada lado cargado se obtienen las fuerzas nodales equivalentes en los 
# nodos y se añaden al vector de fuerzas superficieales.
for i in range(NLCD):
    e     = CD["elemento"][1]-1
    lado  = CD["lado"][i]
    carga = CD[["tix", "tiy", "tjx", "tjy", "tkx", "tky"]].loc[i].to_numpy()
    fte   = ConvierteVector_ft( XNOD[LaG[e,:],:], lado, carga, t[MAT[e]] )
    idx   = GDL[LaG[e]].flatten() # gdl en los que actúan las fuerzas.

    ft[np.ix_(idx)] += fte
    
# Se agrega el vector de fuerzas superficiales calculado al de fuerzas nodales 
# equivalentes.
f += ft


# ==============================================================================
#           Restricciones
# ==============================================================================

# Del ARCHIVO obtengo la información de los gdl restringidos.
RESTRICCIONES = DF['restric']
# Número o cantidad de gdl restringidos.
NRES = RESTRICCIONES.shape[0]

# Contenedor para los gdl que están restrigidos y por ende, conocemos (c)
c = np.empty(NRES, dtype=int)
# Asigno los gdl conocidos.
for i in range(NRES):
   c[i] = GDL[RESTRICCIONES['nodo'][i]-1, RESTRICCIONES['dirección'][i]-1]

# Desplazamientos conocidos.
ac = RESTRICCIONES['desplazamiento'].to_numpy()

# Grados de libertad de desplazamiento desconocidos (d).
d = np.setdiff1d(range(NGDL), c)


# ==============================================================================
#           Configuración para el análisis
# ==============================================================================

# Vector con porcentajes (%) de incrementos de carga. 
LAMBDAS = np.array([0.00, 0.20, 0.40, 0.60, 0.80, 1.00]) 
# Número de incrementos de carga.
NINC = LAMBDAS.shape[0]

# Criterio de convergencia de la solución.
CRITERIO_CONV = 0.01
# Número máximo de iteraciones que debe hacer el ciclo de análisis.
ITER_MAX = 8

# Orden de la cuadratura de Gauss Legendre, tomando el mismo orden en cada eje 
# (xi, eta).
N_GL = 2
X_GL, W_GL = np.polynomial.legendre.leggauss(N_GL)

# Número o cantidad de puntos de Gauss a estudiar en cada EF (valor manual).
NPTG = 4


# ==============================================================================
#           Funciones de forma EF
# ==============================================================================

# Funciones de forma serendípitas y sus derivadas del elemento Q8.
Nforma = lambda xi,eta: np.array(
                        [-((eta - 1)*(xi - 1)*(eta + xi + 1))/4,    # N1
                          ((xi**2 - 1)*(eta - 1))/2,                # N2
                          ((eta - 1)*(xi + 1)*(eta - xi + 1))/4,    # N3
                         -((eta**2 - 1)*(xi + 1))/2,                # N4
                          ((eta + 1)*(xi + 1)*(eta + xi - 1))/4,    # N5
                         -((xi**2 - 1)*(eta + 1))/2,                # N6
                          ((eta + 1)*(xi - 1)*(xi - eta + 1))/4,    # N7
                          ((eta**2 - 1)*(xi - 1))/2              ]) # N8

# Derivadas de las funciones de forma con respecto a xi.
dN_dxi = lambda xi,eta: np.array(
                        [-((eta + 2*xi)*(eta - 1))/4,    # dN1_dxi
                          eta*xi - xi,                   # dN2_dxi
                          ((eta - 2*xi)*(eta - 1))/4,    # dN3_dxi
                          1/2 - eta**2/2,                # dN4_dxi
                          ((eta + 2*xi)*(eta + 1))/4,    # dN5_dxi
                         -xi*(eta + 1),                  # dN6_dxi
                         -((eta - 2*xi)*(eta + 1))/4,    # dN7_dxi
                          eta**2/2 - 1/2              ]) # dN8_dxi

# Derivadas de N con respecto a eta.
dN_deta = lambda xi,eta: np.array(
                         [-((2*eta + xi)*(xi - 1))/4,    # dN1_deta
                           xi**2/2 - 1/2,                # dN2_deta
                           ((xi + 1)*(2*eta - xi))/4,    # dN3_deta
                          -eta*(xi + 1),                 # dN4_deta
                           ((2*eta + xi)*(xi + 1))/4,    # dN5_deta
                           1/2 - xi**2/2,                # dN6_deta
                          -((xi - 1)*(2*eta - xi))/4,    # dN7_deta
                           eta*(xi - 1)               ]) # dN8_deta


# Bajo la consideración de que tendrán pequeñas deformaciones, los 
# desplazamientos se miden respecto a un punto de referencia estático y se 
# puede hacer el cálculo de las matrices de forma y cinemáticas fuera del 
# ciclo principal, de esta forma evitar redundancias y gasto computacional.  

# Espacios de memoria para los arreglos:
# Este arreglo tiene NEF entradas, cada entrada es una matriz de tamaño 
# N_GLxN_GL (un espacio para cada punto de Gauss), y cada sub entrada de esta 
# matriz es del tamaño respectio de la matriz en cuestión (de forma o 
# cinemática) 
N = np.zeros((NEF, N_GL, N_GL, 2, 2*8))  # Matriz de forma en cada punto GL.
B = np.zeros((NEF, N_GL, N_GL, 3, 2*8))  # Matriz cinemática en cada punto GL. 

# Se analiza localmente cada EF.
for e in range(NEF):

    # Se crea un contenedor para matriz para almacenar los jacobianos
    det_Je = np.zeros((N_GL, N_GL))     

    # En cada punto de GL en el sentido xi.
    for p in range(N_GL):
        # En cada punto de GL en el sentido eta.
        for q in range(N_GL):

            # Los puntos de GL.
            xi_gl = X_GL[p];  eta_gl = X_GL[q]

            # Evalúo las funciones de forma y sus derivadas en cada sentido.
            NNforma  = Nforma (xi_gl, eta_gl)
            ddN_dxi  = dN_dxi (xi_gl, eta_gl)
            ddN_deta = dN_deta(xi_gl, eta_gl)

            # Las coordenadas del EF para el cálculo del jacobiano.
            xxe = XNOD[LaG[e], X];   yye = XNOD[LaG[e], Y]

            # Cálculo de las derivadas de las funciones de transformación.
            dx_dxi  = np.sum(ddN_dxi*xxe);    dy_dxi  = np.sum(ddN_dxi*yye)
            dx_deta = np.sum(ddN_deta*xxe);   dy_deta = np.sum(ddN_deta*yye)

            # Cálculo del jacobiano de la transformación.
            Je = np.array([
                [dx_dxi,  dy_dxi],
                [dx_deta, dy_deta]                
                ])
            det_Je[p, q] += np.linalg.det(Je)

            # Ensamblaje de las matrices de forma y cinemáticas en el punto de 
            # GL
            Npq = np.zeros((2, 2*8))
            Bpq = np.zeros((3, 2*8))

            for i in range(8):
                Npq[:,[2*i, 2*i+1]] += np.array([[NNforma[i], 0         ],
                                                 [0,          NNforma[i]]])

                dNi_dx = (+dy_deta*ddN_dxi[i] - dy_dxi*ddN_deta[i])/det_Je[p,q]
                dNi_dy = (-dx_deta*ddN_dxi[i] + dx_dxi*ddN_deta[i])/det_Je[p,q]

                Bpq[:,[2*i, 2*i+1]] += np.array([[dNi_dx, 0     ],
                                                 [0,      dNi_dy],
                                                 [dNi_dy, dNi_dx]])
            N[e,p,q] += Npq
            B[e,p,q] += Bpq
    
    # Se determina si hay puntos con jacobiano negativo, en caso tal se termina
    # el programa y se reporta.
    if np.any(det_Je <= 0):
        raise Exception(f'Hay puntos con det_Je negativo en el elemento {e+1}')


# ==============================================================================
#           Espacios de memoria
# ==============================================================================

# Espacios de memoria para almacenar cantiad importantes en todos los estados 
# del análisis.

EST_ALM  = [None for i in range(NINC)]  # Estados almacenados
ESF_ALM  = [None for i in range(NINC)]  # Esfuerzos almacenados.
DEF_ALM  = [None for i in range(NINC)]  # Deformaciones almacenadas.
DEFE_ALM = [None for i in range(NINC)]  # Deformaciones elásticas almacenadas.
DEFP_ALM = [None for i in range(NINC)]  # Deformaciones plásticas almacenadas.
DES_ALM  = [None for i in range(NINC)]  # Desplazamientos almacenados.
R_I_ALM  = [None for i in range(NINC)]  # Vectores de fuerzas internas ...
R_E_ALM  = [None for i in range(NINC)]  # Vectores de fuerzas externas ...

# Otros contenedores para almacenar los esfuerzos y deformaciones extrapolados 
# a los nodos, y los esfuerzos principales.

ESF_EXTR = [None for i in range(NINC)]
DEF_EXTR = [None for i in range(NINC)]
ESF_PRIN = [None for i in range(NINC)]

# Espacios de memoria para almacenar el estado inicial de la estructura, esto 
# es sin cargas, sin deformaciones y sin esfuerzos (todo son ceros), y el 
# estado general es elástico (1).

# Por cada EF, se tiene una matriz de N_GLxN_GL, cada entrada de esta matriz es
# para el estado del material de un punto de Gauss.
ESTADO = np.ones((NEF, N_GL, N_GL), dtype=int)   # Estado del material en cada EF.

# Por cada EF, se tiene una matriz de N_GLxN_GL, cada una con el estado de 
# esfuerzos, deformaciones o desplazamientos de cada punto de Gauss en 
# tensión plana 
ESFUERZOS     = np.zeros((NEF, N_GL, N_GL, 3))   # Esfuerzos.
DEFORMACIONES = np.zeros((NEF, N_GL, N_GL, 3)) 
DEF_ELASTICAS = np.zeros((NEF, N_GL, N_GL, 3))   # Deformaciones.
DEF_PLASTICAS = np.zeros((NEF, N_GL, N_GL, 3))   # Deformaciones.

# Los desplazamientos son nodales, no en los puntos de Gauss.
DESPLAZAMIENTOS = np.zeros(NGDL)   # Desplazamientos.

R_I = np.zeros(NGDL)   # Vector de fuerzas internas.
R_E = np.zeros(NGDL)   # Vector de fuerzas internas.

# En el estado incial, elástico, ningún EF en sus puntos de Gauss ha 
# sobrepasado la fluencia.
FYB = np.zeros((NEF, N_GL, N_GL))   # Estado del material en cada EF.          
for e in range(NEF):
    for p in range(N_GL):
        for q in range(N_GL):
            FYB[e, p, q] += fy[MAT[e]]


# ==============================================================================
#           Análisis iterativo 
# ==============================================================================

# Para cada incremento de carga lambda se hace un análisis iterativo.
for i in range(NINC):
    print(f'\n--- incremento {i+1}, lambda = {LAMBDAS[i]}. ---')

    # Se almacena el estado anterior de la estructura.
    EST_ALM [i] = ESTADO    
    ESF_ALM [i] = ESFUERZOS
    DEFE_ALM[i] = DEF_ELASTICAS
    DEFP_ALM[i] = DEF_PLASTICAS
    DEF_ALM [i] = DEFORMACIONES
    DES_ALM [i] = DESPLAZAMIENTOS
    R_I_ALM [i] = R_I   
    R_E_ALM [i] = R_E

    # Se aplica el factor lambda i al vector de fuerzas nodales equivalentes f.
    f_reducido = f*LAMBDAS[i]

    # Parámetros para inicar la medición de la convergencia.
    CONVERGENCIA = 1
    ITERACIONES = 0
    
    # El análisis iterativo se realiza mientras que la convergencia sea mayor 
    # al criteiro. 
    while CONVERGENCIA > CRITERIO_CONV:
        ITERACIONES +=1
        
        # Contenedores para las variables del ensamblaje matricial.
        K_t = np.zeros((NGDL, NGDL))    # Matriz de rigidez tangente global.
        idx = NEF*[None]                # Índices asociados a los gdl.
        R_I = np.zeros(NGDL)            # Vector de fuerzas internas global.
        # Se define le vector de fuerzas nodales equivalentes como el vector de
        # fuerzas externas global.
        R_E = f_reducido
        
        # Se analiza localmente cada EF para determinar el estado previo.
        for e in range(NEF):

            # Se crea contenedores para
            Ke = np.zeros((16, 16))   # matriz de rigidez tangente local del EF.
            r_e = np.zeros(16)        # vector de fuerzas nodales equivalentes EF.
            r_i = np.zeros(16)        # vector de fuerzas nodales INTERNAS del EF.

            # En cada punto de GL en el sentido xi.
            for p in range(N_GL):
                # En cada punto de GL en el sentido eta. 
                for q in range(N_GL):

                    # Los puntos de GL son:
                    xi_gl = X_GL[p];  eta_gl = X_GL[q]
                    
                    # Propiedades del material del elemento en el punto (p,q).
                    Ge = G[MAT[e]]; He = H[MAT[e]]; fye = fy[MAT[e]]

                    # Para el EF (e), en el punto (p,q) se evalúa el estado: 
                    # elástico (1) o plástico (2), y se toma la matriz 
                    # constitutiva tangente requerida. 
                    estado_0 = ESTADO[e, p, q] 
                    C = C_elas[MAT[e]]
                    Ct = DeterminaCt(estado_0, C, Ge, He, fye)

                    # Ensamblaje de la matriz de rigidez y vector de fuerzas 
                    # nodales equivalentes del EF. 
                    Ke  += B[e,p,q].T @ Ct @ Bpq * det_Je[p,q]*t[MAT[e]]*W_GL[p]*W_GL[q]
                    r_e += N[e,p,q].T @ b[MAT[e]] * det_Je[p,q]*t[MAT[e]]*W_GL[p]*W_GL[q]     
                    r_i += B[e,p,q].T @ ESFUERZOS[e, p, q] * det_Je[p,q]*t[MAT[e]]*W_GL[p]*W_GL[q]     

            # índice de los gdl del EF (e).
            idx[e] = GDL[LaG[e]].flatten()      
            
            # Ensambaje global en los arreglos principales.
            K_t[np.ix_(idx[e], idx[e])] += Ke   
            R_E[np.ix_(idx[e])] += r_e
            R_I[np.ix_(idx[e])] += r_i

        # Se soluciona el sistema matricial K_t.Dd = -R_I + R_E, intereza 
        # conocer el incremento de desplazamientos.
        R = R_E - R_I 
        D_desplazamientos = np.linalg.inv(K_t) @ R 

        # Añado este incremento de desplazamientos a su arreglo.
        DESPLAZAMIENTOS += D_desplazamientos
        
        # Conocido el incremento de desplazamientos, se calcula el incremento 
        # de deformaciones en cada punto de Gauss.
        for e in range(NEF):
            # Extraigo el incremento de desplazamientos del EF (e).
            D_desplz_e = D_desplazamientos[idx[e]]

            for p in range(N_GL):
                for q in range(N_GL):
                    
                    # Incremento de deformaciones 
                    D_def_0 = B[q, p, q] @ D_desplz_e
                    
                    # Organizo la información del estado previo para el 
                    # análisis de incrementos elastoplasticos 
                    # esfuerzo-deformación y la determinación del estado.

                    esfuerzos_0     = ESFUERZOS[e, p, q]
                    estado_0        = ESTADO[e, p, q] 
                    def_plasticas_0 = DEF_PLASTICAS[e, p, q]
                    
                    Ge = G[MAT[e]]; He = H[MAT[e]]; fy_0 = fy[MAT[e]] 
                    C  = C_elas[MAT[e]]
                    
                    # Se calcula el nuevo estado.
                    #     S_new,   D_defp,  fyb, estado_f
                    esfuerzos_f, D_defp_f, fy_f, estado_f = IncrementoElastoplastico_IH(D_def_0, def_plasticas_0, esfuerzos_0, C, Ge, He, fy_0, estado_0)

                    # Estos son resultados generales, por lo que se deben 
                    # simplificar nuevamente a tensión plana.
                    esfuerzos_f = np.delete(esfuerzos_f, (2, 4, 5), 0)  #sx, sy, txy.
                    D_defp_f    = np.delete(D_defp_f, (2, 4, 5), 0)     #Dexp, Deyp, Dgxyp.

                    # Actualizo e incremento los respectivos valores.
                    ESTADO[e, p, q]        = estado_f    
                    ESFUERZOS[e, p, q]     = esfuerzos_f                     
                    FYB[e, p, q]           = fy_f  
                    DEF_ELASTICAS[e, p, q] += D_def_0  
                    DEF_PLASTICAS[e, p, q] += D_defp_f       


        # Cálculo de la convergencia 
        CONVERGENCIA = np.linalg.norm(R)**2 / (1 + np.linalg.norm(R_E)**2)      
        print(f'convergencia = {CONVERGENCIA}')
        
        # Si se supera el número máximo de iteraciones, el ciclo debe terminar.
        if ITERACIONES == ITER_MAX:
            # Se falsifica la convergencia para cerrar el cilo.
            CONVERGENCIA = 0.0001    
            #print("\nNo converge el cálculo.")
        else:
            continue
    
    # Guardo registro del vectoe de fuerzas externas.
    R_E_ALM [i] = R_E
      
    # Las deformaciones completas de la estructura son l asuma del coponente 
    # elástico y plástico.
    DEFORMACIONES = DEF_ELASTICAS + DEF_PLASTICAS

    # Se extrapolan las deformaciones y los esfuerzos a los nodos.
    sx, sy, txy, ex, ey, gxy = ExtrapolaEsfDef(DEFORMACIONES, ESFUERZOS, NNDS, NEF, LaG)

    # La deformación ez para tensión plana se puede calcular como:
    ez = -(nu/E)*(sx + sy)

    # Se calculan los esfuerzos principales y sus direcciones.
    s1, s2, tmax, ang = CalculaEsfuerzosPrinicipal_TP(sx, sy, txy)

    # Debo guardar estos últimos cálculos para la iteración i.
    ESF_EXTR[i] = [sx, sy, txy]
    DEF_EXTR[i] = [ex, ey, ez, gxy]
    ESF_PRIN[i] = [s1, s2, tmax, ang]


# =============================================================================
#           Graficación del incremento último 
# =============================================================================

# Malla.
# -----------------------------------------------------------------------------
# Organizo el vector de desplazamientos para que sea compatible con el de 
# coordenadas nodales XNOD.
XDEF = np.reshape(DESPLAZAMIENTOS, (NNDS,2))

# Se presenta una gráfica estática de la estructura deformada con la carga 
# completamente aplicada.
GraficarEstructura() 

# Esfuerzos y deformaciones
# ------------------------------------------------------------------------------

# Deformaciones y esfuerzos en el último incremento.
ex_inc_f  = DEF_EXTR[-1][0];    sx_inc_f  = ESF_EXTR[-1][0]
ey_inc_f  = DEF_EXTR[-1][1];    sy_inc_f  = ESF_EXTR[-1][1]
ez_inc_f  = DEF_EXTR[-1][2];    txy_inc_f = ESF_EXTR[-1][2]
gxy_inc_f = DEF_EXTR[-1][3];    

# Esfuerzos principales
s1, s2, tmax, ang = CalculaEsfuerzosPrinicipal_TP(sx_inc_f, sy_inc_f, txy_inc_f)

# Campo de esfuerzos
campo_esf = [sx_inc_f, sy_inc_f, txy_inc_f]
var_campo_esf = [r'$\sigma_x$ [Pa]', r'$\sigma_y$ [Pa]', r'$\tau_{xy}$ [Pa]']

# Campo de deformaciones
campo_def = [ex_inc_f, ey_inc_f, ez_inc_f, gxy_inc_f]
var_campo_def = [r'$\epsilon_x$', r'$\epsilon_y$', r'$\epsilon_z$', r'$\gamma_{xy}$ [rad]']

# Campo de esfuerzos principales
campo_esfp = [s1, s2, tmax]
var_campo_esfp = [r'$\sigma_1$ [Pa]', r'$\sigma_2$ [Pa]', r'$\tau_{máx}$ [Pa]']
angulos = [ang, ang+np.pi/2, [ ang-np.pi/4, ang+np.pi/4 ]]

# Las gráficas.
GraficarMagnitudesEF(
    campo_esf, 
    "Campo vectorial de esfuerzos", 
    var_campo_esf, 
    LaG, XNOD, angulos=None
    )
GraficarMagnitudesEF(
    campo_def, 
    "Campo vectorial de deformaciones", 
    var_campo_def, 
    LaG, XNOD, angulos=None
    )
GraficarMagnitudesEF(
    campo_esf, 
    "Campo vectorial de esfuerzos principales", 
    var_campo_esfp, 
    LaG, XNOD, 
    angulos=angulos)


# =============================================================================
#           Presentación de valores máximos nodales. 
# =============================================================================

# Contenedores para los valores máximos en cada incremento de:
des_max = np.zeros((NINC, 2))   # Desplazamientos.
def_max = np.zeros((NINC, 4))   # Deformaciones.
esf_max = np.zeros((NINC, 3))   # Esfuerzos.

# Saco los valores máximos de las variables mencionadas en los nodos de cada 
# incremento.
for i in range(NINC):

    # Desplazamientos.
    des_max[i, 0] = np.max( DEFP_ALM[i][0] )
    des_max[i, 1] = np.max( DEFP_ALM[i][1] )

    # Deformaciones.
    def_max[i, 0] = np.max( DEF_EXTR[i][0] )
    def_max[i, 1] = np.max( DEF_EXTR[i][1] )
    def_max[i, 2] = np.max( DEF_EXTR[i][2] )
    def_max[i, 3] = np.max( DEF_EXTR[i][3] )

    # Esfuerzos.
    esf_max[i, 0]= np.max( ESF_EXTR[i][0] )
    esf_max[i, 1]= np.max( ESF_EXTR[i][1] )
    esf_max[i, 2]= np.max( ESF_EXTR[i][2] )

# Organizadores. nombre = [titulo, x_label, y_label]
var_campo_des = [r"$d_x$ [m]", r"$d_y$ [m]"]    

# Se presenta una gráfica que muestra el incremento del valor máximo en cada 
# momento de carga.
GraficarMaximos("desplazamientos", des_max, var_campo_des, LAMBDAS)
GraficarMaximos("deformaciones",   def_max, var_campo_def, LAMBDAS)
GraficarMaximos("esfuerzos",       esf_max, var_campo_esf, LAMBDAS)


# =============================================================================
#           Exportación de resultados a excel. 
# =============================================================================

# Los resultados para cada incremento de carga son almacenados en un archivo de
# excel.

# Fuerzas nodales internas y externas en cada incremento. 
# -----------------------------------------------------------------------------
re_ri_columns = [None for i in range(2*NINC)]
re_ri_data    = [None for i in range(2*NINC)]

for i in range(NINC):
    # Ajusto los valores en una lista.
    re_ri_data[2*i]   = R_E_ALM[i]  # R_E
    re_ri_data[2*i+1] = R_I_ALM[i]  # R_I
    # Ajusto el nombre de las columnas.
    re_ri_columns[2*i]   = f"R_E, lambda={LAMBDAS[i]}"
    re_ri_columns[2*i+1] = f"R_I, lambda={LAMBDAS[i]}"

# Creo al tabla.
tabla_re_ri = pd.DataFrame(
    data = re_ri_data, index = np.arange(NNDS)+1, columns = re_ri_columns
)

tabla_re_ri.index.name = "Nodo"

# Desplazamientos nodales en cada incremento.
# -----------------------------------------------------------------------------
des_columns = [None for i in range(2*NINC)]
des_data    = [None for i in range(2*NINC)]

for i in range(NINC):
    xdef = np.reshape(DES_ALM[i], (NNDS,2))
    # Ajusto los valores en una lista.
    des_data[2*i]   = xdef[0]    # dx
    des_data[2*i+1] = xdef[1]    # dy
    # Ajusto el nombre de las columnas.
    des_columns[2*i]   = f"dx, lambda={LAMBDAS[i]}"
    des_columns[2*i+1] = f"dy, lambda={LAMBDAS[i]}"

tabla_des = pd.DataFrame(
    data = des_data, index = np.arange(NNDS)+1, columns = des_columns
)
tabla_des.index.name = "Nodo"

# Deformaciones nodales en cada incremento.
# -----------------------------------------------------------------------------
def_columns = [None for i in range(4*NINC)]  
def_data    = [None for i in range(4*NINC)]

for i in range(NINC):
    def_lamba_i = DEF_EXTR[i]
    # Ajusto los valores en una lista.
    def_data[2*i]   = def_lamba_i[0]    # ex
    def_data[2*i+1] = def_lamba_i[1]    # ey
    def_data[2*i+2] = def_lamba_i[2]    # ez
    def_data[2*i+3] = def_lamba_i[3]    # gxy
    # Ajusto el nombre de las columnas.
    def_columns[2*i]   = f"ex,  lambda={LAMBDAS[i]}"
    def_columns[2*i+1] = f"ey,  lambda={LAMBDAS[i]}"
    def_columns[2*i+2] = f"ez,  lambda={LAMBDAS[i]}"
    def_columns[2*i+3] = f"gxy, lambda={LAMBDAS[i]}"

tabla_def = pd.DataFrame(
    data = def_data, index = np.arange(NNDS)+1, columns = def_columns
)
tabla_def.index.name = "Nodo"

# Esfuerzos nodales en cada incremento.
# -----------------------------------------------------------------------------
esf_columns = [None for i in range(3*NINC)]  
esf_data    = [None for i in range(3*NINC)]

for i in range(NINC):
    esf_lamba_i = ESF_EXTR[i]
    # Ajusto los valores en una lista.
    esf_data[2*i]   = esf_lamba_i[0]    # sx
    esf_data[2*i+1] = esf_lamba_i[1]    # sy
    esf_data[2*i+2] = esf_lamba_i[2]    # txy
    # Ajusto el nombre de las columnas.
    esf_columns[2*i]   = f"sx,  lambda={LAMBDAS[i]}"
    esf_columns[2*i+1] = f"sy,  lambda={LAMBDAS[i]}"
    esf_columns[2*i+2] = f"txy, lambda={LAMBDAS[i]}"
    
tabla_esf = pd.DataFrame(
    data = esf_data, index = np.arange(NNDS)+1, columns = esf_columns
)
tabla_esf.index.name = "Nodo"

# Esfuerzos nodales principales en cada incremento.
# -----------------------------------------------------------------------------
esf_prin_columns = [None for i in range(3*NINC)]  
esf_prin_data    = [None for i in range(3*NINC)]

for i in range(NINC):
    esf_prin_lamba_i = ESF_PRIN[i]
    # Ajusto los valores en una lista.
    esf_prin_data[2*i]   = esf_prin_lamba_i[0]    # s1
    esf_prin_data[2*i+1] = esf_prin_lamba_i[1]    # s2
    esf_prin_data[2*i+2] = esf_prin_lamba_i[2]    # smax
    # Ajusto el nombre de las columnas.
    esf_prin_columns[2*i]   = f"s1,  lambda={LAMBDAS[i]}"
    esf_prin_columns[2*i+1] = f"s2,  lambda={LAMBDAS[i]}"
    esf_prin_columns[2*i+2] = f"tmax, lambda={LAMBDAS[i]}"

tabla_esf_prin = pd.DataFrame(
    data = esf_prin_data, index = np.arange(NNDS)+1, columns = esf_prin_columns
)
tabla_esf_prin.index.name = "Nodo"

# Se crea un archivo de MS EXCEL que contenga estas tablas.
# -----------------------------------------------------------------------------
archivo_resultados = f"ResultadosAnalisisNL_{ARCHIVO}.xlsx"
writer = pd.ExcelWriter(archivo_resultados, engine="xlsxwriter")

# Se añaden las tablas como hojas al archivo de MS EXCEL
tabla_re_ri.to_excel(writer, sheet_name = 'Re_Ri')
tabla_des.to_excel(writer, sheet_name = 'desplazamientos')
tabla_def.to_excel(writer, sheet_name = 'deformaciones')
tabla_esf.to_excel(writer, sheet_name = 'esfuerzos')
tabla_esf_prin.to_excel(writer, sheet_name = 'esf_principales')
writer.save()


# =============================================================================
#           Animaciones
# =============================================================================

# SIN HACER.

# =============================================================================
print(f"Cálculo finalizado. En '{archivo_resultados}' se guardaron los resultados.")


## Fin 
print('done :)')

"""
POR HACER:
- Optimizar las rutinas NL al caso de tensión plana, sin simplificaciones.
- Crear animaciones de desplazamientos, deformaciones y esfuerzos.
- Verificar proceso de cálculo con ¿software?.
"""