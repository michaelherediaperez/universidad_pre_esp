# -*- coding: utf-8 -*-

"""
Rutinas de análisis no lineal para incorporar al código principal.
Modelo de fluencia de von Mises.

Metodología de Bhatti, cap 8. 
Referirse al PDFs para mejor comprensión.

Hecho por: Michael Heredia Pérez
email: mherediap@unal.edu.co
Fecha: diciembre 2021
Universidad Nacional de Colombia Sede Manizales
"""

# Librerías
import numpy as np 


def CalculaEsfDesviadores(esfuerzos):
    """
    Calcula los esfuerzos desviadores de un campo de esfuerzos.
    
    Datos de entrada:
    - esfuerzos: vector de esfuerzos [sx, sy, sz, txy, txz, tyz].T

    Datos de salida:
    - p: tensor de presioenes.
    - esf_desviadores: arreglo de esfuerzos desvaidores (deviatoriz stress)
                       [sxd, syd, szd, txyd, txzd, tyzd]. 
    """
    
    # Extraigo los valores.
    sx, sy, sz, txy, txz, tyz = esfuerzos

    # Tensor de presiones p.
    p = 1/3*(sx + sy + sz)

    # Esfuerzos desviadores
    sxd = sx - p;   txyd = txy
    syd = sy - p;   txzd = txz
    szd = sz - p;   tyzd = tyz

    esf_desviadores = np.array([sxd, syd, szd, txyd, txzd, tyzd]).T

    return esf_desviadores


def CalculaDefPlasticasEfectivas(def_plasticas):
    """
    Calcula la deformación plástica efectiva ingresado el campo de 
    deformaciones plásticas.
    
    Datos de entrada:
    - def_plasticas: vector de deformaciones [exp, eyp, ezp, gxyp, gxzp, gyzp].T
    
    Datos de salida:
    - defp_efectiva: deformación plástica efectiva.
    """

    # Extraigo las deformacioens plásticas.
    exp, eyp, ezp, gxyp, gxzp, gyzp = def_plasticas

    # Calculo la deformación plástica efectiva.
    defp_efectiva = np.sqrt( 2/3*(exp**2 + eyp**2 + ezp**2) + 1/3*(gxyp**2 + gxzp**2 + gyzp**2) )

    return defp_efectiva


def CalculaPuntoBeta(esfuerzos, D_esf, fy):
    """
    Calcula el punto \beta mediante el método de la ecuación cuadrática de 
    von mises para materiales isotrópicos.
    
    Datos de entrada:
    - esfuerzos: vector de esfuerzos [sx, sy, sz, txy, txz, tyz].T
    - D_esf: vector de incrementos de esfuerzos 
             [Dsx, Dsy, Dsz, Dtxy, Dtxz, Dtyz].
    - fy: esfuerzo de fluencia del material metálico (en el estado).
    
    Datos de salida:
    - beta: raiz de la ecuación cuadrática.
    """

    # Extraigo los valores.
    sx, sy, sz, txy, txz, tyz = esfuerzos
    Dsx, Dsy, Dsz, Dtxy, Dtxz, Dtyz = D_esf

    # Según Bhatti los coeficientes "a" son:
    
    a1 = (Dsx**2 + Dsy**2 + Dsz**2) -(Dsy*Dsx + Dsz*Dsx + Dsy*Dsz) +3*(Dtxy**2 + Dtxz**2 + Dtyz**2)
    a2 = 2*(Dsx*sx + Dsy*sy + Dsz*sz) -Dsx*(sy+sz) -Dsy*(sx+sz) -Dsz*(sx+sy) +6*(Dtxy*txy + Dtxz*txz + Dtyz*tyz)
    a3 = sx**2 + sy**2 + sz**2 +3*(txy**2 + txz**2 + tyz**2) -(sx*sy + sx*sz + sy*sz) - fy**2
        
    # Se soluciona la ecuación cuadrática a1\beta^2 + a2\beta + a3 = 0
    beta = (-a2 + np.sqrt(a2**2 - 4*a1*a3))/(2*a1)
    
    return beta


def dF_dS__vm(esfuerzos, fy):
    """
    Calcula la derivada parcial de la función de fluencia de von Mises 
    respecto al vector de esfuerzos (gradiente).

    Datos de entrada:
    - esfuerzos: vector de esfuerzos en el estado previo
                 [sx, sy, sz, txy, txz, tyz].T
    - fy: Esfuerzo de fluencia del material metálico (en el estado).

    Datos de salida:
    dF_dS: gradiente: [sxd, syd, szd, 2*txyd, 2*txzd, 2*tyzd].T
    """

    # Calculo los esfuerzos desviadores.
    sxd, syd, szd, txyd, txzd, tyzd = CalculaEsfDesviadores(esfuerzos)

    # Calculo del gradiente.
    dF_dS = 3/(2*fy) * np.array([sxd, syd, szd, 2*txyd, 2*txzd, 2*tyzd]).T

    return dF_dS


def F_vm_tp(esfuerzos, fy, H, def_plasticas):
    """
    Función de fluencia para el modelo de von Mises.
    
    Datos de entrada:
    - esfuerzos: vector de esfuerzos en el estado previo
                 [sx, sy, sz, txy, txz, tyz].T
    - fy: esfuerzo de fluencia del material en el estado anterior.
    - H: "strain-hardening parameter"
    - def_plasticas: campo de deformaciones [exp, eyp, ezp, gxyp, gxzp, gyzp].T
    
    Datos de salida:
    - F: valor de la función de fluencia de von Mises.
    """

    # Extraigo los esfuerzos desviadores.
    sxd, syd, szd, txyd, txzd, tyzd = CalculaEsfDesviadores(esfuerzos)

    # Segundo invariante del tensor de esfuerzos desviadores. 
    J2 = 1/2*(sxd**2 + syd**2 + szd**2) + txyd**2 + txzd**2 + tyzd**2 

    # Se calculan la deformación plástica efectiva.
    defp_efectiva = CalculaDefPlasticasEfectivas(def_plasticas)
    # Esfuerzo de fluencia actual. 
    fyb = fy + H*defp_efectiva

    # Función de fluencia de von Mises
    F = np.sqrt(3*J2) - fyb

    return F, fyb


def CalculaMatrizConstPlastica(esfuerzos, G, H, fy):
    """
    Calcula la matriz 'constitutiva plastica' a partir de los esfuerzos 
    desvaidores. 
    
    Datos de entrada:
    - esfuerzos: vector de esfuerzos en el estado previo
                 [sx, sy, sz, txy, txz, tyz].T
    - G: módulo de corte.
    - H: "strain-hardening parameter"
    - fy: esfuerzo de fluencia del material en el estado previo.
        
    Datos de salida:
    - Cp: matriz constitutiva plásticas del elemento.
    """

    # Obtengo los esfuerzos desviadores.
    esf_desviadores = CalculaEsfDesviadores(esfuerzos)

    # Ajusto el vector de esfuerzos desviadores para que sea posible obtener 
    # la matriz.
    vector_esf_des = esf_desviadores[np.newaxis]
    
    # Considerando que durante el flujo plástico no hay cambio de volumen, 
    # el cálculo de la matriz "constitutiva plástica" se simplifica.  
    Cp = (9*G**2)/((3*G + H)*fy**2) * vector_esf_des.T @ vector_esf_des 

    return Cp


def DeterminaCt(estado, C, G, H, fy, tp=True):
    """
    Esta función determina la matriz constitutiva a ser empelada, elástica o 
    'plástica'.

    Datos de entrada:
    - estado: estado actual del material en el EF, elástico (1) o plástico (2)
    - C: matriz constitutiva elástica.
    - G: módulo de corte.
    - H: "strain-hardening parameter"
    - fy: esfuerzo de fluencia del material en el estado previo.
    - tp=True: para definir si la matriz Ct debe ser simplificada a 3x3 o no.
    
    Datos de salida:
    - Ct: matriz constitutiva tangente 
    """

    if estado == 1:
        Ct = C

    elif estado == 2:
        Cp_general = CalculaMatrizConstPlastica(estado, G, H, fy)    
        
        if tp:
            # Esta es una matriz constitutiva 6x6 que se debe reducir a una de 
            # tensión plana 3x3 quitando las filas (axis=0) y columnas (axis=1) 
            # nulas.
            Cp_tp = np.delete(Cp_general, (2, 4, 5), 0)
            Cp_tp = np.delete(Cp_tp, (2, 4, 5), 1)
            # Se calcula la matriz tangente Ct = C - Cp
            Ct = C - Cp_tp
        else:
            Ct = C - Cp_general        
    return Ct


def AjustaC_66(C, esfuerzos, D_def, def_plasticas):
    """
    Ajusta la matriz constitutiva C de tensión plana a un tamaño 6x6 general.
    
    Datos de entrada:
    - C: matriz constitutiva de tensión plana 3x3.
    - D_def: vector de incrementos de deformaciones en tensión plana.
    - def_plasticas: vector de deformaciones plásticas en tensión plana.
    
    Datos de salida:
    - C66: matriz constutiva de tp expandida.
    - esfuerzos6: vector de esfuerzos de tp expandido.
    - D_def6: vector de incremento de deformaciones en tp expandido.
    - def_plasticas6 = vector de deformaciones plásticas en tp expandido.
    """

    # Ajusta la matriz constitutiva.
    C66 = np.c_[C, np.zeros(3), np.zeros(3)]    # En este orrden.
    C66 = np.insert(C66, 2, np.zeros(3), 1)
    C66 = np.insert(C66, 2, np.zeros(6), 0)
    C66 = np.insert(C66, 4, np.zeros(6), 0)
    C66 = np.insert(C66, 5, np.zeros(6), 0)

    # Ajusta el vector de esfuerzos.
    esfuerzos6 = np.insert(esfuerzos, 2, 0, 0)
    esfuerzos6 = np.insert(esfuerzos6, 4, 0, 0)
    esfuerzos6 = np.insert(esfuerzos6, 5, 0, 0)

    # Ajusta el vector de incremento de deformaciones.
    D_def6 = np.insert(D_def, 2, 0, 0)
    D_def6 = np.insert(D_def6, 4, 0, 0)
    D_def6 = np.insert(D_def6, 5, 0, 0)

    # Ajusta el vector de deformacioes plásticas.
    def_plasticas6 = np.insert(def_plasticas, 2, 0, 0)
    def_plasticas6 = np.insert(def_plasticas6, 4, 0, 0)
    def_plasticas6 = np.insert(def_plasticas6, 5, 0, 0)

    #print(C66.shape, esfuerzos6.shape, D_def6.shape, def_plasticas6.shape)
    return C66, esfuerzos6, D_def6, def_plasticas6


def IncrementoElastoplastico_IH(D_def, def_plasticas, esfuerzos, C, G, H, fy, estado_0, tp=True):
    """
    Calculo la regla de incrementos elastoplásticos de esfuerzo-deformación 
    en un punto de gauss dado. 
    
    CONSIDERACIONES:
    - Criterio de fluencia de von Mises.
    - Endurecimeinto isotrópico (materiales metálicos, von Mises)

    Datos de entrada:
    - D_def: vector de incrementos de deformaciones del estado previo.
             [Dex, Dey, Dez, Dgxy, Dgxz, Dgyz].T
    - def_plasticas: vector de deformaciones plásticas del estado previo.
                     [exp, eyp, ezp, gxyp, gxzp, gyzp].T
    - esfuerzos: vector de esfuerzos en el estado previo
                 [sx, sy, sz, txy, txz, tyz].T
    - C: matriz constitutiva elástica.
    - G: módulo de corte.
    - H: "strain-hardening parameter"
    - fy: esfuerzo de fluencia del material en el estado previo.
    - estado_0: estado previo, elástico (1) o plástico (2).
    - tp=True: por defecto, la matriz C yt los vectores de esfeurzos y 
               deformaciones son de tensión plana, entonces se ajustan
               al tamaño 6x6.

    Datos de salida:
    - S_new: nuevo estado de esfuerzos.
    - Beta: punto de intersección de la superficie de fluencia.
    - estado_f: estado final del proceso, elástico (1) o plástico (2)
    """
    
    # Antes, se ajusta la matriz C si es de tensión plana.
    if tp:
        C, esfuerzos, D_def, def_plasticas = AjustaC_66(C, esfuerzos, D_def, def_plasticas)

    # 1. Asumimos un comportamiento elástico de entrada, se calcula un 
    # incremento de esfuerzos elásticos y se propone el nuevo estado de esfuerzos. 
    D_esf = C @ D_def
    S = esfuerzos + D_esf  # (trial state).
    
    # 2. Se calcula la función de fluencia (recordar que esta internamente 
    # trabaja los esfuerzos desviadores).   
    F, fyb = F_vm_tp(S, fy, H, def_plasticas)
    
    # 3. Se estudia el estado de esfuerzos previo y se actualiza.
    # Considerando que durante el flujo plástico no hay cambio de volumen, 
    # el cálculo de la matriz "constitutiva plástica" se simplifica.  
    Cp = CalculaMatrizConstPlastica(S, G, H, fy)

    # Considerando que solo se trabaja con la función de fluencia de von Mises, 
    # pero se debería ajustar para un material general.
    
    # Datos de salida inicializados.
    S_new    = np.zeros(6)
    beta     = float("NaN")
    estado_f = float("NaN") 
    D_defp   = np.zeros(6)

    # Si el estado preVio es ELÁSTICO.
    if estado_0 == 1: 

        # puede continuar elástico. 
        if F <= 0: 
            estado_f = 1
            # por lo tanto el esfuerzo elástico se toma como el actual y no se
            # considera beta.     
            S_new += S   
                          
        # puede cambiar al estado plástico si la función es positiva.
        elif F > 0:     
            estado_f = 2
            # se debe calcular el incremento de esfuerzos plásticos y el punto 
            # de intersección F(S + beta*Ds)=0
            D_esf = (C-Cp) @ D_def
            S_new = esfuerzos + D_esf
            beta  = CalculaPuntoBeta(esfuerzos, D_esf, fy)
            D_defp = (1-beta)*D_def

    # Si el estado previo es PLÁSTICO.
    elif estado_0 == 2: 

        # Para este análisis se rquiere del gradiente de la función de fluencia
        # respecto al vector de esfuerzos.
        dF_dS = dF_dS__vm(esfuerzos, fy)

        # Puede continuar plástico, si el signo del producto entre el gradiente
        # de la función de fluencia y el vector de incrementos de esfuerzos es 
        # positivo.        
        if (dF_dS.T @ D_esf) > 0: 
            estado_f = 2
            D_esf = (C-Cp) @ D_def
            S_new = esfuerzos + D_esf
            beta  = 0
            D_defp = (1-beta)*D_def

        # puede sufrir  una descarga elástica regresando al rango elástico si 
        # el producto anterior es negativo y la función evaluada en el cmapo de 
        # esfuerzos es negativa o igual a cero.
        elif ((dF_dS.T @ D_esf) > 0) and (F <= 0):  
            estado_f = 1  
            # tomando como esfuerzo actual el esfuerzo elástico.
            S_new = S   
        
        # el último caso es si hay una descarga fallida, que pase por el rango 
        # elástico pero se salga de la superficie de fluencia y regrese a 
        # plástico. 
        else:
            estado_f = 2           
            D_esf = (C-Cp) @ D_def
            S_new = esfuerzos + D_esf
            beta  = CalculaPuntoBeta(esfuerzos, D_esf, fy)
            D_defp = (1-beta)*D_def
            
    else: 
        raise Exception("Estado previo inválido, elástico (1) o plástico (2)")

    ## Nuevo estado de deformaciones.
    #dF_dS = dF_dS__vm(esfuerzos, fy)    # El gradiente.
    #a     = 3*G+H
    #lam   = 1/a * dF_dS.T@C * D_def     # Factor de equivalencia def_plasticas. 
    #
    ## Incremento de deformaciones plásticas 
    #D_defp = lam @ dF_dS

    # La función entrega los siguientes resultados.
    return S_new, D_defp, fyb, estado_f

# Fin :)