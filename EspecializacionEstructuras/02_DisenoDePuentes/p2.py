"""
Diseño de puentes. Examen, punto 2.

By:     Michael Heredia Pérez
email:  mherediap@unal.edu.co
Fecha:  mayo/2022

Universidad Nacional de Colombia Sede Manizales
Especialización en Estructuras.
"""

# Librerías de trabajo.
import numpy as np
import matplotlib.pyplot as plt


# Considerando n = suma de todos los dígitos del documento de identidad.
# Calculo n:
CC = 1010134928

n = 0

while CC != 0:          # Mientras haya dígitos en la CC.
    cifra = CC % 10     # Tomo el módulo 10 de CC (último dígito).
    CC //= 10           # Con la división entera eliminio la última cifra.
    n += cifra          # Sumo lacifra eliminada en cada paso.

# Reporto mi valor de n.
print(f"Valor de n para el ejercicio: {n}.")

# Datos del puente:
L = n       # m     Luz del puente.
B = n/2     # m     Ancho del puente.
h = L/40    # m     Espesor del puente.

# Datoas del neopreno:
nprenos = 4 #           Número de neoprenos en que se apoya.
kx_n = 6    # ton/cm    Rigidez horizontal del neopreno zunchado.

# La rigidez total es la suma de las rigideces de los neoprenos [# ton/cm].
kx_r = kx_n*nprenos 

# Material y otras propiedades:
gamma_c = 2.4   # ton/m^3    Peso propio del concreto.
g       = 981   # cm/s^2     Gravedad en sistema cgs.

# Masa y peso del puente.
vol = L*B*h            # m^3        volumen del puente.
m   = gamma_c*vol/g    # ton.s^2/cm masa ajustada por g para las unidades.

# El puente puede simplificarse en un sistema masa-resorte de 1 gdl.
# Se calcula el periodo T de la estructura [seg].
T = ( m/kx_r*4*np.pi**2 )**0.5

print(f"El Periodo T del puente es: {round(T, 2)} seg.")

# Datos para el espectro de aceleraciones CCP-14 para suelo tipo C, Manizales.
PGA  = 0.25     # Peak Ground Acceleration.
S1   = 0.3      # Coeficiente de aceleración espectral en 0.2 s.
Ss   = 0.6      # Coeficiente de aceleración espectral en 1 s.      
FPGA = 1.15     # Valores de factor de sitio, periodo cero.      
Fa   = 1.16     # Valores de factor de sitio, periodo corto.  
Fv   = 1.50     # Valores de factor de sitio, periodo largo.   

# Se construye el espectro de aceleraciones.
SDS = Fa*Ss
SD1 = Fv*S1

# Calculo el desplazamiento espectral Sd sabiendo que el T del puente es 
# mayor a 1 seg. Multiplico por g para tener unidades de cm.
Sd = T**2/(4*np.pi**2)*SD1 / T * g

print(f"Desplazamiento espectral en el sentido horizontal: {round(Sd, 1)} cm.")

# fin :)