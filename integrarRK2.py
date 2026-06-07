from rk2 import paso_rk2
import numpy as np
def integrar_rk2(Y0, h, tiempo_total, mu_T):
    cantidad_pasos = int(tiempo_total / h)
    estados = []
    tiempos = []
    Y = Y0.copy()
    for n in range(cantidad_pasos):
        t = n * h
        estados.append(Y.copy())
        tiempos.append(t)
        Y = paso_rk2(Y, h, mu_T)
    return np.array(tiempos), np.array(estados)