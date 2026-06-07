from orbitaLuna import f_luna
def paso_rk2(Y, h, mu_T):
    k1 = f_luna(Y, mu_T)
    Y_medio = Y + 0.5 * h * k1
    k2 = f_luna(Y_medio, mu_T)
    Y_siguiente = Y + h * k2
    return Y_siguiente