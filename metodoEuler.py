import numpy as np
def f_luna(Y, mu_T):
    x, y, vx, vy = Y
    d_T = np.sqrt(x**2 + y**2)
    ax = -mu_T * x / d_T**3
    ay = -mu_T * y / d_T**3
    return np.array([vx, vy, ax, ay])

def paso_euler(Y, h, mu_T):
    dY = f_luna(Y, mu_T)
    Y_siguiente = Y + h * dY
    return Y_siguiente