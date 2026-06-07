import numpy as np
def f_luna(Y, mu_T):
    x, y, vx, vy = Y
    d_T = np.sqrt(x**2 + y**2)
    ax = -mu_T * x / d_T**3
    ay = -mu_T * y / d_T**3
    return np.array([vx, vy, ax, ay])