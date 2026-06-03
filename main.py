import numpy as np
import math
import matplotlib.pyplot as plt

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ==
# 1) Calcular la órbita Lunar. Validar perigeo, apogeo, velocidad máxima y mínima.
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ==

GM_T = 398600.4418   # Constante de gravitacion universal (G) * masa de la tierra (MT) => (km^3/s^2)
RADIO_PERIGEO = 362600.0 # km
RADIO_APOGEO  = 405400.0 # km
SEGUNDOS_POR_DIA = 24 * 3600
A_LUNA = (RADIO_PERIGEO + RADIO_APOGEO) / 2 #semieje mayor de órbita de la Luna
V_PER_LUNA = math.sqrt(GM_T * (2/RADIO_PERIGEO - 1/A_LUNA)) # velocidad maxima de la Luna en km/s
V_APO_LUNA = math.sqrt(GM_T * (2/RADIO_APOGEO - 1/A_LUNA))  # velocidad mínima de la Luna en km/s

"""
Recibe estado = [x, y, vx, vy], donde x e y representan la posicion de la Luna
y vx y vy representan su velocidad en los ejes correspondientes.

Devuelve las derivadas = [vx, vy, ax, ay] (por propiedades matematicas aclaradas en el enunciado)
"""
def derivadas_luna(estado):
    x, y, vx, vy = estado
    r = math.sqrt(x**2 + y**2)

    a = GM_T / r**2
    ax = -a * (x / r)
    ay = -a * (y / r)

    return np.array([vx, vy, ax, ay])

"""
def paso_euler(estado, dt):
    d = derivadas_luna(estado)
    return estado + dt * d
"""

def paso_rk2(estado, dt):
    k1 = derivadas_luna(estado)
    estado_supuesto = estado + dt * k1
    k2 = derivadas_luna(estado_supuesto )

    return estado + dt * (k1 + k2) / 2

# Recibe las listas de coordenadas y tiempos simulados, y grafica los resultados de RK2.
def graficar_resultados_luna(x_rk2, y_rk2, dist_rk2, tiempos_dias, r_apogeo, r_perigeo):

    x_rk2_plot = np.array(x_rk2) / 1e3
    y_rk2_plot = np.array(y_rk2) / 1e3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.set_title("Órbita Lunar: RK2")
    ax1.plot(x_rk2_plot, y_rk2_plot, label='RK2', color='blue')
    ax1.plot(0, 0, 'go', markersize=10, label='Tierra')
    ax1.set_xlabel("Distancia X (x1000 km)")
    ax1.set_ylabel("Distancia Y (x1000 km)")
    ax1.axis('equal')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Distancia Lunar en el tiempo")
    ax2.plot(tiempos_dias, dist_rk2, label='RK2', color='blue')
    ax2.axhline(r_apogeo, color='red', linestyle=':', label='Apogeo Real')
    ax2.axhline(r_perigeo, color='green', linestyle=':', label='Perigeo Real')
    ax2.set_xlabel("Tiempo (días)")
    ax2.set_ylabel("Distancia a la Tierra (km)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    estado_inicial = np.array([RADIO_PERIGEO, 0.0, 0.0, V_PER_LUNA])

    dt = 500 # Paso de integración de 8 minutos aprox
    tiempo_total = 28 * SEGUNDOS_POR_DIA # 28 días en segundos (tiempo en que la luna da una vuelta completa alrededor de la tierra)

    estado_rk2 = np.copy(estado_inicial)

    tiempos_dias = []
    x_rk2, y_rk2, dist_rk2 = [], [], []
    vel_rk2 = []

    for t in range(0, int(tiempo_total), int(dt)):
        x_rk2.append(estado_rk2[0])
        y_rk2.append(estado_rk2[1])
        dist_rk2.append(math.hypot(estado_rk2[0], estado_rk2[1]))
        vel_rk2.append(math.hypot(estado_rk2[2], estado_rk2[3]))

        tiempos_dias.append(float(t/SEGUNDOS_POR_DIA))

        estado_rk2 = paso_rk2(estado_rk2, dt)

    ancho = 50

    print("⊹₊˚‧︵‿₊୨✧୧₊‿︵‧˚₊⊹".center(ancho))
    print("Resultados órbita lunar en un mes".center(ancho))
    print("PUNTO 1".center(ancho))
    print("- - - - - - - - - -".center(ancho))

    print(f"➤ Apogeo real esperado: {RADIO_APOGEO} km")
    print(f"➤ Apogeo calculado con RK2:   {max(dist_rk2):.2f} km\n")

    print(f"➤ Perigeo real esperado: {RADIO_PERIGEO} km")
    print(f"➤ Perigeo calculado con RK2:   {min(dist_rk2):.2f} km\n")

    print(f"➤ Velocidad Máxima esperada: {V_PER_LUNA:.4f} km/s")
    print(f"➤ Velocidad Máxima RK2:      {max(vel_rk2):.4f} km/s\n")

    print(f"➤ Velocidad Mínima esperada: {V_APO_LUNA:.4f} km/s")
    print(f"➤ Velocidad Mínima RK2:      {min(vel_rk2):.4f} km/s")

    graficar_resultados_luna(x_rk2, y_rk2, dist_rk2, tiempos_dias, RADIO_APOGEO, RADIO_PERIGEO)


if __name__ == "__main__":
    main()