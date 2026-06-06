import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

GM_T = 398600.4418   # Constante de gravitacion universal (G) * masa de la tierra (MT) => (km^3/s^2)
GM_L = 4902.8000 # Constante de gravitacion (G) * masa de la Luna (ML) => (km^3/s^2)
RADIO_PERIGEO = 362600.0 # km
RADIO_APOGEO  = 405400.0 # km
SEGUNDOS_POR_DIA = 24 * 3600
A_LUNA = (RADIO_PERIGEO + RADIO_APOGEO) / 2 #semieje mayor de órbita de la Luna
V_PER_LUNA = math.sqrt(GM_T * (2/RADIO_PERIGEO - 1/A_LUNA)) # velocidad maxima de la Luna en km/s
V_APO_LUNA = math.sqrt(GM_T * (2/RADIO_APOGEO - 1/A_LUNA))  # velocidad mínima de la Luna en km/s
ANCHO = 50

#  Funciones de Punto 1

def derivadas_luna(estado):
    x, y, vx, vy = estado
    r = math.sqrt(x**2 + y**2)
    a = GM_T / r**2
    return np.array([vx, vy, -a*(x/r), -a*(y/r)])

def paso_rk2(estado, dt):
    k1 = derivadas_luna(estado)
    k2 = derivadas_luna(estado + dt * k1)
    return estado + dt * (k1 + k2) / 2

def graficar_resultados_luna(x_rk2, y_rk2, dist_rk2, tiempos_dias, r_apogeo, r_perigeo):
    x_plot = np.array(x_rk2) / 1e3
    y_plot = np.array(y_rk2) / 1e3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.set_title("Órbita lunar: RK2")
    ax1.plot(x_plot, y_plot, label='RK2', color='blue')
    ax1.plot(0, 0, 'go', markersize=10, label='Tierra')
    ax1.set_xlabel("Distancia X (x1000 km)")
    ax1.set_ylabel("Distancia Y (x1000 km)")
    ax1.axis('equal'); ax1.legend(); ax1.grid(True)
    ax2.set_title("Distancia lunar en el tiempo")
    ax2.plot(tiempos_dias, dist_rk2, label='RK2', color='blue')
    ax2.axhline(r_apogeo,  color='red',   linestyle=':', label='Apogeo real')
    ax2.axhline(r_perigeo, color='green', linestyle=':', label='Perigeo real')
    ax2.set_xlabel("Tiempo (días)"); ax2.set_ylabel("Distancia a la Tierra (km)")
    ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.show()


#  Funcion de punto 2

def extraer_condiciones_iniciales(ruta_archivo, inicio_str, fin_str):
    df = pd.read_csv(ruta_archivo, sep=';', header=None, decimal=',')
    df.columns = ['Timestamp', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ']
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%dT%H:%M:%S,%f')

    filtro = (df['Timestamp'] >= inicio_str) & (df['Timestamp'] <= fin_str)
    df_filtrado = df.loc[filtro]

    if df_filtrado.empty:
        raise ValueError("No se encontraron datos en esa franja horaria.")

    primer_fila = df_filtrado.iloc[0]
    r_vec = np.array([primer_fila['X'], primer_fila['Y'], primer_fila['Z']])
    v_vec = np.array([primer_fila['VX'], primer_fila['VY'], primer_fila['VZ']])

    d_T = np.linalg.norm(r_vec)
    v_0 = np.linalg.norm(v_vec)

    v_rad = np.dot(r_vec, v_vec) / d_T
    v_tan = math.sqrt(v_0**2 - v_rad**2)

    print()
    print("⊹₊˚‧︵‿₊୨✧୧₊‿︵‧˚₊⊹".center(ANCHO))
    print(f"Condiciones iniciales en {primer_fila['Timestamp']}".center(ANCHO))
    print("PUNTO 2".center(ANCHO))
    print("- - - - - - - - - -".center(ANCHO))
    print(f"➤ Distancia a la Tierra: {d_T:.2f} km")
    print(f"➤ Velocidad total real: {v_0:.3f} km/s")

    return d_T, v_rad, v_tan

# Funciones de punto 3

def derivadas_orion(estado_orion, estado_luna):
    x, y, vx, vy = estado_orion
    xl, yl, _, _  = estado_luna
    r_T  = math.hypot(x, y)
    ax_T = -GM_T/r_T**2 * (x/r_T)
    ay_T = -GM_T/r_T**2 * (y/r_T)
    dx_L = x - xl;  dy_L = y - yl
    r_L  = math.hypot(dx_L, dy_L)
    ax_L = -GM_L/r_L**2 * (dx_L/r_L)
    ay_L = -GM_L/r_L**2 * (dy_L/r_L)
    return np.array([vx, vy, ax_T+ax_L, ay_T+ay_L])

def paso_rk2_orion(estado_orion, estado_luna, estado_luna_sig, dt):
    k1 = derivadas_orion(estado_orion, estado_luna)
    k2 = derivadas_orion(estado_orion + dt*k1, estado_luna_sig)
    return estado_orion + dt * (k1 + k2) / 2

def graficar_mision(x_luna, y_luna, x_orion, y_orion, idx_min_luna=None):
    x_l = np.array(x_luna) / 1e3
    y_l = np.array(y_luna) / 1e3
    x_o = np.array(x_orion) / 1e3
    y_o = np.array(y_orion) / 1e3

    plt.figure(figsize=(10, 8))
    plt.title("Trayectoria Artemis II (Tierra – Luna – Orion)")

    plt.plot(x_l, y_l, '--', color='gray',   label='Órbita lunar')
    plt.plot(x_o, y_o,       color='orange', label='Trayectoria Orion')

    plt.plot(0, 0, 'go', markersize=12, label='Tierra')
    plt.plot(x_l[0],  y_l[0],  'b^', markersize=8,  label='Luna (inicio)')
    plt.plot(x_l[-1], y_l[-1], 'ko', markersize=6,  label='Luna (final)')


    plt.xlabel("Distancia X (x1000 km)")
    plt.ylabel("Distancia Y (x1000 km)")
    plt.axis('equal'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()

# Funciones de punto 4

def paso_euler(estado, dt):
    d = derivadas_luna(estado)
    return estado + dt * d

def paso_euler_orion(estado_orion, estado_luna, dt):
    k1 = derivadas_orion(estado_orion, estado_luna)
    return estado_orion + dt * k1


def graficar_comparacion(x_luna, y_luna, x_orion_rk2, y_orion_rk2, x_orion_euler, y_orion_euler):
    x_l = np.array(x_luna) / 1e3
    y_l = np.array(y_luna) / 1e3
    x_o_rk2 = np.array(x_orion_rk2) / 1e3
    y_o_rk2 = np.array(y_orion_rk2) / 1e3
    x_o_eu = np.array(x_orion_euler) / 1e3
    y_o_eu = np.array(y_orion_euler) / 1e3

    plt.figure(figsize=(10, 8))
    plt.title("Comparación de métodos: RK2 vs Euler")

    plt.plot(x_l, y_l, '--', color='gray', label='Órbita Lunar')

    plt.plot(x_o_rk2, y_o_rk2, color='orange', label='Orion (RK2)', linewidth=2)
    plt.plot(x_o_eu, y_o_eu, color='red', linestyle='-.', label='Orion (Euler)', linewidth=2)

    plt.plot(0, 0, 'go', markersize=12, label='Tierra')
    plt.plot(x_l[-1], y_l[-1], 'ko', markersize=6, label='Luna (posición final)')

    plt.xlabel("Distancia X (x1000 km)")
    plt.ylabel("Distancia Y (x1000 km)")
    plt.axis('equal'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()

#Funcion de punto 5
def analizar_largo_plazo():
    print("\n" + "⊹₊˚‧︵‿₊୨✧୧₊‿︵‧˚₊⊹".center(ANCHO))
    print("Análisis a largo plazo (Tierra-Luna)".center(ANCHO))
    print("PUNTO 5".center(ANCHO))
    print("- - - - - - - - - -".center(ANCHO))

    estado_rk2 = np.array([RADIO_PERIGEO, 0.0, 0.0, V_PER_LUNA], dtype=float)
    estado_euler = np.copy(estado_rk2)

    dt = 3600
    meses_a_simular = 6
    tiempo_total = meses_a_simular * 28 * SEGUNDOS_POR_DIA

    x_rk2_hist, y_rk2_hist = [], []
    x_eu_hist, y_eu_hist = [], []

    print(f"➤ Simulando {meses_a_simular} meses con dt = {dt}s...")

    for t in range(0, int(tiempo_total), int(dt)):
        x_rk2_hist.append(estado_rk2[0])
        y_rk2_hist.append(estado_rk2[1])
        x_eu_hist.append(estado_euler[0])
        y_eu_hist.append(estado_euler[1])

        estado_rk2 = paso_rk2(estado_rk2, dt)
        estado_euler = paso_euler(estado_euler, dt)

    x_r_plot = np.array(x_rk2_hist) / 1e3
    y_r_plot = np.array(y_rk2_hist) / 1e3
    x_e_plot = np.array(x_eu_hist) / 1e3
    y_e_plot = np.array(y_eu_hist) / 1e3

    plt.figure(figsize=(9, 9))
    plt.title(f"Comportamiento de órbita a largo plazo ({meses_a_simular} meses)")

    plt.plot(x_r_plot, y_r_plot, color='blue', label='Órbita RK2')

    plt.plot(x_e_plot, y_e_plot, color='red', alpha=0.7, linewidth=1, label='Órbita Euler')

    plt.plot(0, 0, 'go', markersize=12, label='Tierra')

    plt.xlabel("Distancia X (x1000 km)")
    plt.ylabel("Distancia Y (x1000 km)")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # PUNTO 1
    estado_inicial = np.array([RADIO_PERIGEO, 0.0, 0.0, V_PER_LUNA])
    dt = 500
    tiempo_total = 28 * SEGUNDOS_POR_DIA
    estado_rk2 = np.copy(estado_inicial)
    tiempos_dias, x_rk2, y_rk2, dist_rk2, vel_rk2 = [], [], [], [], []

    for t in range(0, int(tiempo_total), int(dt)):
        x_rk2.append(estado_rk2[0]);  y_rk2.append(estado_rk2[1])
        dist_rk2.append(math.hypot(estado_rk2[0], estado_rk2[1]))
        vel_rk2.append(math.hypot(estado_rk2[2], estado_rk2[3]))
        tiempos_dias.append(t / SEGUNDOS_POR_DIA)
        estado_rk2 = paso_rk2(estado_rk2, dt)

    print("⊹₊˚‧︵‿₊୨✧୧₊‿︵‧˚₊⊹".center(ANCHO))
    print("Resultados órbita lunar en un mes".center(ANCHO))
    print("PUNTO 1".center(ANCHO))
    print("- - - - - - - - - -".center(ANCHO))
    print(f"➤ Apogeo real esperado: {RADIO_APOGEO} km")
    print(f"➤ Apogeo calculado RK2: {max(dist_rk2):.2f} km\n")
    print(f"➤ Perigeo real esperado: {RADIO_PERIGEO} km")
    print(f"➤ Perigeo calculado RK2: {min(dist_rk2):.2f} km\n")
    print(f"➤ Velocidad máxima esperada: {V_PER_LUNA:.4f} km/s")
    print(f"➤ Velocidad máxima RK2: {max(vel_rk2):.4f} km/s\n")
    print(f"➤ Velocidad mínima esperada: {V_APO_LUNA:.4f} km/s")
    print(f"➤ Velocidad mínima RK2: {min(vel_rk2):.4f} km/s")
    graficar_resultados_luna(x_rk2, y_rk2, dist_rk2, tiempos_dias, RADIO_APOGEO, RADIO_PERIGEO)

    # PUNTO 2
    archivo_datos = 'Artemis-II-Data.csv'
    d_T_orion_0, v_rad_0, v_tan_0 = extraer_condiciones_iniciales(
        archivo_datos,
        '2026-04-03 04:00:00',
        '2026-04-03 06:00:00'
    )

    # PUNTO 3
    print()
    print("⊹₊˚‧︵‿₊୨✧୧₊‿︵‧˚₊⊹".center(ANCHO))
    print("Simulación de Orion".center(ANCHO))
    print("PUNTO 3".center(ANCHO))
    print("- - - - - - - - - -".center(ANCHO))

    angulo_optimo = 15

    alpha_rad = math.radians(angulo_optimo)
    vx_rotado = v_rad_0 * math.cos(alpha_rad) - v_tan_0 * math.sin(alpha_rad)
    vy_rotado = v_rad_0 * math.sin(alpha_rad) + v_tan_0 * math.cos(alpha_rad)

    estado_orion = np.array([
        d_T_orion_0 * math.cos(alpha_rad),
        d_T_orion_0 * math.sin(alpha_rad),
        vx_rotado,
        vy_rotado
    ])

    estado_luna_mision = np.array([RADIO_PERIGEO, 0.0, 0.0, V_PER_LUNA])

    dt_mision = 60
    tiempo_maximo_mision = 20 * SEGUNDOS_POR_DIA
    x_orion_hist, y_orion_hist = [], []
    x_luna_hist,  y_luna_hist = [], []
    dist_min_luna  = float('inf')
    idx_min_luna   = 0

    for t in range(0, int(tiempo_maximo_mision), dt_mision):
        x_orion_hist.append(estado_orion[0]); y_orion_hist.append(estado_orion[1])
        x_luna_hist.append(estado_luna_mision[0]);  y_luna_hist.append(estado_luna_mision[1])

        el_sig = paso_rk2(estado_luna_mision, dt_mision)
        eo_sig = paso_rk2_orion(estado_orion, estado_luna_mision, el_sig, dt_mision)

        d_l = math.hypot(estado_orion[0] - estado_luna_mision[0], estado_orion[1] - estado_luna_mision[1])
        if d_l < dist_min_luna:
            dist_min_luna = d_l
            idx_min_luna  = len(x_orion_hist) - 1

        estado_luna_mision = el_sig
        estado_orion= eo_sig

        d_t = math.hypot(estado_orion[0], estado_orion[1])
        if d_t < 6500 and t > SEGUNDOS_POR_DIA:
            print(f"➤ ¡Reingreso atmosférico exitoso a los {t/SEGUNDOS_POR_DIA:.2f} días!")
            break

    print(f"➤ Distancia mínima a la Luna: {dist_min_luna:.0f} km")

    graficar_mision(x_luna_hist, y_luna_hist, x_orion_hist, y_orion_hist, idx_min_luna)


    # PUNTO 4
    print()
    print("⊹₊˚‧︵‿₊୨✧୧₊‿︵‧˚₊⊹".center(ANCHO))
    print("Comparación RK2 vs Euler".center(ANCHO))
    print("PUNTO 4".center(ANCHO))
    print("- - - - - - - - - -".center(ANCHO))

    estado_orion_euler = np.array([
        d_T_orion_0 * math.cos(alpha_rad),
        d_T_orion_0 * math.sin(alpha_rad),
        vx_rotado,
        vy_rotado
    ])
    estado_luna_euler = np.array([RADIO_PERIGEO, 0.0, 0.0, V_PER_LUNA])

    x_orion_eu_hist, y_orion_eu_hist = [], []
    dist_min_luna_eu = float('inf')

    for t in range(0, int(tiempo_maximo_mision), int(dt_mision)):
        x_orion_eu_hist.append(estado_orion_euler[0])
        y_orion_eu_hist.append(estado_orion_euler[1])

        el_sig_eu = paso_euler(estado_luna_euler, dt_mision)
        eo_sig_eu = paso_euler_orion(estado_orion_euler, estado_luna_euler, dt_mision)

        dxL = estado_orion_euler[0] - estado_luna_euler[0]
        dyL = estado_orion_euler[1] - estado_luna_euler[1]
        dist_luna = math.hypot(dxL, dyL)
        if dist_luna < dist_min_luna_eu:
            dist_min_luna_eu = dist_luna

        estado_luna_euler = el_sig_eu
        estado_orion_euler = eo_sig_eu

        dist_tierra = math.hypot(estado_orion_euler[0], estado_orion_euler[1])
        if dist_tierra < 6500 and t > SEGUNDOS_POR_DIA:
            print(f"➤ Euler: Reingreso atmosférico a los {t/SEGUNDOS_POR_DIA:.2f} días.")
            break

    print(f"➤ Distancia mínima a la Luna (RK2): {dist_min_luna:.0f} km")
    print(f"➤ Distancia mínima a la Luna (Euler): {dist_min_luna_eu:.0f} km")

    graficar_comparacion(x_luna_hist, y_luna_hist,x_orion_hist, y_orion_hist,x_orion_eu_hist, y_orion_eu_hist)

    #PUNTO 5
    analizar_largo_plazo()

if __name__ == "__main__":
    main()