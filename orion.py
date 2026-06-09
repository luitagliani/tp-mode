import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MU_T = 398600.435507       # km^3/s^2
MU_L = 4902.800118         # km^3/s^2

RADIO_PERIGEO = 363300.0   # km
RADIO_APOGEO = 405500.0    # km
SEGUNDOS_POR_DIA = 24 * 3600

v0_luna = 1.076            # km/s

Y0_LUNA = np.array([
    RADIO_PERIGEO,
    0.0,
    0.0,
    v0_luna
])
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

def paso_rk2(Y, h, mu_T):
    k1 = f_luna(Y, mu_T)

    Y_medio = Y + 0.5 * h * k1

    k2 = f_luna(Y_medio, mu_T)

    Y_siguiente = Y + h * k2

    return Y_siguiente

def integrar_luna(Y0, h, tiempo_total, mu_T, metodo):
    cantidad_pasos = int(tiempo_total / h)

    estados = []
    tiempos = []

    Y = Y0.copy()

    for n in range(cantidad_pasos):
        t = n * h

        estados.append(Y.copy())
        tiempos.append(t)

        if metodo == "euler":
            Y = paso_euler(Y, h, mu_T)
        elif metodo == "rk2":
            Y = paso_rk2(Y, h, mu_T)
        else:
            raise ValueError("Metodo no reconocido")

    return np.array(tiempos), np.array(estados)
# ============================================================
# INTERPOLACION DE LA POSICION LUNAR
# ============================================================

def crear_interpolador_luna(t_luna, estados_luna):
    x_luna = estados_luna[:, 0]
    y_luna = estados_luna[:, 1]

    def posicion_luna(t):
        xL = np.interp(t, t_luna, x_luna)
        yL = np.interp(t, t_luna, y_luna)

        return xL, yL

    return posicion_luna
# ============================================================
# LECTURA DE TELEMETRIA DE ORION
# ============================================================

def leer_telemetria_orion(ruta_csv):
    columnas = ["datetime", "x", "y", "z", "vx", "vy", "vz"]

    df = pd.read_csv(
        ruta_csv,
        sep=";",
        header=None,
        decimal=",",
        names=columnas
    )

    df["datetime"] = pd.to_datetime(
        df["datetime"].str.replace(",", ".", regex=False),
        utc=True
    )

    return df


def obtener_condicion_inicial_orion(ruta_csv):
    df = leer_telemetria_orion(ruta_csv)

    inicio = pd.Timestamp("2026-04-03T04:00:00Z")
    fin = pd.Timestamp("2026-04-03T06:00:00Z")

    df_intervalo = df[
        (df["datetime"] >= inicio) &
        (df["datetime"] <= fin)
    ].copy()

    if len(df_intervalo) == 0:
        raise ValueError("No hay datos entre las 4 y las 6 UTC del 3 de abril")

    fila = df_intervalo.iloc[len(df_intervalo) // 2]

    Y0_orion = np.array([
        fila["x"],
        fila["y"],
        fila["vx"],
        fila["vy"]
    ], dtype=float)

    print("\nCondicion inicial de Orion tomada de telemetria:")
    print("Fecha UTC =", fila["datetime"])
    print("x =", fila["x"], "km")
    print("y =", fila["y"], "km")
    print("vx =", fila["vx"], "km/s")
    print("vy =", fila["vy"], "km/s")

    return Y0_orion, fila
# ============================================================
# AJUSTE DEL ANGULO INICIAL DE ORION
# ============================================================

def ajustar_angulo_velocidad(Y0_orion, delta_theta_deg, factor_velocidad):
    x, y, vx, vy = Y0_orion

    velocidad = np.sqrt(vx**2 + vy**2)
    velocidad = velocidad * factor_velocidad
    theta = np.arctan2(vy, vx)

    delta = np.deg2rad(delta_theta_deg)
    theta_nuevo = theta + delta

    vx_nuevo = velocidad * np.cos(theta_nuevo)
    vy_nuevo = velocidad * np.sin(theta_nuevo)

    return np.array([x, y, vx_nuevo, vy_nuevo])
# ============================================================
# EDO DE ORION
# ============================================================

def f_orion(Y, t, posicion_luna):
    xO, yO, vxO, vyO = Y

    xL, yL = posicion_luna(t)

    dT = np.sqrt(xO**2 + yO**2)
    dL = np.sqrt((xL - xO)**2 + (yL - yO)**2)

    axT = -MU_T * xO / dT**3
    ayT = -MU_T * yO / dT**3

    axL = MU_L * (xL - xO) / dL**3
    ayL = MU_L * (yL - yO) / dL**3

    ax = axT + axL
    ay = ayT + ayL

    return np.array([vxO, vyO, ax, ay])
# ============================================================
# METODOS NUMERICOS PARA ORION
# ============================================================

def paso_euler_orion(Y, t, h, posicion_luna):
    return Y + h * f_orion(Y, t, posicion_luna)


def paso_rk2_orion(Y, t, h, posicion_luna):
    k1 = f_orion(Y, t, posicion_luna)

    Y_medio = Y + 0.5 * h * k1
    t_medio = t + 0.5 * h

    k2 = f_orion(Y_medio, t_medio, posicion_luna)

    return Y + h * k2
# ============================================================
# INTEGRACION DE ORION
# ============================================================

def integrar_orion(
    Y0_orion,
    h,
    tiempo_total,
    posicion_luna,
    metodo,
    r_corte,
    t_minimo
):
    cantidad_pasos = int(tiempo_total / h)

    tiempos = []
    estados = []

    Y = Y0_orion.copy()

    for n in range(cantidad_pasos + 1):
        t = n * h

        tiempos.append(t)
        estados.append(Y.copy())

        dT = np.sqrt(Y[0]**2 + Y[1]**2)

        if dT <= r_corte and t > t_minimo:
            print("Regreso detectado con metodo:", metodo)
            print("t =", t / SEGUNDOS_POR_DIA, "dias")
            print("dT =", dT, "km")
            break

        if metodo == "euler":
            Y = paso_euler_orion(Y, t, h, posicion_luna)
        elif metodo == "rk2":
            Y = paso_rk2_orion(Y, t, h, posicion_luna)
        else:
            raise ValueError("Metodo no reconocido para Orion")

    return np.array(tiempos), np.array(estados)
# ============================================================
# METRICAS DE ORION
# ============================================================

def calcular_distancias_orion(t_orion, estados_orion, posicion_luna):
    xO = estados_orion[:, 0]
    yO = estados_orion[:, 1]
    vxO = estados_orion[:, 2]
    vyO = estados_orion[:, 3]

    dT = np.sqrt(xO**2 + yO**2)
    vO = np.sqrt(vxO**2 + vyO**2)

    dL = []

    for t, x, y in zip(t_orion, xO, yO):
        xL, yL = posicion_luna(t)
        distancia_luna = np.sqrt((xL - x)**2 + (yL - y)**2)
        dL.append(distancia_luna)

    dL = np.array(dL)

    metricas = {
        "distancia_min_tierra": np.min(dT),
        "distancia_max_tierra": np.max(dT),
        "distancia_min_luna": np.min(dL),
        "velocidad_min": np.min(vO),
        "velocidad_max": np.max(vO)
    }

    return metricas, dT, dL, vO
# ============================================================
# GRAFICO DE TRAYECTORIA DE ORION
# ============================================================

def graficar_trayectoria_orion(
    estados_luna,
    estados_orion_euler,
    estados_orion_rk2,
    h
):
    xL = estados_luna[:, 0] / 1000
    yL = estados_luna[:, 1] / 1000

    xE = estados_orion_euler[:, 0] / 1000
    yE = estados_orion_euler[:, 1] / 1000

    xR = estados_orion_rk2[:, 0] / 1000
    yR = estados_orion_rk2[:, 1] / 1000

    plt.figure(figsize=(8, 8))
    plt.plot(xL, yL, label="Orbita lunar")
    plt.plot(xE, yE, label="Orion Euler")
    plt.plot(xR, yR, label="Orion RK2")
    plt.scatter(0, 0, label="Tierra")

    plt.xlabel("x [miles de km]")
    plt.ylabel("y [miles de km]")
    plt.title(f"Trayectoria de Orion y orbita lunar - h = {h} s")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"orion_trayectoria_h_{h}.png", dpi=300)
    plt.close()

# ============================================================
# GRAFICO DE DISTANCIA ORION-TIERRA
# ============================================================

def graficar_distancia_tierra_orion(
    t_euler, dT_euler,
    t_rk2, dT_rk2,
    h
):
    dias_euler = t_euler / SEGUNDOS_POR_DIA
    dias_rk2 = t_rk2 / SEGUNDOS_POR_DIA

    plt.figure(figsize=(10, 5))
    plt.plot(dias_euler, dT_euler, label="Euler")
    plt.plot(dias_rk2, dT_rk2, label="RK2")
    plt.xlabel("Tiempo [dias]")
    plt.ylabel("Distancia Orion-Tierra [km]")
    plt.title(f"Distancia Orion-Tierra - h = {h} s")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"orion_distancia_tierra_h_{h}.png", dpi=300)
    plt.close()

# ============================================================
# GRAFICO DE DISTANCIA ORION-LUNA
# ============================================================

def graficar_distancia_luna_orion(
    t_euler, dL_euler,
    t_rk2, dL_rk2,
    h
):
    dias_euler = t_euler / SEGUNDOS_POR_DIA
    dias_rk2 = t_rk2 / SEGUNDOS_POR_DIA

    plt.figure(figsize=(10, 5))
    plt.plot(dias_euler, dL_euler, label="Euler")
    plt.plot(dias_rk2, dL_rk2, label="RK2")
    plt.xlabel("Tiempo [dias]")
    plt.ylabel("Distancia Orion-Luna [km]")
    plt.title(f"Distancia Orion-Luna - h = {h} s")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"orion_distancia_luna_h_{h}.png", dpi=300)
    plt.close()

# ============================================================
# EJECUCION DE ORION
# ============================================================

def simular_orion():
    ruta_csv = "Artemis-II-Data.csv"

    h = 60
    dias_simulados = 12
    tiempo_total = dias_simulados * SEGUNDOS_POR_DIA

    r_corte = 7000.0
    t_minimo = 3 * SEGUNDOS_POR_DIA

    delta_theta_deg = -78
    factor_velocidad = 1.16

    t_luna, estados_luna = integrar_luna(
        Y0_LUNA,
        h,
        tiempo_total,
        metodo="rk2",
        mu_T=MU_T
    )

    posicion_luna = crear_interpolador_luna(t_luna, estados_luna)

    Y0_orion, fila_orion = obtener_condicion_inicial_orion(ruta_csv)

    Y0_orion_ajustado = ajustar_angulo_velocidad(
        Y0_orion,
        delta_theta_deg,
        factor_velocidad=1.18
    )

    t_orion_euler, estados_orion_euler = integrar_orion(
        Y0_orion_ajustado,
        h,
        tiempo_total,
        posicion_luna,
        metodo="euler",
        r_corte=r_corte,
        t_minimo=t_minimo
    )

    t_orion_rk2, estados_orion_rk2 = integrar_orion(
        Y0_orion_ajustado,
        h,
        tiempo_total,
        posicion_luna,
        metodo="rk2",
        r_corte=r_corte,
        t_minimo=t_minimo
    )

    metricas_euler, dT_euler, dL_euler, v_euler = calcular_distancias_orion(
        t_orion_euler,
        estados_orion_euler,
        posicion_luna
    )

    metricas_rk2, dT_rk2, dL_rk2, v_rk2 = calcular_distancias_orion(
        t_orion_rk2,
        estados_orion_rk2,
        posicion_luna
    )

    print("\nMetricas Orion - Euler")
    for clave, valor in metricas_euler.items():
        print(clave, "=", valor)

    print("\nMetricas Orion - RK2")
    for clave, valor in metricas_rk2.items():
        print(clave, "=", valor)

    graficar_trayectoria_orion(
        estados_luna,
        estados_orion_euler,
        estados_orion_rk2,
        h
    )

    graficar_distancia_tierra_orion(
        t_orion_euler,
        dT_euler,
        t_orion_rk2,
        dT_rk2,
        h
    )

    graficar_distancia_luna_orion(
        t_orion_euler,
        dL_euler,
        t_orion_rk2,
        dL_rk2,
        h
    )

def probar_deltas_y_velocidades_orion():
    ruta_csv = "Artemis-II-Data.csv"

    h = 60
    dias_simulados = 12
    tiempo_total = dias_simulados * SEGUNDOS_POR_DIA

    r_corte = 7000.0
    t_minimo = 3 * SEGUNDOS_POR_DIA

    deltas = [-85, -82, -80, -78, -76, -75, -74, -72, -70, -68, -65]
    factores = [1.06, 1.08, 1.10, 1.12, 1.1, 1.16]

    t_luna, estados_luna = integrar_luna(
        Y0_LUNA,
        h,
        tiempo_total,
        metodo="rk2",
        mu_T=MU_T
    )

    posicion_luna = crear_interpolador_luna(t_luna, estados_luna)

    Y0_orion, fila_orion = obtener_condicion_inicial_orion(ruta_csv)

    print("\n" + "="*50)
    print("Probando diferentes deltas de angulo inicial para Orion")
    print("delta | factor | max Tierra | min Tierra | min Luna | max Velocidad | min Velocidad")
    print("-"*100)

    resultados = []

    for delta in deltas:
        for factor_velocidad in factores:
            print(f"\nSimulando con delta_theta_deg = {delta} grados")
            Y0_orion_ajustado = ajustar_angulo_velocidad(
                Y0_orion,
                delta, factor_velocidad
            )

            t_orion_rk2, estados_orion_rk2 = integrar_orion(
                Y0_orion_ajustado,
                h,
                tiempo_total,
                posicion_luna,
                metodo="rk2",
                r_corte=r_corte,
                t_minimo=t_minimo
            )

            metricas_rk2, dT_rk2, dL_rk2, v_rk2 = calcular_distancias_orion(
                t_orion_rk2,
                estados_orion_rk2,
                posicion_luna
            )

            distancia_max_tierra = metricas_rk2["distancia_max_tierra"]
            distancia_min_tierra = metricas_rk2["distancia_min_tierra"]
            distancia_min_luna = metricas_rk2["distancia_min_luna"]
            velocidad_max = metricas_rk2["velocidad_max"]

            print(
                f"{delta:5} | {factor_velocidad:6.2f} | "
                f"{distancia_max_tierra:12.1f} | {distancia_min_tierra:12.1f} | {distancia_min_luna:10.1f} | "
                f"{velocidad_max:14.3f}"
            )

            resultados.append({
                "delta": delta,
                "factor_velocidad": factor_velocidad,
                "distancia_max_tierra": distancia_max_tierra,
                "distancia_min_tierra": distancia_min_tierra,
                "distancia_min_luna": distancia_min_luna,
                "velocidad_max": velocidad_max
            })
    mejor = min(resultados, key=lambda r: r["distancia_min_luna"])
    print("\nMejor resultado (min Luna mas grande):")
    print(f"delta = {mejor['delta']} grados, factor_velocidad = {mejor['factor_velocidad']}, "
          f"distancia_min_luna = {mejor['distancia_min_luna']} km")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    simular_orion()
    # probar_deltas_y_velocidades_orion()