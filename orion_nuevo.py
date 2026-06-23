import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MU_T = 398600.435507       # km^3/s^2
MU_L = 4902.800118         # km^3/s^2

RADIO_PERIGEO = 363300.0   # km
RADIO_APOGEO = 405500.0    # km
SEGUNDOS_POR_DIA = 24 * 3600

v0_luna = 1.076            # km/s

# Y0_LUNA = np.array([
#     RADIO_PERIGEO,
#     0.0,
#     0.0,
#     v0_luna
# ])

def obtener_condicion_inicial_luna(angulo_grados):
    """
    Calcula la posición y velocidad inicial de la Luna dados un ángulo de fase inicial.
    """
    theta = np.deg2rad(angulo_grados)
    
    # Posición usando trigonometría básica (x = r*cos, y = r*sin)
    x = RADIO_PERIGEO * np.cos(theta)
    y = RADIO_PERIGEO * np.sin(theta)
    
    # La velocidad siempre es perpendicular al radio en una órbita circular aproximada
    vx = -v0_luna * np.sin(theta)
    vy =  v0_luna * np.cos(theta)
    
    return np.array([x, y, vx, vy])

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
# GRAFICO DE TRAYECTORIA DE ORION (SISTEMA INERCIAL)
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
    plt.plot(xL, yL, color='gray', linestyle='--', label="Orbita lunar")
    plt.plot(xE, yE, label="Orion Euler", alpha=0.7)
    plt.plot(xR, yR, color='blue', label="Orion RK2")
    plt.scatter(0, 0, color='green', s=100, label="Tierra")

    plt.xlabel("x [miles de km]")
    plt.ylabel("y [miles de km]")
    plt.title(f"Trayectoria de Orion y orbita lunar (Inercial) - h = {h} s")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"orion_trayectoria_inercial_h_{h}.png", dpi=300)
    plt.close()

# ============================================================
# GRAFICO DE TRAYECTORIA DE ORION (SISTEMA SINÓDICO/ROTANTE)
# ============================================================

def graficar_trayectoria_orion_sistema_rotante(
    t_orion_rk2,
    estados_orion_rk2,
    posicion_luna,
    h
):
    xO = estados_orion_rk2[:, 0]
    yO = estados_orion_rk2[:, 1]

    x_rot = []
    y_rot = []
    r_luna_arr = []

    for t, x, y in zip(t_orion_rk2, xO, yO):
        xL, yL = posicion_luna(t)

        # Calculamos el ángulo de la luna en este instante t
        theta = np.arctan2(yL, xL)

        # Aplicamos matriz de rotación para anclar la línea Tierra-Luna al eje X
        x_orion_rot = x * np.cos(theta) + y * np.sin(theta)
        y_orion_rot = -x * np.sin(theta) + y * np.cos(theta)

        x_rot.append(x_orion_rot / 1000)
        y_rot.append(y_orion_rot / 1000)

        # Guardamos la distancia exacta a la luna
        r_luna_arr.append(np.sqrt(xL**2 + yL**2) / 1000)

    x_rot = np.array(x_rot)
    y_rot = np.array(y_rot)
    r_luna_promedio = np.mean(r_luna_arr)

    # Buscar índice de máximo acercamiento a la luna
    dL = np.sqrt((x_rot - r_luna_promedio)**2 + y_rot**2)
    indice_min_luna = np.argmin(dL)

    plt.figure(figsize=(10, 8))

    plt.plot(x_rot, y_rot, color='blue', label="Orion RK2 (Forma de 8)")

    plt.scatter(0, 0, color='green', s=100, label="Tierra")
    
    # En este sistema, la Luna está estática sobre el eje X
    plt.scatter(r_luna_promedio, 0, color='gray', s=80, label="Luna")

    plt.scatter(x_rot[0], y_rot[0], marker="o", color='cyan', label="Inicio Orion")
    plt.scatter(x_rot[-1], y_rot[-1], marker="x", color='red', label="Fin Orion")
    plt.scatter(
        x_rot[indice_min_luna],
        y_rot[indice_min_luna],
        marker="*",
        color='gold',
        s=150,
        label="Máx. acercamiento lunar"
    )

    plt.xlabel("X Sinódico [miles de km]")
    plt.ylabel("Y Sinódico [miles de km]")
    plt.title(f"Trayectoria de Orion en Sistema Sinódico Tierra-Luna - h = {h} s")
    plt.axis("equal")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"orion_trayectoria_rotante_h_{h}.png", dpi=300)
    plt.close()

# ============================================================
# OTROS GRÁFICOS (DISTANCIAS)
# ============================================================

def graficar_distancia_tierra_orion(t_euler, dT_euler, t_rk2, dT_rk2, h):
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

def graficar_distancia_luna_orion(t_euler, dL_euler, t_rk2, dL_rk2, h):
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
    # Asegúrate de que el nombre coincida exactamente con tu archivo
    ruta_csv = "Artemis-II-Data.csv"

    h = 60
    dias_simulados = 18
    tiempo_total = dias_simulados * SEGUNDOS_POR_DIA

    r_corte = 7000.0
    t_minimo = 3 * SEGUNDOS_POR_DIA

    # Parámetros que controlan la "puntería" hacia la Luna.
    # Si notas que el 8 no se cierra bien, acá es donde tenés que ajustar.
    delta_theta_deg = 2
    factor_velocidad = 1.162
    angulo_luna_inicial = -150

    # t_luna, estados_luna = integrar_luna(
    #     Y0_LUNA,
    #     h,
    #     tiempo_total,
    #     metodo="rk2",
    #     mu_T=MU_T
    # )


    Y0_LUNA_ROTADA = obtener_condicion_inicial_luna(angulo_luna_inicial)

    # El resto sigue igual, pero usando Y0_LUNA_ROTADA
    t_luna, estados_luna = integrar_luna(
        Y0_LUNA_ROTADA,
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
        factor_velocidad
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
        print(clave, "=", round(valor, 2))

    print("\nMetricas Orion - RK2")
    for clave, valor in metricas_rk2.items():
        print(clave, "=", round(valor, 2))

    graficar_trayectoria_orion(
        estados_luna,
        estados_orion_euler,
        estados_orion_rk2,
        h
    )

    graficar_distancia_tierra_orion(
        t_orion_euler, dT_euler,
        t_orion_rk2, dT_rk2, h
    )

    graficar_distancia_luna_orion(
        t_orion_euler, dL_euler,
        t_orion_rk2, dL_rk2, h
    )

    # Este es el gráfico fundamental que mostrará el 8
    graficar_trayectoria_orion_sistema_rotante(
        t_orion_rk2,
        estados_orion_rk2,
        posicion_luna,
        h
    )

def probar_deltas_y_velocidades_orion():
    ruta_csv = "Artemis-II-Data.csv"

    h = 60
    dias_simulados = 18
    tiempo_total = dias_simulados * SEGUNDOS_POR_DIA

    r_corte = 7000.0
    t_minimo = 3 * SEGUNDOS_POR_DIA

    deltas = np.arange(-73, -71, 0.1)
    factores = np.arange(1.15, 1.17, 0.002)

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
                f"{delta:5} | {factor_velocidad:6.3f} | "
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

def probar_angulos_luna_y_orion():
    ruta_csv = "Artemis-II-Data.csv"

    h = 60
    dias_simulados = 18
    tiempo_total = dias_simulados * SEGUNDOS_POR_DIA

    r_corte = 7000.0
    t_minimo = 3 * SEGUNDOS_POR_DIA

    # 1. Definimos el barrido (Arrays de numpy)
    angulos_luna = np.arange(-190, -140, 5)  # Probamos posiciones de la Luna de -130 a -90 grados
    deltas_orion = np.arange(-5, 6, 1)      # Probamos ajustar Orion entre -5 y +5 grados
    factor_velocidad = [1.16, 1.162]                # Dejamos la velocidad fija por ahora

    Y0_orion, fila_orion = obtener_condicion_inicial_orion(ruta_csv)

    print("\n" + "="*70)
    print("Buscando el 8 perfecto: Barrido de Angulo Lunar y Delta Orion")
    print("Ang Luna | Delta Orion | min Luna | min Tierra | max Tierra")
    print("-" * 70)

    resultados = []

    for factor in factor_velocidad:
        for ang_luna in angulos_luna:
            # A) Calculamos la nueva orbita de la Luna para este angulo
            Y0_LUNA_ROTADA = obtener_condicion_inicial_luna(ang_luna)
            t_luna, estados_luna = integrar_luna(
                Y0_LUNA_ROTADA, h, tiempo_total, metodo="rk2", mu_T=MU_T
            )
            posicion_luna = crear_interpolador_luna(t_luna, estados_luna)

            for delta in deltas_orion:
                # B) Ajustamos levemente la punteria de Orion
                Y0_orion_ajustado = ajustar_angulo_velocidad(
                    Y0_orion, delta, factor
                )

                # C) Simulamos el viaje
                t_orion_rk2, estados_orion_rk2 = integrar_orion(
                    Y0_orion_ajustado, h, tiempo_total, posicion_luna,
                    metodo="rk2", r_corte=r_corte, t_minimo=t_minimo
                )

                # D) Calculamos las metricas
                metricas_rk2, dT_rk2, dL_rk2, v_rk2 = calcular_distancias_orion(
                    t_orion_rk2, estados_orion_rk2, posicion_luna
                )

                d_min_luna = metricas_rk2["distancia_min_luna"]
                d_min_tierra = metricas_rk2["distancia_min_tierra"]
                d_max_tierra = metricas_rk2["distancia_max_tierra"]

                print(f"{ang_luna:8.1f} | {delta:11.1f} | {d_min_luna:8.1f} | {d_min_tierra:10.1f} | {d_max_tierra:10.1f}")

                resultados.append({
                    "ang_luna": ang_luna,
                    "delta": delta,
                    "min_luna": d_min_luna,
                    "min_tierra": d_min_tierra
                })

    # Filtrar los que no chocan con la Luna (mayor a 2000 km)
    validos = [r for r in resultados if r["min_luna"] > 2000]
    if validos:
        mejor = min(validos, key=lambda r: r["min_tierra"])
        print("\n¡Mejor combinacion encontrada!")
        print(f"Angulo Luna = {mejor['ang_luna']}°, Delta Orion = {mejor['delta']}°")
        print(f"Min Luna = {mejor['min_luna']:.1f} km, Min Tierra = {mejor['min_tierra']:.1f} km")
    else:
        print("\nNo se encontraron trayectorias que sobrevivan a la Luna en este barrido.")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    simular_orion()
    # probar_deltas_y_velocidades_orion()
    # probar_angulos_luna_y_orion()