import numpy as np
import matplotlib.pyplot as plt

# Constantes y datos de referencia
MU_T = 398600.435507       # km^3/s^2
RADIO_PERIGEO = 363300.0   # km
RADIO_APOGEO = 405500.0    # km
SEGUNDOS_POR_DIA = 24 * 3600

# Velocidad inicial calibrada numericamente
v0_luna = 1.076            # km/s

# Condicion inicial: Luna cerca del perigeo
Y0 = np.array([
    RADIO_PERIGEO,  # x0 en km
    0.0,            # y0 en km
    0.0,            # vx0 en km/s
    v0_luna         # vy0 en km/s
])

# EDO DE LA ÓRBITA LUNAR
def f_luna(Y, mu_T):
    x, y, vx, vy = Y

    d_T = np.sqrt(x**2 + y**2)

    ax = -mu_T * x / d_T**3
    ay = -mu_T * y / d_T**3

    return np.array([vx, vy, ax, ay])

# MÉTODO DE EULER
def paso_euler(Y, h, mu_T):
    dY = f_luna(Y, mu_T)
    Y_siguiente = Y + h * dY

    return Y_siguiente

# MÉTODO RK2 PUNTO MEDIO
def paso_rk2(Y, h, mu_T):
    k1 = f_luna(Y, mu_T)

    Y_medio = Y + 0.5 * h * k1

    k2 = f_luna(Y_medio, mu_T)

    Y_siguiente = Y + h * k2

    return Y_siguiente

# INTEGRADOR GENERAL
def integrar(Y0, h, tiempo_total, mu_T, metodo):
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

# MÉTRICAS DE LA ÓRBITA
def calcular_metricas(estados, mu_T):
    x = estados[:, 0]
    y = estados[:, 1]
    vx = estados[:, 2]
    vy = estados[:, 3]

    r = np.sqrt(x**2 + y**2)
    v = np.sqrt(vx**2 + vy**2)

    metricas = {
        "r_min": np.min(r),
        "r_max": np.max(r),
        "v_min": np.min(v),
        "v_max": np.max(v)
    }

    return metricas, r, v

# COMPARACIÓN EULER VS RK2
def comparar_euler_rk2(h):
    tiempo_total = 28 * SEGUNDOS_POR_DIA

    t_euler, estados_euler = integrar(
        Y0, h, tiempo_total, MU_T, metodo="euler"
    )

    t_rk2, estados_rk2 = integrar(
        Y0, h, tiempo_total, MU_T, metodo="rk2"
    )

    metricas_euler, r_euler, v_euler = calcular_metricas(
        estados_euler, MU_T
    )

    metricas_rk2, r_rk2, v_rk2 = calcular_metricas(
        estados_rk2, MU_T
    )

    print("\nComparacion para h =", h, "s")
    print("\nEuler:")
    for clave, valor in metricas_euler.items():
        print(clave, "=", valor)

    print("\nRK2:")
    for clave, valor in metricas_rk2.items():
        print(clave, "=", valor)

    return (
        t_euler, estados_euler, r_euler, v_euler,
        t_rk2, estados_rk2, r_rk2, v_rk2
    )

# GRÁFICOS DE VALIDACIÓN DE UNA ÓRBITA
def graficar_comparacion(
    t_euler, estados_euler, r_euler,
    t_rk2, estados_rk2, r_rk2,
    h
):
    dias_euler = t_euler / SEGUNDOS_POR_DIA
    dias_rk2 = t_rk2 / SEGUNDOS_POR_DIA

    x_euler = estados_euler[:, 0] / 1000
    y_euler = estados_euler[:, 1] / 1000

    x_rk2 = estados_rk2[:, 0] / 1000
    y_rk2 = estados_rk2[:, 1] / 1000

    # Grafico 1: trayectoria en el plano x-y
    plt.figure(figsize=(8, 8))
    plt.plot(x_euler, y_euler, label="Euler")
    plt.plot(x_rk2, y_rk2, label="RK2")
    plt.scatter(0, 0, label="Tierra")
    plt.xlabel("x [miles de km]")
    plt.ylabel("y [miles de km]")
    plt.title(f"Orbita lunar simulada - h = {h} s")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"comparacion_orbita_luna_h_{h}.png")
    plt.close()

    # Grafico 2: distancia Tierra-Luna
    plt.figure(figsize=(10, 5))
    plt.plot(dias_euler, r_euler, label="Euler")
    plt.plot(dias_rk2, r_rk2, label="RK2")
    plt.axhline(RADIO_PERIGEO, linestyle=":", label="Perigeo esperado")
    plt.axhline(RADIO_APOGEO, linestyle=":", label="Apogeo esperado")
    plt.xlabel("Tiempo [dias]")
    plt.ylabel("Distancia Tierra-Luna [km]")
    plt.title(f"Distancia Tierra-Luna en el tiempo - h = {h} s")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"comparacion_luna_h_{h}.png")
    plt.close()


# ANÁLISIS DE DISTINTOS PASOS TEMPORALES

def analizar_pasos_temporales(lista_h, dias_simulados):
    tiempo_total = dias_simulados * SEGUNDOS_POR_DIA
    resultados = []

    for h in lista_h:
        t_euler, estados_euler = integrar(
            Y0, h, tiempo_total, MU_T, metodo="euler"
        )

        t_rk2, estados_rk2 = integrar(
            Y0, h, tiempo_total, MU_T, metodo="rk2"
        )

        metricas_euler, r_euler, v_euler = calcular_metricas(
            estados_euler, MU_T
        )

        metricas_rk2, r_rk2, v_rk2 = calcular_metricas(
            estados_rk2, MU_T
        )

        resultados.append({
            "h": h,
            "euler": metricas_euler,
            "rk2": metricas_rk2
        })

    print("\n" + "=" * 70)
    print(f"Análisis de distintos pasos h durante {dias_simulados} días")
    print("=" * 70)

    for resultado in resultados:
        h = resultado["h"]

        print(f"\nPaso h = {h} s")

        print("Euler:")
        print(f" r_min = {resultado['euler']['r_min']:.2f} km")
        print(f" r_max = {resultado['euler']['r_max']:.2f} km")
        print(f" v_min = {resultado['euler']['v_min']:.6f} km/s")
        print(f" v_max = {resultado['euler']['v_max']:.6f} km/s")

        print("RK2:")
        print(f" r_min = {resultado['rk2']['r_min']:.2f} km")
        print(f" r_max = {resultado['rk2']['r_max']:.2f} km")
        print(f" v_min = {resultado['rk2']['v_min']:.6f} km/s")
        print(f" v_max = {resultado['rk2']['v_max']:.6f} km/s")

    return resultados


# GRÁFICO DE APOGEO VS PASO TEMPORAL

def graficar_comparacion_apogeo(resultados_h, dias_simulados):
    h_values = [res["h"] for res in resultados_h]
    r_max_euler = [res["euler"]["r_max"] for res in resultados_h]
    r_max_rk2 = [res["rk2"]["r_max"] for res in resultados_h]

    plt.figure(figsize=(8, 5))
    plt.plot(h_values, r_max_euler, marker="o", label="Euler")
    plt.plot(h_values, r_max_rk2, marker="o", label="RK2")
    plt.axhline(RADIO_APOGEO, linestyle=":", label="Apogeo esperado")
    plt.xscale("log")
    plt.xlabel("Paso temporal h [s]")
    plt.ylabel("Apogeo máximo simulado [km]")
    plt.title(f"Comparación del apogeo máximo vs h - {dias_simulados} días")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"comparacion_apogeo_vs_h_{dias_simulados}_dias.png", dpi=300)
    plt.close()

# ANÁLISIS DE LARGO PLAZO
def graficar_distancia_largo_plazo(
  t_euler, r_euler,
  t_rk2, r_rk2,
  h,
  dias_simulados
):
  dias_euler = t_euler / SEGUNDOS_POR_DIA
  dias_rk2 = t_rk2 / SEGUNDOS_POR_DIA

  plt.figure(figsize=(12, 5))
  plt.plot(dias_euler, r_euler, label="Euler")
  plt.plot(dias_rk2, r_rk2, label="RK2")
  plt.axhline(RADIO_PERIGEO, linestyle=":", label="Perigeo esperado")
  plt.axhline(RADIO_APOGEO, linestyle=":", label="Apogeo esperado")
  plt.xlabel("Tiempo [días]")
  plt.ylabel("Distancia Tierra-Luna [km]")
  plt.title(f"Distancia Tierra-Luna a largo plazo - h = {h} s - {dias_simulados} días")
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.savefig(f"distancia_largo_plazo_h_{h}_{dias_simulados}_dias.png", dpi=300)
  plt.close()

def graficar_orbita_largo_plazo(
  estados_euler,
  estados_rk2,
  h,
  dias_simulados
):
  x_euler = estados_euler[:, 0] / 1000
  y_euler = estados_euler[:, 1] / 1000

  x_rk2 = estados_rk2[:, 0] / 1000
  y_rk2 = estados_rk2[:, 1] / 1000

  plt.figure(figsize=(8, 8))
  plt.plot(x_euler, y_euler, label="Euler")
  plt.plot(x_rk2, y_rk2, label="RK2")
  plt.scatter(0, 0, label="Tierra")
  plt.xlabel("x [miles de km]")
  plt.ylabel("y [miles de km]")
  plt.title(f"Trayectoria lunar a largo plazo - h = {h} s - {dias_simulados} días")
  plt.axis("equal")
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.savefig(f"orbita_largo_plazo_h_{h}_{dias_simulados}_dias.png", dpi=300)
  plt.close()

def analizar_largo_plazo(h, dias_simulados):
  resultado = comparar_euler_rk2(h, dias_simulados)

  (
    t_euler, estados_euler, r_euler, v_euler,
    t_rk2, estados_rk2, r_rk2, v_rk2,
    metricas_euler, metricas_rk2
  ) = resultado

  graficar_distancia_largo_plazo(
    t_euler, r_euler,
    t_rk2, r_rk2,
    h,
    dias_simulados
  )

  graficar_orbita_largo_plazo(
    estados_euler,
    estados_rk2,
    h,
    dias_simulados
  )


if __name__ == "__main__":
  # Comparación del apogeo para distintos h

    lista_h = [100, 500, 1000, 1800, 3600, 7200]
    dias_analisis_h = 28

    resultados_h = analizar_pasos_temporales(
        lista_h,
        dias_analisis_h
    )

    graficar_comparacion_apogeo(
        resultados_h,
        dias_analisis_h
    )

  # Validación de una órbita lunar aproximada
    h_validacion = 500
    dias_validacion = 28

    resultado_validacion = comparar_euler_rk2(
        h_validacion,
        dias_validacion
    )

    (
        t_euler, estados_euler, r_euler, v_euler,
        t_rk2, estados_rk2, r_rk2, v_rk2,
        metricas_euler, metricas_rk2
    ) = resultado_validacion

    graficar_comparacion(
        t_euler, estados_euler, r_euler,
        t_rk2, estados_rk2, r_rk2,
        h_validacion,
        dias_validacion
    )
  #  Varios meses para observar varias órbitas lunares
    h_largo_plazo = 3600
    dias_largo_plazo = 180

    analizar_largo_plazo(
    h_largo_plazo,
    dias_largo_plazo
    )
    
