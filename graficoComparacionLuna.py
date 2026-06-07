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

    print("Comparacion para h =", h, "s")
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
    plt.show()

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
    plt.show()


# Ejecucion para un paso temporal elegido
h = 500

resultado = comparar_euler_rk2(h)

(
    t_euler, estados_euler, r_euler, v_euler,
    t_rk2, estados_rk2, r_rk2, v_rk2
) = resultado

graficar_comparacion(
    t_euler, estados_euler, r_euler,
    t_rk2, estados_rk2, r_rk2,
    h
)
#Anaisis pasos temporales

def analizar_pasos_temporales(lista_h):
    tiempo_total = 28 * SEGUNDOS_POR_DIA
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

    return resultados


lista_h = [100, 500, 1000, 1800, 3600, 7200]

resultados_h = analizar_pasos_temporales(lista_h)

for resultado in resultados_h:
    h = resultado["h"]

    print("\nPaso h =", h, "s")

    print("Euler:")
    print("r_min =", resultado["euler"]["r_min"])
    print("r_max =", resultado["euler"]["r_max"])

    print("RK2:")
    print("r_min =", resultado["rk2"]["r_min"])
    print("r_max =", resultado["rk2"]["r_max"])

if __name__ == "__main__":
    h = 500
    resultado = comparar_euler_rk2(h)

    (
        t_euler, estados_euler, r_euler, v_euler,
        t_rk2, estados_rk2, r_rk2, v_rk2
    ) = resultado

    graficar_comparacion(
        t_euler, estados_euler, r_euler,
        t_rk2, estados_rk2, r_rk2,
        h)
    
    lista_h = [100, 500, 1000, 1800, 3600, 7200]
    analizar_pasos_temporales(lista_h)