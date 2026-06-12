#  Trayectoria de la Cápsula Orion de la Misión Artemis II

## Descripción
Este proyecto contiene la simulación numérica de la órbita lunar y la trayectoria de la cápsula Orion durante la misión Artemis II. 
El modelo resuelve el sistema de ecuaciones diferenciales en dos dimensiones e implementa la comparación entre los métodos de Euler, 
Runge-Kutta de orden 2 (RK2) y Euler-Cromer.

## Requisitos y dependencias
Para poder ejecutar el código, es necesario tener instalado **Python 3** junto con las siguientes librerías:
- `numpy`
- `pandas`
- `matplotlib`

## Ejecución
El proyecto está dividido en dos scripts principales para facilitar su evaluación. Para correr las simulaciones y generar
automáticamente todos los gráficos presentados en el informe, abra la consola en el directorio del proyecto y ejecute los siguientes comandos
según lo que se requiera ver:

**1. Análisis de la órbita lunar y métodos numéricos (Puntos 1, 4, 5 y 6)**

Este script ejecuta la validación de la órbita de la Luna, el análisis de los distintos pasos temporales (h) y la simulación a largo plazo (180 días) para comparar Euler, RK2 y Euler-Cromer.
```bash
python3 graficoComparacionLuna.py
```

**2. Simulación de la trayectoria de Orion (Puntos 2 y 3)**

Este script carga los datos de la telemetría, aplica los ajustes que realizamos y simula el viaje de la cápsula y el retorno a la Tierra.
```bash
python3 orion.py
```

## Aclaración sobre las condiciones iniciales
Tal como se justifica en las conclusiones del informe, al adaptar los datos de telemetría 3D a un modelo 2D, se produce una pérdida de energía. 
Para asegurar que la nave alcance la trayectoria correcta, los parámetros de ajuste ya se encuentran configurados por defecto en el código principal:
- **Factor de multiplicador de velocidad:** 1.16
- **Ángulo de ajuste inicial:** -78 grados
