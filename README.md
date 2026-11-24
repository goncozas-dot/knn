# Modelos KNN para clasificacion

Aplicacion educativa en Python con dos clasificadores K-Nearest Neighbors (KNN):

1. Predice si un cliente estaria satisfecho con el servicio.
2. Predice si un empresario invertiria en un software para automatizar precios.

## Requisitos y ejecucion

### Pagina informativa

Abre `index.html` en un navegador para consultar los dos desarrollos, sus
variables, las matrices de confusion y las metricas de la evaluacion. El diseno
esta pensado para pantallas de PC y no requiere instalar paquetes adicionales.
Las imagenes de las matrices son generadas con Matplotlib. Para generarlas o
actualizarlas con el modelo ejecuta:

```text
python generate_charts.py
```

Tambien puedes servir la carpeta localmente:

```text
python -m http.server 8000
```

Luego visita `http://localhost:8000`. La pagina es informativa; las predicciones
interactivas siguen disponibles en la aplicacion de consola.

### Aplicacion de consola

Se necesita Python 3. Instala las librerias y ejecuta el menu:

```text
pip install -r requirements.txt
python main.py
```

## Variables de entrada

- **Satisfaccion:** calificacion del servicio, tiempo de espera, si se resolvio el
  problema, amabilidad y relacion precio-calidad.
- **Inversion:** presupuesto mensual, numero de empleados, horas dedicadas a
  precios, errores de precios, ahorro estimado y costo del software.

En cada opcion se informa la exactitud y el reporte de clasificacion sobre una
particion de prueba, se muestra la matriz de confusion y luego se solicita un
caso para predecir.

## Importante sobre los datos

La carpeta no contenia datos de clientes ni de empresarios. Por eso, los
conjuntos de entrenamiento se generan dentro de `main.py` como **datos
sinteticos de demostracion**. Las etiquetas siguen reglas simples definidas en
el codigo; los resultados no representan clientes reales ni deben usarse para
decisiones comerciales. Para obtener predicciones fiables, reemplaza las
funciones `create_satisfaction_data` y `create_investment_data` por datos
reales etiquetados con las mismas columnas.

El escalado de variables se incluye porque KNN calcula distancias y las
variables tienen unidades diferentes. El valor de `k` utilizado es 5.
