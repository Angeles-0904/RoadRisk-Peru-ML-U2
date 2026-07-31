# Guion breve de exposicion

## 1. Problema

El proyecto aborda la priorizacion de accidentes de transito en carretera. La aplicacion estima el riesgo de que un accidente registre fallecidos a partir de datos iniciales como departamento, via, kilometro, modalidad, hora y fecha.

## 2. Producto

Mostrar la app `RoadRisk Peru`:

1. Abrir la pantalla web.
2. Ingresar un caso de accidente.
3. Presionar `Calcular riesgo`.
4. Explicar la probabilidad, el umbral operativo y la clasificacion.
5. Mostrar que tambien existe API en `/predict`, no solo formulario.

## 3. Modelo

Explicar:

- Problema supervisado de clasificacion binaria.
- Target: `fatal = 1` si `FALLECIDOS > 0`.
- Dataset de entrenamiento: SUTRAN 2020-2021 porque contiene casos fatales y no fatales.
- ONSV 2021-2025 se usa para analisis de contexto de siniestros fatales.
- Preprocesamiento: limpieza, fechas, variables temporales, imputacion, escalado y one-hot encoding.
- Modelos comparados: regresion logistica y Random Forest.
- Optimizacion: GridSearchCV con validacion cruzada estratificada y ROC AUC.
- Umbral: se ajusto por F2 para priorizar recall.

Metricas iniciales:

- ROC AUC: 0.6675.
- Recall: 0.7356.
- Precision: 0.1581.
- F1: 0.2602.

## 4. Despliegue

Explicar que esta listo para Render:

- `render.yaml` define build y start.
- Build: instala dependencias y entrena el modelo.
- Start: levanta FastAPI con Uvicorn.
- Tambien existe `Dockerfile` para despliegue por contenedor.

## 5. Integracion continua

Mostrar `.github/workflows/ci.yml`:

- Se ejecuta en push y pull request.
- Instala dependencias.
- Entrena el modelo.
- Ejecuta pruebas.
- Publica artefactos.

## 6. Mantenimiento automatico

Mostrar `.github/workflows/retrain.yml`:

- Se ejecuta automaticamente el primer dia de cada mes.
- Tambien se puede lanzar manualmente.
- Reentrena el modelo con los datos disponibles en `data/raw`.
- Publica el nuevo modelo y las metricas.

Frase clave para defenderlo:

> El modelo no queda congelado despues del primer entrenamiento. El pipeline permite reentrenarlo automaticamente cuando se incorporan nuevos datos, manteniendo el producto dentro de un ciclo regular de mantenimiento de software.

## 7. Cierre

RoadRisk Peru no es solo un notebook: es un producto con aplicacion, API, modelo encapsulado, despliegue preparado, integracion continua, reentrenamiento automatico e informe tecnico para mantenimiento por un equipo TI.

