# Informe tecnico - RoadRisk Peru

## 1. Resumen del producto

RoadRisk Peru es una aplicacion web/API que estima la probabilidad de que un accidente de transito en carretera registre fallecidos. El objetivo no es reemplazar la evaluacion policial o medica, sino priorizar alertas y apoyar decisiones tempranas de respuesta.

## 2. Datos utilizados

Fuentes:

- `data/raw/sutran_accidentes_2020_2021.csv`: base SUTRAN con accidentes en carreteras, fallecidos y heridos.
- `data/raw/onsv_siniestros_fatales_2021_2025.csv`: base ONSV de siniestros fatales.
- `data/raw/onsv_personas_involucradas_2021_2025.csv`: base ONSV de personas involucradas.

La base SUTRAN se usa para entrenamiento supervisado porque contiene casos con `FALLECIDOS = 0` y `FALLECIDOS > 0`. Las bases ONSV se usan como contexto tecnico para describir el patron de siniestros fatales 2021-2025.

## 3. Variable objetivo

La variable objetivo es:

```text
fatal = 1 si FALLECIDOS > 0
fatal = 0 si FALLECIDOS = 0
```

Es una tarea de clasificacion binaria.

## 4. Caracteristicas del modelo

Variables de entrada:

- Departamento.
- Codigo de via.
- Kilometro.
- Modalidad del accidente.
- Hora del siniestro.
- Mes.
- Dia de semana.
- Indicador de noche.

Se evita usar la cantidad de fallecidos como entrada porque es la variable objetivo. El numero de heridos tambien se excluye del modelo base para reducir fuga de informacion y permitir una prediccion mas temprana.

## 5. Limpieza y preprocesamiento

El pipeline ejecuta:

- Lectura con codificacion `latin1` y separador `;`.
- Normalizacion de nombres de columnas.
- Conversion de fechas `YYYYMMDD`.
- Extraccion de hora, mes y dia de semana.
- Limpieza numerica del kilometro.
- Imputacion de valores faltantes.
- Escalado de variables numericas.
- One-hot encoding de variables categoricas.

## 6. Entrenamiento e hiperparametros

Se comparan dos familias de modelos:

- Regresion logistica balanceada.
- Random Forest con pesos balanceados.

La optimizacion usa `GridSearchCV` con validacion cruzada estratificada de 5 particiones y metrica principal `ROC AUC`. Luego se selecciona un umbral operativo optimizado con F2, dando mas peso al recall porque en seguridad vial conviene reducir falsos negativos.

Los resultados exactos quedan en `models/metrics.json` despues de ejecutar el entrenamiento.

Resultados del entrenamiento inicial ejecutado el 29/05/2026:

- Filas modeladas: 8,067.
- Tasa de accidentes con fallecidos: 11.71%.
- Mejor modelo: Random Forest.
- Hiperparametros seleccionados: `n_estimators=180`, `max_depth=8`, `min_samples_leaf=8`.
- ROC AUC de prueba: 0.6675.
- Recall de prueba: 0.7356.
- Precision de prueba: 0.1581.
- F1 de prueba: 0.2602.
- Umbral operativo: 0.42.

La precision es baja porque la clase fatal es minoritaria. Para una aplicacion de seguridad vial se priorizo recall, ya que es mas grave no alertar un posible caso fatal que generar una alerta preventiva adicional.

## 7. Aplicacion

La aplicacion esta implementada con FastAPI:

- `GET /`: interfaz web.
- `POST /predict-form`: prediccion desde formulario.
- `POST /predict`: prediccion como API JSON.
- `GET /health`: endpoint de salud.

## 8. Despliegue

Opcion recomendada: Render.

Archivos incluidos:

- `render.yaml`: configuracion de despliegue.
- `requirements.txt`: dependencias.
- `Dockerfile`: despliegue alternativo por contenedor.

Flujo sugerido:

1. Subir el proyecto a GitHub.
2. Conectar el repositorio con Render.
3. Render ejecuta `pip install -r requirements.txt && python -m src.roadrisk.train`.
4. Render inicia `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.

## 9. Integracion continua

El workflow `.github/workflows/ci.yml` se ejecuta en cada push o pull request:

- Instala dependencias.
- Entrena el modelo.
- Ejecuta pruebas automatizadas.
- Publica artefactos del modelo.

Esto demuestra que el software no depende de ejecutar pasos manuales antes de integrarse.

## 10. Mantenimiento y reentrenamiento automatico

El workflow `.github/workflows/retrain.yml` se ejecuta:

- Manualmente con `workflow_dispatch`.
- Automaticamente el dia 1 de cada mes.

El flujo reentrena el modelo con los datos presentes en `data/raw`, valida pruebas y publica los artefactos generados. Si se agrega una nueva version del CSV al repositorio, el siguiente ciclo de reentrenamiento incorpora esos datos sin rehacer manualmente el proceso.

En una operacion real, el equipo TI podria reemplazar o versionar el archivo SUTRAN con nuevos cortes de datos. El pipeline mensual ejecutaria el mismo proceso de limpieza, optimizacion, evaluacion y publicacion de artefactos. Si las metricas bajan frente a la version anterior, el equipo podria detener el despliegue o ajustar el umbral antes de liberar el nuevo modelo.

## 11. Organizacion del codigo

```text
app/                  Aplicacion FastAPI
src/roadrisk/          Codigo de datos, entrenamiento y prediccion
data/raw/              CSV originales
data/processed/        Datos limpios generados
models/                Modelo y metricas
reports/               Informe y figuras
tests/                 Pruebas automatizadas
.github/workflows/     CI y reentrenamiento automatico
```

## 12. Consideraciones de mantenimiento

- Versionar cambios de datos con nombres claros.
- Revisar `models/metrics.json` despues de cada reentrenamiento.
- Comparar recall, precision y ROC AUC contra la version anterior.
- Si el recall baja demasiado, ajustar el umbral operativo o ampliar datos.
- Mantener separadas las fuentes de entrenamiento y las fuentes de contexto para evitar fuga de informacion.
