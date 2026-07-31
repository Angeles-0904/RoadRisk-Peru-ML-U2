# Guion de exposicion - RoadRisk Peru (Unidad 2) · 12 minutos

## Minuto 0-1: Problema y objetivo

> "Cada ano cientos de personas mueren en las carreteras del Peru. El problema no es solo el accidente, sino saber donde y cuando concentrar la prevencion. RoadRisk Peru responde dos preguntas: (1) dado un accidente que acaba de ocurrir, cual es la probabilidad de que haya fallecidos? y (2) existen patrones ocultos de accidentes que nos digan que tipo de siniestro es mas riesgoso? La segunda pregunta es la novedad de esta unidad: la resolvemos con aprendizaje NO supervisado dentro de una aplicacion desplegada en produccion con mantenimiento automatizado."

## Minuto 1-3: Arquitectura del sistema

Mostrar el diagrama de arquitectura:

- Cliente web (dos pestanas: Prediccion de riesgo y Analisis de patrones) -> FastAPI.
- FastAPI expone 8 endpoints: 4 existentes (/, /predict, /predict-form, /health) y 4 nuevos (/api/v1/clusters, /api/v1/clusters/assign, /api/v1/models/registry, /api/v1/monitoring/report).
- Capa de logica `src/roadrisk` con modulos separados: data, train, predict, clustering, train_clustering, registry, retrain, monitoring.
- Capa de modelos: registro versionado `models/random_forest/vN/` y `models/clustering/vN/` + `registry.json`.
- GitHub Actions: ci.yml (push/PR) y retrain.yml (diario).

Frase clave:

> "No creamos una aplicacion aparte: integramos el modulo no supervisado dentro del ciclo de vida del producto existente, manteniendo intacta la funcionalidad de la Unidad 1."

## Minuto 3-5: Dataset y modelos

- Dataset SUTRAN 2020-2021: 8,067 accidentes; 11.7% con fallecidos. ONSV 2021-2025 como contexto.
- 8 variables: departamento, codigo_via, kilometro, modalidad, hora, mes, dia_semana, es_noche.
- Modelo supervisado: Random Forest con GridSearchCV (n_estimators=180, max_depth=8, min_samples_leaf=8), umbral 0.42 optimizado por F2.
- Modelo no supervisado: K-Means. Seleccion de k con tres criterios: metodo del codo, Silhouette y Davies-Bouldin -> k=2.
- Preprocesamiento comun: imputacion, escalado y one-hot encoding.

## Minuto 5-7: Funcionamiento de la aplicacion

DEMO en vivo:

1. Pestana "Prediccion de riesgo": ingresar un caso (LIMA, PE-1S, km 24, DESPISTE, 19h, mayo, lunes, nocturno) -> probabilidad fatal y clasificacion con umbral.
2. Pestana "Analisis de patrones": se muestran los 2 clusters con tarjetas (tamano, tasa fatal, modalidad, departamento, hora).
3. Asignar el mismo siniestro a un cluster: devuelve CL-01 o CL-02 con su perfil y distancia al centroide.
4. Mostrar el monitor de deriva integrado (estado OK / alertas).

Frase clave:

> "El mismo formulario de la Unidad 1 sigue funcionando. La novedad es que ahora el siniestro tambien se clasifica dentro de un patron, y la app monitorea si los datos en produccion se estan desviando de los de entrenamiento."

## Minuto 7-9: MLOps, CI/CD y mantenimiento

- CI (`ci.yml`): en cada push -> instala dependencias, entrena supervisado, entrena clustering, verifica el registro, ejecuta 17 pruebas.
- Mantenimiento (`retrain.yml`): diario 05:00 UTC y manual -> carga datos, entrena un RETADOR, lo compara contra el CAMPEON en produccion:
  - Si el retador mejora recall -> se promueve.
  - Si empeora -> se conserva el campeon y la version queda registrada.
- Resultado real de la validacion: retador v2 con los mismos datos (recall 0.7356 = campeon v1) -> decision KEPT_CHAMPION, produccion se mantiene en v1.
- Monitoreo: PSI para numericas + chi-cuadrado para categoricas, con alertas y estado (OK/REVISION/ACCION_REQUERIDA).
- Versionamiento manual: cada version guarda modelo, metricas y metadatos (fecha, algoritmo, parametros). Estado: random_forest en v1, clustering en v2.

Frase clave:

> "El modelo no queda congelado: el pipeline lo reentrena, lo compara y lo versiona automaticamente cada dia. Solo se promueve una version si realmente mejora a la anterior."

## Minuto 9-11: Resultados

- Supervisado: ROC AUC 0.6675, recall 0.7356, precision 0.1581, F1 0.2602. (Se priorizo recall: es peor no alertar un caso fatal.)
- No supervisado: k=2, silhouette 0.2330, Davies-Bouldin 1.6603.
- Hallazgo principal (patron oculto): los siniestros se separan en diurnos (CL-01, 73%) y nocturnos (CL-02, 27%). El patron nocturno (choques, hora media 20h, 100% nocturno) tiene 9% mas riesgo relativo (1.09x vs 0.97x).
- Monitoreo: estado OK, sin alertas (referencia = datos actuales; el sistema detectaria deriva con un CSV nuevo).
- 17/17 pruebas en verde.

## Minuto 11-12: Conclusiones

1. RoadRisk Peru cumple la Unidad 2: aplicacion en produccion con elementos inteligentes de aprendizaje NO supervisado dentro del ciclo de vida del software.
2. No se elimino funcionalidad: la prediccion de la Unidad 1 sigue operando; se anadio analisis de patrones, versionamiento, monitoreo y retraining.
3. Decisiones de ingenieria justificadas: registro manual en vez de MLflow (cero infraestructura, compatible con Render free), K-Means con seleccion de k por 3 criterios, recall como metrica de negocio.
4. El sistema es mantenible y auditable: versiones historicas, reportes de comparacion y de deriva en cada ciclo.

---

# English version (Unidad 2 presentation - 12 minutes)

## Minutes 0-1: Problem and objective

"Every year, hundreds of people die on Peruvian highways. The real problem is knowing where and when to focus prevention. RoadRisk Peru answers two questions: (1) given a just-occurred accident, how likely is it to involve fatalities? and (2) are there hidden accident patterns that reveal which types of crashes are riskier? The second question is new in this unit: we solve it with UNSUPERVISED learning inside a production application with automated maintenance."

## Minutes 1-3: System architecture

Web client with two tabs (Risk prediction and Pattern analysis) -> FastAPI. Eight endpoints: four from Unit 1 (/, /predict, /predict-form, /health) plus four new ones (clusters, clusters/assign, models/registry, monitoring/report). The `src/roadrisk` package separates concerns: data, train, predict, clustering, registry, retrain, monitoring. Models are versioned under `models/random_forest/vN/` and `models/clustering/vN/`. GitHub Actions runs CI on every push and automated retraining daily.

Key phrase: "We did not create a separate application: we integrated the unsupervised module into the existing product life cycle, keeping Unit 1 functionality untouched."

## Minutes 3-5: Dataset and models

SUTRAN 2020-2021: 8,067 crashes, 11.7% fatal. ONSV 2021-2025 as context. Eight features: department, road code, kilometer, accident type, hour, month, weekday, night indicator. Supervised model: Random Forest with GridSearchCV (180 trees, depth 8, min_samples_leaf 8), 0.42 threshold optimized by F2. Unsupervised model: K-Means; k selected with three criteria (elbow, silhouette, Davies-Bouldin) -> k=2. Shared preprocessing: imputation, scaling, one-hot encoding.

## Minutes 5-7: Application demo

1. "Risk prediction" tab: enter a case (LIMA, PE-1S, km 24, DESPISTE, 7pm, May, Monday, night) -> fatal probability and classification.
2. "Pattern analysis" tab: two cluster cards (size, fatal rate, dominant accident type, department, average hour).
3. Assign the same crash to a cluster: returns CL-01 or CL-02 with its profile and distance to centroid.
4. Show the integrated drift monitor (OK status / alerts).

Key phrase: "The Unit 1 form still works. The novelty: the crash is now also classified within a pattern, and the app monitors whether production data is drifting from training data."

## Minutes 7-9: MLOps, CI/CD and maintenance

CI (`ci.yml`) on every push: install deps, train supervised, train clustering, verify registry, run 17 tests. Maintenance (`retrain.yml`) daily at 05:00 UTC and on demand: load data -> train a CHALLENGER -> compare against the production CHAMPION. If the challenger improves recall, promote it; otherwise keep the champion and only register the new version. Real validation result: challenger v2 with identical data (recall 0.7356 = champion v1) -> KEPT_CHAMPION, production stays at v1. Monitoring uses PSI for numeric features and chi-square for categorical ones, with alerts and status (OK / REVIEW / ACTION REQUIRED). Registry status: random_forest at v1, clustering at v2.

Key phrase: "The model is never frozen: the pipeline retrains, compares, and versions it automatically every day. A version is promoted only if it actually improves on the previous one."

## Minutes 9-11: Results

Supervised: ROC AUC 0.6675, recall 0.7356, precision 0.1581, F1 0.2602 (recall prioritized: failing to alert a fatal case is worse). Unsupervised: k=2, silhouette 0.2330, Davies-Bouldin 1.6603. Main finding: crashes split into daytime (CL-01, 73%) and nighttime (CL-02, 27%) patterns; the nighttime pattern (crashes, average hour 8pm, 100% night) has 9% higher relative risk (1.09x vs 0.97x). Monitoring: OK status, no alerts. All 17 tests pass.

## Minutes 11-12: Conclusions

1. RoadRisk Peru meets Unit 2: a production application with intelligent UNSUPERVISED learning elements inside the software life cycle.
2. No functionality was removed: Unit 1 prediction still runs; we added pattern analysis, model versioning, monitoring, and automated retraining.
3. Justified engineering decisions: manual registry instead of MLflow (zero extra infrastructure, Render-free compatible), K-Means with 3-criteria k selection, recall as the business metric.
4. The system is maintainable and auditable: historical versions, comparison reports, and drift reports on every cycle.
