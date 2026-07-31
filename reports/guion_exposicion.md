# Guion de exposicion - RoadRisk Peru (Unidad 2) · 12 minutos · Equipo de 4

## Division de roles (segun la rubrica)

| Persona | Rubrica que cubre | Pts | Minutos |
|---|---|---|---|
| Integrante 1 | Apertura (problema/objetivo) + Funcionamiento de la aplicacion EN INGLES | 3 | 0-1 y 5-7 |
| Integrante 2 | Entrenamiento del modelo + Resultados | 3 | 3-5 y 9-11 |
| Integrante 3 | Arquitectura del sistema + Despliegue | 2 | 1-3 |
| Integrante 4 | Pipelines de mantenimiento e IC + Pruebas + Conclusiones | 4 | 7-9 y 11-12 |

Enlaces para la exposicion:

- Aplicacion desplegada: `https://roadrisk-peru-ml-u2.onrender.com`
- Repositorio: `https://github.com/Angeles-0904/RoadRisk-Peru-ML-U2`
- Documentacion API: `https://roadrisk-peru-ml-u2.onrender.com/docs`

---

## Minuto 0-1: Problema y objetivo (Integrante 1)

> "Buenos dias. Somos el equipo de RoadRisk Peru. El problema: cada ano ocurren accidentes de transito en las carreteras de Peru, y algunos terminan con fallecidos. Nuestro objetivo es doble: primero, predecir el riesgo de que un accidente sea fatal con un modelo supervisado; segundo, descubrir patrones ocultos de accidentes con aprendizaje no supervisado. Todo en una aplicacion desplegada en produccion, con mantenimiento e integracion continua automaticos. Les presento la arquitectura."

## Minuto 1-3: Arquitectura y despliegue (Integrante 3)

> "Nuestra aplicacion usa FastAPI. Tiene dos partes: prediccion de riesgo y analisis de patrones. El codigo esta organizado en modulos: data para datos, train para entrenamiento, clustering para el modelo no supervisado, registry para versiones y monitoring para monitoreo. Los modelos se guardan versionados en models/. El despliegue es en Render: el Dockerfile instala las dependencias, entrena los dos modelos y levanta la app. El codigo esta en GitHub, en el repositorio RoadRisk-Peru-ML-U2, y los datos vienen de dos fuentes oficiales."

*(transicion)* "Ahora, companera, cuentanos sobre los datos y el modelo."

## Minuto 3-5: Dataset, modelo y entrenamiento (Integrante 2)

> "Trabajamos con SUTRAN 2020-2021: mas de 8,000 accidentes en carreteras. La variable objetivo es fatal: 1 si hubo fallecidos, 0 si no. Usamos 8 caracteristicas: departamento, via, kilometro, modalidad, hora, mes, dia de semana y si es de noche. Para el modelo supervisado comparamos regresion logistica y Random Forest con GridSearchCV. El mejor fue Random Forest con 180 arboles, profundidad 8 y un umbral ajustado para priorizar el recall. El objetivo es no dejar de alertar un accidente fatal."

*(transicion)* "Ahora les mostraremos como funciona la aplicacion."

## Minuto 5-7: Funcionamiento de la aplicacion EN INGLES (Integrante 1)

> "Good morning. Now I am going to show you how the application works."

**Paso 1 - Prediction:**
> "Here we have two tabs. The first one is 'Risk Prediction'. We enter the data of an accident: department Lima, road PE-1S, kilometer 24, type of accident 'despiste' (roll off the road), hour 7 pm, month May, day Monday, and night. We press 'Calculate'. The app gives us a 36% probability of fatalities. The result is 'Moderate Risk'. The model is a Random Forest, and the operating threshold is 0.42. Simple and fast."

**Paso 2 - Patterns:**
> "Now the second tab: 'Pattern Analysis'. This is the unsupervised learning part. The K-Means algorithm found two groups of accidents. CL-01 is the day pattern: more than 5,900 cases, mostly 'despiste', average hour 10 am. CL-02 is the night pattern: more than 2,100 cases, mostly 'choque' (crash), average hour 8 pm, 100% at night."

**Paso 3 - Assign:**
> "We send the same accident to this tab. The app assigns it to CL-02, the night pattern. Why is this important? Because CL-02 has a higher fatal rate: 12.8% versus 11.3%. The night pattern is 9% riskier."

**Paso 4 - Monitoring:**
> "And here we see the monitoring box: the system checks if the data changed. The status is OK, no alerts. Thank you. Now my teammate will explain the pipelines."

## Minuto 7-9: Pipelines, mantenimiento y pruebas (Integrante 4)

> "El mantenimiento esta automatizado con GitHub Actions. Tenemos dos flujos: ci.yml se ejecuta en cada push o pull request; instala dependencias, entrena los dos modelos, verifica el registro y ejecuta 17 pruebas automatizadas. retrain.yml se ejecuta todos los dias a las 5 de la manana; carga los datos, entrena un modelo nuevo llamado retador y lo compara con el que esta en produccion, el campeon. Regla: solo si el retador mejora el recall, se promueve. Si empeora, se conserva el campeon. Tambien reentrena el clustering y ejecuta el monitoreo de deriva con PSI. Cada ciclo genera reportes de comparacion y de monitoreo, y registra la version."

*(transicion)* "Ahora, los resultados."

## Minuto 9-11: Resultados (Integrante 2)

> "Estos son los resultados del modelo supervisado: ROC AUC de 0.67, recall de 0.74, es decir, detectamos el 74% de los accidentes fatales. La precision es 0.16, porque la clase fatal es minoritaria: solo el 12% de los accidentes. Priorizamos el recall porque en seguridad vial es peor no alertar. El clustering encontro k=2 con un silhouette de 0.23. El patron nocturno CL-02 tiene mayor riesgo. El monitoreo reporta estado OK sin alertas. La validacion del reentrenamiento mantuvo la version v1 como campeon, porque el retador no la supero, cumpliendo la regla."

## Minuto 11-12: Conclusiones (Integrante 4)

> "En conclusion, RoadRisk Peru cumple el objetivo de la segunda unidad: es una aplicacion en produccion con aprendizaje no supervisado, con mantenimiento automatico e integracion continua. No eliminamos funcionalidad: la prediccion sigue funcionando y sumamos patrones, versionado, monitoreo y reentrenamiento. El sistema es mantenible: cualquier equipo TI puede reentrenar, comparar y versionar modelos con el informe tecnico. Gracias por su atencion. Preguntas?"

---

## Tips para la demo (Integrante 1)

- Pronunciacion facil: "despiste" -> des-pis-te (roll off the road) · "choque" -> cho-ke (crash) · "fatal rate" -> fei-tal reit.
- Usa la app en vivo mientras hablas: 1) llena el formulario, 2) muestra 36% / Moderate Risk, 3) cambia de pestana y senala CL-01/CL-02, 4) asigna y muestra CL-02, 5) senala "estado=OK".
- Si te trabas: di "In simple words: at night, accidents are riskier".

---

# English version (full, for reference)

## Minutes 0-1: Problem and objective

"Every year, hundreds of people die on Peruvian highways. The real problem is knowing where and when to focus prevention. RoadRisk Peru answers two questions: (1) given a just-occurred accident, how likely is it to involve fatalities? and (2) are there hidden accident patterns that reveal which types of crashes are riskier? The second question is new in this unit: we solve it with UNSUPERVISED learning inside a production application with automated maintenance."

## Minutes 1-3: System architecture

Web client with two tabs (Risk prediction and Pattern analysis) -> FastAPI. Eight endpoints: four from Unit 1 (/, /predict, /predict-form, /health) plus four new ones (clusters, clusters/assign, models/registry, monitoring/report). The `src/roadrisk` package separates concerns: data, train, predict, clustering, registry, retrain, monitoring. Models are versioned under `models/random_forest/vN/` and `models/clustering/vN/`. GitHub Actions runs CI on every push and automated retraining daily.

Key phrase: "We did not create a separate application: we integrated the unsupervised module into the existing product life cycle, keeping Unit 1 functionality untouched."

## Minutes 3-5: Dataset and models

SUTRAN 2020-2021: 8,067 crashes, 11.7% fatal. ONSV 2021-2025 as context. Eight features: department, road code, kilometer, accident type, hour, month, weekday, night indicator. Supervised model: Random Forest with GridSearchCV (180 trees, depth 8, min_samples_leaf 8), 0.42 threshold optimized by F2. Unsupervised model: K-Means; k selected with three criteria (elbow, silhouette, Davies-Bouldin) -> k=2. Shared preprocessing: imputation, scaling, one-hot encoding.

## Minutes 5-7: Application demo (in English)

1. "Risk prediction" tab: enter a case (LIMA, PE-1S, km 24, DESPISTE, 7pm, May, Monday, night) -> fatal probability and classification.
2. "Pattern analysis" tab: two cluster cards (size, fatal rate, dominant accident type, department, average hour).
3. Assign the same crash to a cluster: returns CL-01 or CL-02 with its profile and distance to centroid.
4. Show the integrated drift monitor (OK status / alerts).

Key phrase: "The Unit 1 form still works. The novelty: the crash is now also classified within a pattern, and the app monitors whether production data is drifting from training data."

## Minutes 7-9: MLOps, CI/CD and maintenance

CI (`ci.yml`) on every push: install deps, train supervised, train clustering, verify registry, run 17 tests. Maintenance (`retrain.yml`) daily at 05:00 UTC and on demand: load data -> train a CHALLENGER -> compare against the production CHAMPION. If the challenger improves recall, promote it; otherwise keep the champion and only register the new version. Monitoring uses PSI for numeric features and chi-square for categorical ones, with alerts and status (OK / REVIEW / ACTION REQUIRED).

Key phrase: "The model is never frozen: the pipeline retrains, compares, and versions it automatically every day. A version is promoted only if it actually improves on the previous one."

## Minutes 9-11: Results

Supervised: ROC AUC 0.6675, recall 0.7356, precision 0.1581, F1 0.2602 (recall prioritized: failing to alert a fatal case is worse). Unsupervised: k=2, silhouette 0.2330, Davies-Bouldin 1.6603. Main finding: crashes split into daytime (CL-01, 73%) and nighttime (CL-02, 27%) patterns; the nighttime pattern (crashes, average hour 8pm, 100% night) has 9% higher relative risk (1.09x vs 0.97x). Monitoring: OK status, no alerts. All 17 tests pass.

## Minutes 11-12: Conclusions

1. RoadRisk Peru meets Unit 2: a production application with intelligent UNSUPERVISED learning elements inside the software life cycle.
2. No functionality was removed: Unit 1 prediction still runs; we added pattern analysis, model versioning, monitoring, and automated retraining.
3. Justified engineering decisions: manual registry instead of MLflow (zero extra infrastructure, Render-free compatible), K-Means with 3-criteria k selection, recall as the business metric.
4. The system is maintainable and auditable: historical versions, comparison reports, and drift reports on every cycle.
