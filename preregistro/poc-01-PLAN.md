---
phase: poc-factor-timing
plan: 01
plan_type: execute
wave: 1
depends_on: []
files_modified:
  - pyproject.toml
  - src/factorlab/timing/__init__.py
  - src/factorlab/timing/data.py
  - src/factorlab/timing/features.py
  - src/factorlab/timing/models.py
  - src/factorlab/timing/backtest.py
  - scripts_run_timing.py
  - notebooks/poc-01-factor-timing.ipynb
  - data/raw/timing/
  - data/interim/timing_results.parquet
  - reports/poc-01-timing-report.md
autonomous: false
description: "Motor del PoC: modelo que elige factores mes a mes y un segundo modelo (meta-labeling) que decide cuándo no confiar en él, evaluados fuera de muestra"
type: Plan
about: "Tesina-MCD-Factor-Investing-IA"
---

<objective>
## Goal

Construir y evaluar **fuera de muestra** el motor del PoC:

1. Un **modelo principal** que cada mes elige los 2 factores con mejor perspectiva para el mes siguiente.
2. Un **modelo de confiabilidad** (meta-labeling) que estima si esa elección superará a una mezcla neutral. Si no confía, el portafolio se queda en la mezcla neutral.

Las dos piezas se comparan contra dos baselines: siempre invertido en todos los factores, y rotación ingenua por momentum.

## Purpose

- **Para el PoC** (entrega aproximada: 2026-11-26, demo más video): este plan produce los resultados que la demo de poc-02 muestra. Sin un motor honesto, la demo es solo maquillaje.
- **Para la tesina:** es la Fase 2 en miniatura sobre factores clásicos, que es justo el fallback documentado. Si el gate no aporta, lo sabremos ahora y no en el tetra 5.

## Output

- Módulo `src/factorlab/timing/` con datos, features, modelos y backtest, reutilizable en la Fase 2
- `timing_results.parquet`: por mes, la predicción, la confianza, la decisión y el retorno de cada estrategia
- Notebook con las figuras que después reutiliza la demo
- `reports/poc-01-timing-report.md` con los resultados contra un criterio preregistrado
</objective>

<context>
## Project Context
@.paul/PROJECT.md
@.paul/ROADMAP.md
@.paul/STATE.md

## Source Files
@src/factorlab/config.py
@pyproject.toml

## Fuentes verificadas el 2026-11-12
- Ken French, `F-F_Research_Data_5_Factors_2x3_CSV.zip` y `F-F_Momentum_Factor_CSV.zip`: mensuales, de 1963-07 a 2026-09, construidos con CRSP (sin sesgo de supervivencia)
- FRED, vía `fredgraph.csv?id=` sin API key: T10Y3M (desde 1982), BAA10Y (desde 1986), VIXCLS (desde 1990), UNRATE y CPIAUCSL (desde 1947–48)
- No instaladas en el venv: scikit-learn, streamlit, plotly
</context>

<acceptance_criteria>

## AC-1: Datos descargados, cacheados y alineados sin mirar al futuro
```gherkin
Given las URLs de Ken French y FRED
When se corre la descarga
Then los CSV crudos quedan en data/raw/timing/ y una segunda corrida no vuelve a descargar
And el panel mensual tiene los 6 factores (Mkt-RF, SMB, HML, RMW, CMA, Mom) en decimales, no en porcentaje
And las series macro con publicación rezagada (UNRATE, CPI) se desplazan 1 mes: el dato de mayo solo es usable al cierre de junio
And un test automático confirma que ningún feature del mes t usa información posterior al cierre de t
```

## AC-2: Validación walk-forward sin fugas
```gherkin
Given el panel de features y el retorno del mes siguiente como objetivo
When se entrena el modelo principal
Then se usa una ventana expansiva, reentrenando cada 12 meses, con el primer mes fuera de muestra en 2000-01
And el modelo de confiabilidad se entrena SOLO con predicciones fuera de muestra del modelo principal, así que su evaluación arranca en 2005-01
And cualquier umbral (el de abstención incluido) se elige con datos de entrenamiento, nunca con el periodo de evaluación
```

## AC-3: Cuatro estrategias comparadas con costos y significancia
```gherkin
Given las decisiones mensuales de 2005-01 a 2026-09
When se corre el backtest
Then se reportan 4 estrategias: (1) equal-weight de los 6 factores, (2) top-2 por momentum de 12 meses, (3) modelo principal solo, (4) modelo principal + gate
And cada una trae retorno anual, volatilidad, Sharpe, máximo drawdown, hit rate y rotación
And se aplica un costo de 10 pb por cada cambio de posición, con sensibilidad a 0 y 25 pb
And la diferencia de Sharpe entre (4) y (3) lleva un intervalo de confianza al 90% por bootstrap en bloques de 12 meses
```

## AC-4: El gate se juzga contra un criterio fijado antes de ver resultados
```gherkin
Given el criterio preregistrado: el gate "funciona" si Sharpe(4) > Sharpe(3) Y en los meses de abstención la elección del modelo principal quedó por debajo de la mezcla neutral en más del 50% de los casos
When se evalúa el periodo 2005-01 a 2026-09
Then el reporte da un veredicto FUNCIONA o NO FUNCIONA sin reinterpretar el criterio
And si el veredicto es NO FUNCIONA, el resultado se reporta igual como hallazgo, no se re-ajusta el modelo hasta que pase
```

</acceptance_criteria>

<tasks>

<task type="auto">
  <name>Task 1: Datos, features y prueba de no-fuga</name>
  <files>pyproject.toml, src/factorlab/timing/__init__.py, src/factorlab/timing/data.py, src/factorlab/timing/features.py, data/raw/timing/</files>
  <action>
    **Dependencias.** Agregar a `pyproject.toml` un extra `[project.optional-dependencies] poc = ["scikit-learn>=1.5", "streamlit>=1.40", "plotly>=5.24"]` e instalar con `pip install -e ".[poc]"`. Sin torch ni lightgbm: scikit-learn basta y en macOS evita problemas con libomp.

    **data.py.**
    - Descargar los dos zips de Ken French y parsear solo la tabla MENSUAL. El CSV trae después una tabla anual que hay que cortar: las filas mensuales tienen fecha de 6 dígitos (AAAAMM) y las anuales de 4.
    - Convertir de porcentaje a decimal y unir FF5 con Mom. Conservar RF aparte.
    - FRED: bajar T10Y3M, BAA10Y, VIXCLS, UNRATE y CPIAUCSL. Las series diarias se llevan a fin de mes con el último dato disponible del mes.
    - Cachear en `data/raw/timing/`; si el archivo existe, no se vuelve a descargar.

    **features.py.** Cada feature del mes t usa solo información disponible al cierre de t:
    - Por factor: retorno acumulado a 1, 3, 6 y 12 meses; volatilidad de 12 meses; drawdown desde el máximo de 12 meses.
    - Mercado: T10Y3M, BAA10Y y su cambio a 3 meses. VIX en su versión de fin de mes. Volatilidad realizada del mercado a 3 meses, como sustituto del VIX antes de 1990.
    - Macro rezagada 1 mes: cambio de UNRATE a 3 meses e inflación interanual del CPI.
    - El objetivo del mes t es el retorno de cada factor en t+1.
    - Muestra útil: desde 1986-01, porque BAA10Y empieza ahí. Si el VIX deja hoyos antes de 1990, se excluye de los modelos y se usa la volatilidad realizada.

    **Test de no-fuga:** recalcular los features truncando los datos crudos en el mes t y verificar que coinciden con los del panel completo, en 20 meses al azar.

    Declarar en el reporte: FRED se usa en su vintage actual, no histórico (sin ALFRED). Las revisiones de UNRATE y CPI introducen un sesgo menor.
  </action>
  <verify>
    - `python -c "from factorlab.timing import data; data.load_factors()"` devuelve los 6 factores de 1963-07 a 2026-09 en decimales (la media de Mkt-RF ronda 0.005–0.007)
    - Segunda corrida de la descarga: 0 requests (se ve en el log)
    - Test de no-fuga: 20 de 20 meses idénticos
  </verify>
  <done>AC-1 satisfecho</done>
</task>

<task type="auto">
  <name>Task 2: Baselines y modelo principal en walk-forward</name>
  <files>src/factorlab/timing/models.py, src/factorlab/timing/backtest.py</files>
  <action>
    **Baselines:**
    - Equal-weight de los 6 factores, que es la mezcla neutral.
    - Top-2 por retorno de 12 meses, que es la rotación ingenua.

    **Modelo principal:**
    - Formulación de ranking. Para cada (mes, factor), etiqueta = 1 si el retorno del factor en t+1 supera la mediana de los 6 factores en t+1. Se entrena un clasificador sobre el panel apilado (mes × factor), con la identidad del factor como one-hot.
    - Cada mes se toman los 2 factores con mayor probabilidad, en equal-weight.
    - Dos variantes: `HistGradientBoostingClassifier` (principal) y `MLPClassifier` (una red chica de 1–2 capas con features estandarizados dentro de cada ventana de entrenamiento). Las dos van al reporte; la demo usa la mejor según el periodo de **validación interna**, no el de evaluación.
    - Walk-forward: ventana expansiva desde 1986, reentrenando cada enero, con el primer mes fuera de muestra en 2000-01. Los hiperparámetros se fijan con una búsqueda chica sobre 1986–1999 y quedan congelados.
    - Guardar por mes: los 2 factores elegidos, la probabilidad de cada factor y el retorno realizado de la elección.

    Evitar: estandarizar con estadísticas de toda la muestra, elegir hiperparámetros viendo 2000+, y usar `train_test_split` aleatorio en una serie de tiempo.
  </action>
  <verify>
    - Predicciones fuera de muestra para cada mes de 2000-01 a 2026-09 (sin hoyos)
    - Test: el modelo del año Y solo vio datos con objetivo hasta diciembre de Y−1
    - Accuracy del ranking reportada para ambas variantes
  </verify>
  <done>AC-2 satisfecho para el modelo principal</done>
</task>

<task type="auto">
  <name>Task 3: Gate de confiabilidad (meta-labeling), backtest y reporte</name>
  <files>src/factorlab/timing/models.py, src/factorlab/timing/backtest.py, scripts_run_timing.py, notebooks/poc-01-factor-timing.ipynb, data/interim/timing_results.parquet, reports/poc-01-timing-report.md</files>
  <action>
    **Meta-labeling** (López de Prado; en la línea de Isufaj et al.):
    - Meta-etiqueta del mes t = 1 si el retorno de la elección del modelo principal en t+1 supera al del equal-weight.
    - Features del meta-modelo: las mismas de mercado y macro, más la confianza del principal (la probabilidad del top-2 y la brecha contra el 3.º) y el desempeño reciente del principal (hit rate de los últimos 12 meses, calculado solo con meses ya cerrados).
    - Se entrena **solo con predicciones fuera de muestra** del principal: con 2000–2004 como primer bloque, la evaluación arranca en 2005-01. Reentrenamiento cada enero, con ventana expansiva.
    - Clasificador: regresión logística y `HistGradientBoostingClassifier`; la elección se hace con validación interna.
    - Umbral de abstención: el que maximiza el Sharpe **dentro de la ventana de entrenamiento** de cada año. Si p_meta < umbral, el portafolio se queda en equal-weight.

    **Backtest 2005-01 a 2026-09** de las 4 estrategias:
    - Métricas de AC-3, con costos de 0, 10 y 25 pb por cambio de posición.
    - Bootstrap en bloques de 12 meses (2,000 réplicas, semilla 20261112) para la diferencia de Sharpe entre (4) y (3).
    - Precisión de las abstenciones: en los meses en que el gate se abstuvo, ¿qué % de las veces la elección del principal quedó por debajo del equal-weight?
    - Desglose por subperiodo (2005–2009, 2010–2019, 2020–2026) para ver si el resultado depende de la crisis de 2008.

    **Salidas:**
    - `timing_results.parquet` con una fila por mes: features clave, elección, probabilidades, p_meta, decisión y retorno de las 4 estrategias.
    - Notebook con las figuras que la demo reutiliza: rendimiento acumulado de las 4 estrategias, meses de abstención sobre la línea de tiempo, tabla de métricas y calibración del meta-modelo.
    - Reporte con:
      - el veredicto contra el criterio de AC-4, escrito tal cual;
      - la tabla de métricas;
      - las limitaciones: vintage de FRED, 6 factores académicos y no investibles, costos simplificados, y la factibilidad de shortear los long-short;
      - un párrafo "qué significa para la Fase 2 de la tesina".

    El criterio de AC-4 está fijado ANTES de correr. Si el gate no funciona, se reporta así; no se itera sobre el periodo de evaluación buscando que pase.
  </action>
  <verify>
    - `python scripts_run_timing.py` corre de punta a punta y escribe el parquet y las figuras
    - Test: ninguna fila del meta-modelo de entrenamiento usa una predicción del principal hecha dentro de su propia muestra
    - Reporte con el veredicto FUNCIONA / NO FUNCIONA y el intervalo de confianza de la diferencia de Sharpe
    - Notebook ejecutado sin errores
  </verify>
  <done>AC-2 (gate), AC-3 y AC-4 satisfechos</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <what-built>Motor del PoC evaluado fuera de muestra, con veredicto sobre el gate de confiabilidad</what-built>
  <how-to-verify>
    1. Leer `reports/poc-01-timing-report.md`: el veredicto y la tabla de las 4 estrategias.
    2. Ver la figura de rendimiento acumulado y la de meses de abstención en el notebook.
    3. Decidir la narrativa del video según el resultado: "el gate mejora el riesgo-retorno" o "medimos con honestidad y el gate no aporta: esto es lo que aprendimos".
    4. Confirmar qué 3–4 vistas debe tener la demo de poc-02.
  </how-to-verify>
  <resume-signal>Escribe "aprobado" con la narrativa elegida, o describe qué ajustar</resume-signal>
</task>

</tasks>

<boundaries>

## DO NOT CHANGE
- Todo lo de la Fase 1: `src/factorlab/{config,edgar_probe,descriptors,sectors,universe,returns_probe}.py`, los reportes de 01-01 a 01-03 y el plan 01-04
- `data/interim/*` de la Fase 1
- `.paul/phases/01-factores-latentes/`

## SCOPE LIMITS
- Solo trabajo en la rama `poc/factor-timing`
- Datos: Ken French y FRED, nada más. No hay EDGAR ni panel de empresas en este PoC
- Sin RL, sin LLMs, sin deep learning pesado (torch): una MLP de scikit-learn cubre la parte de redes
- No se itera sobre el periodo de evaluación 2005–2026 para mejorar el resultado
- La demo Streamlit y el guion del video son poc-02, no este plan

</boundaries>

<verification>
Before declaring plan complete:
- [ ] Datos cacheados; test de no-fuga 20/20
- [ ] Predicciones fuera de muestra sin hoyos desde 2000-01; meta-modelo desde 2005-01
- [ ] Test de que el meta-modelo no ve predicciones dentro de muestra
- [ ] 4 estrategias con métricas, costos 0/10/25 pb e intervalo de confianza por bootstrap
- [ ] Veredicto contra el criterio preregistrado de AC-4
- [ ] Notebook y script corren de punta a punta
- [ ] Checkpoint resuelto con la narrativa del video
- [ ] All acceptance criteria met
</verification>

<success_criteria>
- Resultado fuera de muestra honesto, sea positivo o negativo
- Figuras listas para reutilizarse en la demo
- Código en `src/factorlab/timing/` reutilizable para la Fase 2
- Plan cerrado con margen para poc-02: meta, 2026-11-18
</success_criteria>

<output>
After completion, create `.paul/phases/poc-factor-timing/poc-01-SUMMARY.md`
</output>
