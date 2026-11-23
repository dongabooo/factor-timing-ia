# PoC poc-01 — Factor timing con gate de confiabilidad

*Rama `poc/factor-timing` · 2026-11-12 · Evaluación fuera de muestra 2005-01 a 2026-08 (260 meses)*

## Resumen

**Veredicto contra el criterio preregistrado: NO FUNCIONA.** El gate de confiabilidad
bajó el Sharpe del modelo principal de 0.44 a 0.36 (costo de 10 pb). En los 146 meses en
que se abstuvo, la elección del modelo principal habría superado a la mezcla neutral el
57.5% de las veces. Su confianza resultó **anti-informativa**: AUC fuera de muestra de
0.43, por debajo del 0.50 de una moneda.

Ninguna estrategia de timing supera, en riesgo-retorno, a quedarse siempre invertido en
los 6 factores por igual (Sharpe 0.55). El modelo principal sí elige mejor que la
mediana (55% de acierto) y empata con equal-weight **antes de costos** (0.552 contra
0.550), pero rota el 75% del portafolio cada mes y los costos se comen la ventaja.

## 1. Criterio preregistrado (AC-4)

Fijado en el plan antes de correr nada: el gate "funciona" si se cumplen **las dos**
condiciones.

| Condición | Medido | ¿Cumple? |
|-----------|--------|:--------:|
| Sharpe(principal + gate) > Sharpe(principal) | 0.361 contra 0.440 | ✗ |
| Precisión de abstención > 50% | 42.5% (62 de 146 meses) | ✗ |

Bootstrap en bloques de 12 meses (2,000 réplicas): la diferencia de Sharpe es **−0.079**,
con IC 90% de **[−0.271, +0.122]**. En el 72% de las réplicas el gate empeora al modelo
principal. El intervalo cruza cero, así que tampoco hay evidencia de que lo empeore de
forma significativa. Lo que sí queda claro es que **no hay evidencia de que lo mejore**.

## 2. Las 4 estrategias, 2005-2026, costo de 10 pb

| Estrategia | Retorno anual | Volatilidad | Sharpe | Máx. drawdown | Rotación mensual |
|------------|--------------:|------------:|-------:|--------------:|-----------------:|
| Equal-weight (siempre invertido) | 2.2% | 4.2% | **0.550** | −12.3% | 0.00 |
| Momentum 12m (rotación ingenua) | 3.8% | 9.7% | 0.436 | −24.5% | 0.38 |
| Modelo principal (MLP) | 3.3% | 8.2% | 0.440 | −28.2% | 0.75 |
| Principal + gate | 2.1% | 6.5% | 0.361 | −22.4% | 0.49 |

*Los retornos son en exceso: factores long-short y mercado menos tasa libre de riesgo.
El hit rate contra equal-weight del gate (21.9% en la tabla del notebook) está deprimido
por construcción: en los meses de abstención el gate ES equal-weight y empata. Contado
solo en los meses en que invirtió, su hit rate es 51.8%.*

**Sensibilidad a costos** (Sharpe):

| Estrategia | 0 pb | 10 pb | 25 pb |
|------------|-----:|------:|------:|
| Equal-weight | 0.550 | 0.550 | 0.550 |
| Momentum 12m | 0.482 | 0.436 | 0.365 |
| Modelo principal | **0.552** | 0.440 | 0.273 |
| Principal + gate | 0.453 | 0.361 | 0.223 |

**Por subperiodo** (Sharpe, 10 pb):

| Estrategia | 2005–2009 | 2010–2019 | 2020–2026 |
|------------|----------:|----------:|----------:|
| Equal-weight | 0.106 | 0.762 | 0.622 |
| Momentum 12m | 0.275 | 0.302 | 0.742 |
| Modelo principal | −0.107 | 0.696 | 0.711 |
| Principal + gate | −0.118 | **0.985** | 0.343 |

El gate gana con claridad en 2010–2019. Pero un resultado que solo aparece en un
subperiodo y se invierte en el siguiente no es evidencia: buscar el subperiodo
favorable sería justo lo que el criterio preregistrado prohíbe.

## 3. Diagnóstico: por qué falla el gate

- **Su confianza apunta al revés.** Los meses con menor p_meta (quintil inferior,
  ~0.42) son aquellos en que el principal acertó **más** (71%). La probabilidad media
  cuando el principal falló (0.578) es mayor que cuando acertó (0.552).
- **Se abstiene demasiado.** 56% de los meses en total; 71% en 2010–2019. El umbral se
  elige cada año con los últimos 36 meses de la ventana de entrenamiento, y ese tramo es
  ruidoso: el umbral salta entre 0 y 0.70 de un año a otro.
- **Muestra chica para meta-labeling.** El meta-modelo arranca con 59 meses de
  entrenamiento en 2005 y llega a ~310 en 2026. La regresión logística ni siquiera
  supera a una constante en validación (Brier 0.35 contra 0.25), y gradient boosting
  apenas lo hace (0.243).
- **Lo que sí funciona es el modelo principal como selector**: 55% de acierto contra
  la mediana, sin fugas, y en riqueza acumulada termina en $2.02 por cada $1 contra
  $1.62 del equal-weight. Lo hace con el doble de volatilidad y con mucha rotación.

## 4. Robustez (no entra al veredicto)

El gate aplicado sobre la otra variante del principal (gradient boosting, que no fue
elegida en validación) también empeora: Sharpe de 0.624 a 0.552, con 101 meses de
abstención. El resultado negativo del gate no depende de qué modelo principal se use.

Nota: ese gradient boosting tiene un Sharpe de 0.624, mayor que el equal-weight. No se
adopta: la elección del principal se congeló con validación 1996–1999, que favoreció a
la MLP, y cambiarla después de ver 2005–2026 sería seleccionar con el periodo de
evaluación.

## 5. Diseño y controles

| Elemento | Implementación |
|----------|----------------|
| Datos | Ken French (FF5 + Momentum, mensual, 1963-07 a 2026-09) y FRED (T10Y3M, BAA10Y, UNRATE, CPI) |
| Muestra de modelado | 1986-01 a 2026-09 (BAA10Y empieza en 1986) |
| Features | Por factor: retornos a 1/3/6/12m, volatilidad y drawdown de 12m, más su rango entre factores. Mercado: curva, crédito y sus cambios a 3m, volatilidad realizada de 3m. Macro: desempleo y CPI rezagados 1 mes |
| Modelo principal | Ranking: P(factor supera a la mediana de los 6 en t+1). Top-2 en equal-weight. MLP (32, 16) elegida contra gradient boosting con validación 1996–1999 |
| Gate | Meta-labeling: P(top-2 supera a equal-weight en t+1). Gradient boosting elegido contra logística con Brier 2003–2004. Umbral elegido cada año dentro de la ventana de entrenamiento |
| Walk-forward | Ventana expansiva, reentrenando cada enero, con embargo de 1 mes. Principal fuera de muestra desde 2000; gate desde 2005 |
| No-fuga | Features recalculados truncando los datos en t: 20 de 20 idénticos. Control negativo (quitando el rezago de UNRATE): 0 de 20, así que el test sí detecta fugas |
| Meta sin contaminación | El meta-modelo solo ve predicciones fuera de muestra del principal (assert en el runner) |

## 6. Limitaciones

- **Vintage de FRED:** se usa el dato actual, no el publicado en su momento. Las
  revisiones de UNRATE y CPI introducen un sesgo menor. Corregirlo requiere ALFRED.
- **Factores académicos, no investibles:** los factores de Ken French son portafolios
  long-short teóricos. Implementarlos exige shortear y paga costos mayores a los
  modelados; los ETFs de factores son long-only y no los replican.
- **Costos simplificados:** un costo lineal por rotación, y el rebalanceo del
  equal-weight se trata como gratuito.
- **Ken French también se revisa** entre versiones (cambios menores de CRSP).
- **VIX excluido de los modelos** porque empieza en 1990; se sustituye con volatilidad
  realizada.

## 7. Qué significa para la Fase 2 de la tesina

1. **El meta-labeling no es gratis.** Con ~300 meses de historia, un segundo modelo no
   aprende cuándo desconfiar del primero. La Fase 2 necesita más observaciones: por
   ejemplo, decisiones semanales, o el panel de factores descubiertos en la Fase 1, que
   multiplica las series. También puede necesitar un gate más simple (reglas de
   régimen, umbrales de dispersión) en vez de otro modelo de ML.
2. **Los costos deciden.** Un selector con 55% de acierto empata antes de costos y
   pierde después. La Fase 2 debe penalizar la rotación dentro del objetivo, no solo
   medirla al final.
3. **El baseline equal-weight es difícil de vencer.** Debe ser la referencia principal
   de la Fase 2, no solo "siempre invertido en el mercado".
4. El código de `src/factorlab/timing/` (walk-forward con embargo, test de no-fuga,
   bootstrap pareado) se reutiliza tal cual.

## Archivos

| Archivo | Contenido |
|---------|-----------|
| `src/factorlab/timing/` | data, features, models, backtest |
| `scripts_run_timing.py` | Corrida completa (log en `data/interim/timing_run.log`) |
| `data/interim/timing_results.parquet` | Una fila por mes: probabilidades, picks, p_meta, decisión y retornos |
| `data/interim/timing_metrics.parquet` | Métricas por estrategia, costo y subperiodo |
| `data/interim/timing_summary.json` | Veredicto, umbrales por año, bootstrap y robustez |
| `notebooks/poc-01-factor-timing.ipynb` | Figuras (copias en `reports/figures/poc-01/`) |
