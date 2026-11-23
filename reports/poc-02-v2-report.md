# PoC poc-02 — Experimento v2: freno de rotación

*Rama `poc/factor-timing` · 2026-11-13 · **Exploratorio**: diseñado después de ver los resultados de poc-01*

## Resumen

**Veredicto contra el criterio preregistrado: NO FUNCIONA.** La validación anterior a 2005
eligió **no frenar** (δ = 0). Con δ = 0, el v2 es idéntico al modelo principal: Sharpe
0.440 contra un umbral de "> 0.440". El freno nunca llegó a aplicarse en la evaluación.

Esto no es una falla de ejecución. Es el hallazgo: **en los datos que se podían usar para
diseñar el freno (1996–2004), el modelo tenía tanta ventaja que frenar solo le quitaba
retorno.** Esa ventaja se encogió a menos de una cuarta parte después de 2005, y ahí sí
habría convenido frenar. Pero eso solo se sabe mirando el periodo de evaluación, que es
justo lo que el diseño prohíbe.

## 1. Criterio preregistrado

Fijado en `poc-02-PLAN.md` antes de correr: el v2 funciona si su Sharpe a 10 pb en
2005-01..2026-08 supera **0.440** (el modelo principal de poc-01).

| Condición | Medido | ¿Cumple? |
|-----------|--------|:--------:|
| Sharpe(v2) > 0.440 | 0.440 (δ = 0, idéntico al principal) | ✗ |

## 2. Elección del freno (validación 1996–2004, 10 pb)

| δ | Sharpe en validación | Rotación mensual |
|--:|---------------------:|-----------------:|
| **0** (elegido) | **1.53** | 0.70 |
| 0.01 | 1.32 | 0.69 |
| 0.02 | 1.20 | 0.65 |
| 0.03 | 1.18 | 0.59 |
| 0.05 | 1.09 | 0.56 |
| 0.075 | 1.00 | 0.52 |
| 0.10 | 0.85 | 0.44 |
| 0.15 | 0.68 | 0.25 |

Cada punto de freno bajó el Sharpe en validación. El resultado no depende de usar
1996–1999: esos años son optimistas, porque con ellos se eligieron los hiperparámetros de
la MLP. Pero 2000–2004, que son predicciones puramente fuera de muestra, cuentan la misma
historia: Sharpe de 1.39 sin freno, 1.00 con δ = 0.03 y 0.61 con δ = 0.10.

Controles automáticos (asserts del script):
- Con δ = 0 los pesos son idénticos a los del principal.
- Con δ = 1 la rotación es cero.
- La rotación baja de forma monótona al subir δ.
- El principal reproduce el 0.440 de poc-01.

## 3. Diagnóstico: la ventaja del modelo se encogió

Retorno mensual de la elección top-2 menos equal-weight, antes de costos:

| Periodo | Exceso bruto mensual | Tipo de dato |
|---------|---------------------:|--------------|
| 1996–1999 | **0.89%** | Validación (optimista) |
| 2000–2004 | **0.76%** | Fuera de muestra, antes de la evaluación |
| 2005–2026 | **0.18%** | Evaluación |

Un modelo que rota el 75% de su portafolio cada mes paga unos 0.08% mensuales a 10 pb.
Contra una ventaja de 0.8% eso es ruido, y frenar sale caro. Contra una ventaja de 0.18%,
el costo se lleva casi la mitad. **La rotación no era un problema en el pasado. Se volvió
problema cuando la señal se debilitó.**

**Matiz importante: no es un quiebre limpio en 2005.** 2005 es solo donde cortamos la
muestra. Por bloques de 5 años, el exceso del modelo y el máximo que habría obtenido un
"oráculo" que conoce el futuro y elige los 2 mejores factores cada mes:

| Bloque | Exceso del modelo (%/mes) | Máximo posible (oráculo) | Fracción capturada |
|--------|------:|------:|------:|
| 2000–2004 | 0.76 | 4.10 | 0.18 |
| 2005–2009 | −0.06 | 2.77 | −0.02 |
| 2010–2014 | −0.15 | 2.14 | −0.07 |
| 2015–2019 | 0.54 | 2.65 | 0.20 |
| 2020–2026 | 0.35 | 3.61 | 0.10 |

Lo que hay es una **década perdida, 2005–2014** (crisis financiera, crash del momentum de
2009, tasas en cero), seguida de una recuperación parcial. Además, 2000–2004 fue
excepcional: tras la burbuja .com, HML, CMA, SMB y RMW pagaron entre 10% y 15% anual
mientras el mercado perdía. En 2005–2026 esos mismos factores rindieron cerca de 0%. Esto
es descriptivo, porque mira el periodo de evaluación, y no cambia ninguna decisión.

Esto es consistente con el deterioro de las anomalías después de publicarse que documentan
McLean y Pontiff (2016, *Journal of Finance*). Aquí es una hipótesis: este experimento no
la prueba.

## 4. Evaluación 2005–2026 (información, fuera del veredicto)

Con δ = 0, el v2 reproduce exactamente al principal:

| Estrategia | Sharpe 0 pb | Sharpe 10 pb | Sharpe 25 pb | Rotación |
|------------|------:|------:|------:|------:|
| Equal-weight | 0.550 | 0.550 | 0.550 | 0.00 |
| Momentum 12m | 0.482 | 0.435 | 0.362 | 0.39 |
| Principal = v2 | 0.552 | 0.440 | 0.271 | 0.76 |

Bootstrap pareado v2 − equal-weight: diferencia de −0.110, con IC 90% [−0.414, +0.256].

## 5. Lo que este resultado le dice a la Fase 2

1. **El costo de rotación no se resuelve con un parámetro calibrado en el pasado** si la
   fuerza de la señal cambia en el tiempo. Un freno fijo es la herramienta equivocada.
   Lo que hace falta es que el freno dependa de cuánta ventaja *espera* el modelo, por
   ejemplo maximizando retorno esperado menos costo en cada decisión.
2. **La no estacionariedad es el problema de fondo.** Lo era también del gate de poc-01:
   aprendió a confiar en un régimen que ya no existía. La Fase 2 necesita validación que
   detecte el deterioro de la señal, por ejemplo con ventanas móviles en vez de
   expansivas o con monitoreo del exceso reciente.
3. **El protocolo funcionó.** Sin las reglas preregistradas, habría sido fácil elegir un δ
   mirando 2005–2026 y reportar una "mejora". Dos experimentos honestos con resultado
   negativo valen más que uno ajustado.

## 6. Limitaciones

- **Exploratorio:** la idea surgió de ver poc-01. Aun así, δ se eligió sin tocar 2005+.
- **1996–1999** usa el modelo cuyos hiperparámetros se eligieron con ese mismo periodo, así
  que su Sharpe está inflado. 2000–2004 por sí solo lleva a la misma decisión.
- Se probó una sola regla de freno (histéresis sobre probabilidades). Pesos suaves o
  rebalanceo menos frecuente no se probaron, por diseño: un solo intento.
- Aplican las limitaciones de poc-01: vintage de FRED, factores no investibles y costos
  lineales.

## Archivos

| Archivo | Contenido |
|---------|-----------|
| `scripts_run_timing_v2.py` | Corrida completa (log en `data/interim/timing_v2_run.log`) |
| `src/factorlab/timing/backtest.py` | `weights_hysteresis` (nueva) |
| `data/interim/timing_v2_delta.json` | δ congelado y tabla de validación, escrito antes de evaluar |
| `data/interim/timing_v2_results.parquet` | Picks, rotación y retornos mensuales de v2 |
| `data/interim/timing_v2_summary.json` | Veredicto, bootstrap y diagnóstico |
