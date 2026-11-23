# PoC poc-03 — Experimento v3: trimestral, tilt y reducción en estrés

*Rama `poc/factor-timing` · 2026-11-14 · Examen en factores **Developed ex-US** (Ken French), nunca usados antes en el proyecto*

## Resumen

**Veredicto formal contra el criterio preregistrado: FUNCIONA.** En el examen internacional
(2001-03 a 2026-08, 10 pb), el v3 obtuvo Sharpe **1.434** contra **1.128** del top-2 mensual.

**Lectura correcta: el v3 se convirtió en equal-weight.** La calibración en EE.UU.
1996–2004 eligió **λ = 0, sin tilt**. Con λ = 0, el freno por estrés no tiene nada que
reducir (κ = 1 por desempate), así que el v3 es exactamente la cartera equal-weight. Lo
que ganó en el examen limpio no fueron las tres ideas. Fue la conclusión disciplinada de
**no hacer timing**.

**Lo más interesante está debajo del veredicto.** El modelo trimestral sí tiene habilidad
fuera de muestra, incluso en el mercado que nunca vio (sección 3). Pero en el periodo de
calibración esa habilidad no se notaba, y por eso no se usó. Es el espejo de poc-02: allá
la validación era demasiado favorable, aquí demasiado desfavorable.

## 1. Criterio preregistrado

Fijado en `poc-03-PLAN.md` (commit `51c2b1f`) antes de cargar datos internacionales:

| Condición | Medido | ¿Cumple? |
|-----------|--------|:--------:|
| Sharpe(v3) > Sharpe(top-2 mensual), Developed ex-US, 10 pb | 1.434 contra 1.128 | ✓ |

Bootstrap pareado v3 − top-2 mensual: diferencia de **+0.306**, IC 90% **[−0.079, +0.629]**.
El top-2 queda por encima del v3 en el 9.7% de las réplicas.

**Defecto del criterio, que se reconoce:** no previmos que la calibración pudiera colapsar
el v3 en equal-weight. Un criterio más fino habría exigido también que el v3 no fuera
idéntico a equal-weight, o lo habría comparado contra él. Queda como lección para
preregistrar la Fase 2.

## 2. Calibración (EE.UU. 1996–2004, 10 pb, promedio de 3 fases de rebalanceo)

| λ \ κ | 0 | 0.25 | 0.5 | 0.75 | 1 |
|------:|------:|------:|------:|------:|------:|
| **0** | **1.325** | 1.325 | 1.325 | 1.325 | 1.325 |
| 0.5 | 0.979 | 1.000 | 1.013 | 1.015 | 1.010 |
| 1 | 0.825 | 0.849 | 0.858 | 0.851 | 0.834 |
| 2 | 0.724 | 0.749 | 0.753 | 0.741 | 0.718 |
| 5 | 0.670 | 0.696 | 0.702 | 0.693 | 0.673 |
| 10 | 0.648 | 0.672 | 0.678 | 0.669 | 0.650 |

- Cualquier inclinación empeoró el resultado en validación.
- El freno por estrés ayudó un poco (+0.03 con κ = 0.5), pero no lo suficiente para
  compensar el tilt.

Los parámetros se escribieron a disco antes de descargar los datos internacionales. El
orden es visible en `timing_v3_run.log`.

## 3. El hallazgo: habilidad que la calibración no vio

Calidad de las elecciones del modelo trimestral (top-2 contra equal-weight, por trimestre):

| Periodo | Trimestres | Exceso por trimestre | Acierto | Correlación de rangos (IC) |
|---------|------:|------:|------:|------:|
| EE.UU. 1996–2004 (calibración) | 36 | 0.36% | 58% | **−0.04** |
| EE.UU. 2005–2026 (ya visto) | 86 | 0.89% | 66% | **+0.14** |
| **Developed ex-US 2001–2026 (nunca visto)** | 102 | 0.73% | 62% | **+0.17** |

- En la calibración, el ranking del modelo trimestral no tenía relación con el resultado
  (IC ≈ 0). Por eso inclinar no ayudaba.
- **En el mercado que nunca vio, el mismo modelo ordena bien los factores** (IC +0.17,
  62% de acierto). Es la evidencia más limpia del PoC de que la señal existe y se
  traslada entre mercados.
- El modelo mensual también eligió bien en ex-US: +0.40% mensual bruto sobre
  equal-weight, más del doble que en EE.UU.

**¿Por qué entonces no gana en Sharpe?** Por diversificación. En ex-US los 6 factores
están poco correlacionados entre sí, así que el equal-weight tiene una volatilidad de
apenas 3.0% anual. Concentrarse en 2 factores sube el retorno pero dispara el riesgo:

| Developed ex-US, 10 pb | Retorno anual | Volatilidad | Sharpe | Máx. caída | Rotación |
|---|---:|---:|---:|---:|---:|
| Equal-weight = v3 | 4.4% | 3.0% | **1.434** | −7.9% | 0.00 |
| Top-2 mensual (poc-01) | **8.4%** | 7.4% | 1.128 | −16.4% | 0.66 |
| Top-2 trimestral | 6.9% | 7.1% | 0.987 | −13.4% | 0.24 |
| Momentum 12m | 7.2% | 8.3% | 0.881 | −28.1% | 0.26 |

Un tilt moderado, entre elegir 2 y no inclinar nada, habría podido capturar parte de la
habilidad sin perder la diversificación. **No lo probamos sobre ex-US** porque eso
consumiría el examen limpio. Queda como hipótesis para la Fase 2.

## 4. EE.UU. 2005–2026 (exploratorio: periodo ya visto)

| Estrategia | Sharpe 0 pb | 10 pb | 25 pb | Rotación |
|---|---:|---:|---:|---:|
| Equal-weight = v3 | 0.550 | 0.550 | 0.550 | 0.00 |
| Top-2 mensual (poc-01) | 0.552 | 0.440 | 0.271 | 0.76 |
| **Top-2 trimestral** | **0.655** | **0.614** | **0.553** | 0.30 |

En EE.UU., el rebalanceo trimestral por sí solo es la única variante del PoC que supera
al equal-weight a cualquier costo. Pero no se repite en ex-US (0.987 contra 1.434) y el
periodo ya estaba visto, así que no es evidencia confirmatoria.

## 5. Ablación (informativa, con λ = 0 y κ = 1 congelados)

Con esos parámetros, dos ablaciones son triviales:
- "Solo tilt" = equal-weight.
- "Solo estrés" = top-2 mensual, porque κ = 1 no reduce nada.

La única ablación informativa es "solo trimestral" (tablas de arriba).

Meses en estrés según la regla: 83 de 306 en ex-US y 37 de 260 en EE.UU.

## 6. Qué significa para la tesina

1. **La señal existe y se traslada entre mercados** (IC +0.17 en datos nunca vistos). La
   pregunta de la Fase 2 no es "¿hay señal?" sino "¿cómo convertirla en mejor
   riesgo-retorno que la diversificación?".
2. **El cuello de botella es la calibración con poca historia.** En poc-02 y en poc-03,
   9 años de validación dieron la respuesta equivocada en direcciones opuestas. La Fase 2
   necesita validar en varios mercados a la vez (el panel internacional multiplica las
   observaciones) o con más regímenes dentro de la ventana.
3. **Concentrar es caro en riesgo.** La forma de explotar la señal debe preservar la
   diversificación: tilts acotados, no elegir 2 de 6.
4. **Preregistrar bien es difícil.** El criterio de poc-03 no previó el colapso a
   equal-weight.
5. **Examen limpio que queda:** Ken French publica también factores de **mercados
   emergentes**, independientes de EE.UU. y de los desarrollados. Es el siguiente examen
   limpio disponible para la Fase 2.

## 7. Limitaciones

- En ex-US, las variables macro son de EE.UU. (FRED) como proxy global. La volatilidad y
  los retornos son de la propia región.
- Hiperparámetros de la MLP heredados de poc-01 (objetivo mensual), sin re-tuning para el
  objetivo trimestral.
- Pesos constantes dentro del trimestre (rebalanceo implícito sin costo), costos
  lineales, factores long-short no investibles directamente. Mismo vintage de FRED que
  poc-01.
- Ex-US arranca en 1990-11, así que su primer año fuera de muestra (2001) entrena con
  ~9 años.

## Archivos

| Archivo | Contenido |
|---------|-----------|
| `src/factorlab/timing/v3.py` | Objetivo de 3 meses, walk-forward con embargo de 3 meses, tilt, estrés y rebalanceo periódico |
| `src/factorlab/timing/data.py` | `load_factors(region)` (aditivo) |
| `scripts_run_timing_v3.py` | Corrida completa; log en `data/interim/timing_v3_run.log` |
| `data/interim/timing_v3_params.json` | (λ, κ) congelados y la rejilla |
| `data/interim/timing_v3_summary.json` | Veredicto, bootstrap y diagnóstico por región |
| `data/interim/timing_v3_returns.parquet`, `timing_v3_weights_*.parquet`, `timing_v3_metrics.parquet` | Series para la demo |
