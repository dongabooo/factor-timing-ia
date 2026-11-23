---
phase: poc-factor-timing
plan: 03
plan_type: execute
wave: 1
depends_on: ["poc-01", "poc-02"]
files_modified:
  - src/factorlab/timing/data.py
  - src/factorlab/timing/v3.py
  - scripts_run_timing_v3.py
  - data/raw/timing/
  - data/interim/timing_v3_*.parquet
  - data/interim/timing_v3_summary.json
  - reports/poc-03-v3-report.md
autonomous: false
description: "v3: asignación trimestral con tilt desde equal-weight y reducción de la apuesta en estrés, calibrada en EE.UU. antes de 2005 y juzgada en factores Developed ex-US nunca vistos"
type: Plan
about: "Tesina-MCD-Factor-Investing-IA"
---

<objective>
## Goal

Probar una sola hipótesis compuesta, el **v3**, con tres cambios propuestos por el autor:

1. **Rebalanceo trimestral.** Decisiones al cierre de marzo, junio, septiembre y diciembre. El modelo predice el retorno de los **3 meses siguientes**.
2. **Asignación porcentual (tilt).** `w_i = max(0, 1/6 + λ·(p_i − media(p)))`, renormalizado a 1. Es long-only. Con λ = 0 equivale a equal-weight.
3. **Menos apuesta en estrés.** `w = EW + s_t·(w_tilt − EW)`.
   - `s_t = κ` si el mercado está en estrés; si no, `s_t = 1`.
   - Estrés en t: retorno del mercado a 12 meses < 0 **y** volatilidad realizada a 3 meses del mercado > su mediana expansiva hasta t (mínimo 36 meses de historia).
   - Con κ = 1 nunca se reduce. Con κ = 0, en estrés se queda en equal-weight.

El examen decisivo es en **factores Developed ex-US (Ken French), que nunca se han usado en este proyecto**.

## Purpose

- **PoC:** convierte un experimento contaminado (EE.UU. 2005-2026 ya se vio) en uno creíble.
- **Tesina:** prueba si las prácticas de índices (rebalanceo periódico, tilts acotados) y el manejo de riesgo por régimen hacen viable el timing de factores, y si la receta se traslada a otro mercado.

## Output
- `data.load_factors(region)` aditivo para Developed ex-US
- `src/factorlab/timing/v3.py`: objetivo de 3 meses, walk-forward con embargo de 3 meses, tilt, estrés y backtest trimestral
- `scripts_run_timing_v3.py`, salidas `timing_v3_*` y `reports/poc-03-v3-report.md`
</objective>

<context>
@.paul/phases/poc-factor-timing/poc-01-SUMMARY.md
@.paul/phases/poc-factor-timing/poc-02-SUMMARY.md
@src/factorlab/timing/data.py
@src/factorlab/timing/features.py
@src/factorlab/timing/models.py
@src/factorlab/timing/backtest.py

## Datos verificados el 2026-11-14 (solo encabezado y fechas, sin mirar retornos)
- `Developed_ex_US_5_Factors_CSV.zip`: Mkt-RF, SMB, HML, RMW, CMA, RF, mensual 1990-07 a 2026-09
- `Developed_ex_US_Mom_Factor_CSV.zip`: columna `WML` (se renombra a `Mom`), 1990-11 a 2026-09
</context>

<acceptance_criteria>

## AC-1: Receta congelada con EE.UU. antes de 2005
```gherkin
Given el MLP con la arquitectura y los hiperparámetros congelados de poc-01 (hidden=(32,16), alpha=1e-3), sin re-tuning
When se elige (λ, κ) en la rejilla λ ∈ {0, 0.5, 1, 2, 3, 5, 10} × κ ∈ {0, 0.25, 0.5, 0.75, 1}
Then el criterio es el Sharpe neto a 10 pb en EE.UU. 1996-01..2004-12, promediado sobre las 3 fases de rebalanceo trimestral (meses de decisión {3,6,9,12}, {1,4,7,10} y {2,5,8,11}) para no depender de un calendario
And 1996-1999 usa el modelo entrenado hasta 1995-09 y 2000-2004 el walk-forward OOS
And en empate (diferencia < 0.01) gana el λ más chico y después el κ más grande, es decir, la opción que menos interviene
And (λ, κ) se escribe a disco ANTES de cargar los datos internacionales y antes de calcular cualquier métrica de EE.UU. 2005+
```

## AC-2: Mecánica correcta y sin fugas
```gherkin
Given el objetivo de 3 meses (retorno compuesto t+1..t+3)
When se entrena cada enero del año Y
Then solo entran meses t <= 30 de septiembre de Y-1 (embargo de 3 meses)
And los asserts verifican: λ=0 → pesos EW; κ=1 → s_t=1 siempre; pesos >= 0 y suman 1; la rotación es cero en los meses que no son de rebalanceo
And dentro del trimestre los pesos se mantienen constantes y el costo se cobra solo en el rebalanceo (simplificación declarada)
```

## AC-3: Veredicto preregistrado en el examen internacional
```gherkin
Given la misma receta (arquitectura, features, λ, κ) aplicada a Developed ex-US, con el MLP entrenado en walk-forward sobre la historia de esa región (primer año OOS: 2001) y las variables macro de EE.UU. (FRED) como proxy global
When se evalúa en 2001-03..2026-08, a 10 pb
Then el v3 "funciona" si Sharpe(v3) > Sharpe(top-2 mensual de poc-01 con la misma región, el mismo entrenamiento y la misma ventana)
And el reporte da FUNCIONA o NO FUNCIONA sin reinterpretar
And como información, fuera del veredicto, se reportan: v3 contra equal-weight, costos 0/10/25 pb, bootstrap pareado (bloque 12, 2000 réplicas), subperiodos, rotación, y lo mismo para EE.UU. 2005-2026 rotulado "exploratorio, periodo ya visto"
And una ablación informativa con (λ, κ) congelados: solo trimestral (top-2), solo tilt (mensual) y solo estrés (top-2 mensual)
And si NO FUNCIONA no se ajusta nada
```

</acceptance_criteria>

<tasks>

<task type="auto">
  <name>Task 1: Datos internacionales, módulo v3 y elección de (λ, κ) en EE.UU.</name>
  <files>src/factorlab/timing/data.py, src/factorlab/timing/v3.py, scripts_run_timing_v3.py</files>
  <action>
    - `data.py` (aditivo):
      - Agregar `load_factors(region="us")` con la opción `"dev_ex_us"`, que reutiliza `_download` y `_parse_french_monthly`.
      - Renombrar `WML` a `Mom` y recortar al rango común.
      - La llamada por defecto debe devolver exactamente lo mismo que hoy.
    - `v3.py`:
      - `targets_3m(factors)`: retorno compuesto de t+1 a t+3, indexado en t.
      - `walk_forward(panel, params, first_year, embargo_months=3)`: copia de la lógica de `models.walk_forward_primary` con el corte de entrenamiento parametrizado. No se modifica `models.py`.
      - `weights_tilt(probs, lam)`, `stress_flag(wide)` y `weights_v3(probs, wide, lam, kappa, phase)`. Los pesos se fijan en los meses de decisión de la fase y se mantienen constantes el resto del trimestre.
    - En el script, Fase A (solo EE.UU., antes de 2005):
      - Construir el panel trimestral de EE.UU.
      - Probabilidades de 1996-1999: modelo entrenado hasta 1995-09. Probabilidades 2000+: walk-forward.
      - Barrer la rejilla de (λ, κ) en 1996-2004, promediando las 3 fases.
      - Elegir y guardar `timing_v3_params.json`.
  </action>
  <verify>Imprime la tabla de la rejilla y el (λ, κ) elegido; pasan los asserts de AC-2; `load_factors()` sin argumentos es idéntico al de poc-01 (assert)</verify>
  <done>AC-1 y AC-2 satisfechos</done>
</task>

<task type="auto">
  <name>Task 2: Examen internacional, EE.UU. exploratorio, ablación y reporte</name>
  <files>scripts_run_timing_v3.py, data/interim/timing_v3_*.parquet, data/interim/timing_v3_summary.json, reports/poc-03-v3-report.md</files>
  <action>
    - Fase B: leer `timing_v3_params.json` (no recalcular).
    - Para cada región (Developed ex-US: OOS desde 2001, evaluación 2001-03..2026-08; EE.UU.: evaluación 2005-01..2026-08):
      - Probabilidades walk-forward del modelo trimestral (para el v3) y del modelo mensual de poc-01 (para el baseline top-2 mensual), las dos entrenadas con la historia de la región.
      - Estrategias: ew, mom12, primary (top-2 mensual), v3 y las 3 ablaciones.
      - Costos 0/10/25 pb, subperiodos y bootstrap.
    - En la región internacional, las features macro son las de FRED (EE.UU.) y la volatilidad realizada sale de su propio Mkt-RF. Declararlo.
    - Guardar los retornos mensuales y los pesos del v3 (para la demo), las métricas y el summary con el veredicto.
    - Reporte en español, mismo estilo que poc-01 y poc-02: veredicto primero; después el examen internacional, EE.UU. exploratorio, la ablación, el diagnóstico y las limitaciones.
  </action>
  <verify>El script corre sin errores; el summary tiene veredicto, (λ, κ) y métricas por región; las cifras del reporte coinciden con el JSON</verify>
  <done>AC-3 satisfecho</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <what-built>Resultado del v3 y su reporte</what-built>
  <how-to-verify>Leer el veredicto y la tabla internacional en reports/poc-03-v3-report.md; confirmar qué entra a la demo</how-to-verify>
  <resume-signal>"aprobado" o ajustes (que no sean cambiar parámetros ni criterio)</resume-signal>
</task>

</tasks>

<boundaries>

## DO NOT CHANGE
- `features.py`, `models.py` y las funciones existentes de `backtest.py`; `data.py` solo cambia de forma aditiva.
- Salidas y reportes de poc-01 y poc-02.
- Fase 1 y la rama `main`.

## SCOPE LIMITS
- Sin re-tuning del MLP ni features nuevas.
- Una sola hipótesis (el v3). Las ablaciones son informativas y usan (λ, κ) congelados.
- Ningún parámetro se elige con datos internacionales ni con EE.UU. 2005+.

</boundaries>

<verification>
- [ ] `timing_v3_params.json` escrito antes de leer los datos internacionales (orden visible en el log)
- [ ] Asserts de AC-2 pasan
- [ ] Veredicto consistente entre el JSON y el reporte
- [ ] `git diff` sin cambios en archivos protegidos
</verification>

<success_criteria>
- Veredicto preregistrado en datos nunca vistos
- Series del v3 listas para la demo (poc-04)
</success_criteria>

<output>
`.paul/phases/poc-factor-timing/poc-03-SUMMARY.md`
</output>
