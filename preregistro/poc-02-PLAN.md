---
phase: poc-factor-timing
plan: 02
plan_type: execute
wave: 1
depends_on: ["poc-01"]
files_modified:
  - src/factorlab/timing/backtest.py
  - scripts_run_timing_v2.py
  - data/interim/timing_v2_results.parquet
  - data/interim/timing_v2_summary.json
  - reports/poc-02-v2-report.md
autonomous: false
description: "Experimento v2 exploratorio: freno de rotación (banda de histéresis) sobre el modelo principal, con tamaño elegido antes de 2005 y criterio preregistrado"
type: Plan
about: "Tesina-MCD-Factor-Investing-IA"
---

<objective>
## Goal

Probar si un **freno de rotación** convierte la ventaja bruta del modelo principal (MLP, poc-01) en ventaja neta de costos.

Regla del freno (banda de histéresis δ): el portafolio guarda sus 2 factores actuales. Cada mes, el factor más débil del portafolio se sustituye por el mejor de afuera **solo si** la probabilidad del de afuera le gana por más de δ. Se repite hasta 2 veces (puede cambiar ambos factores). Con δ = 0 la regla es idéntica al modelo principal de poc-01.

## Purpose

- **Para el video (narrativa B):** cierra la historia. "El modelo elige bien, pero rota tanto que los costos se comen la ventaja; ¿qué pasa si le enseñamos a quedarse quieto?"
- **Para la tesina:** pone a prueba la lección 2 de poc-01: la Fase 2 debe penalizar la rotación dentro de la decisión.

## Output

- Función `weights_hysteresis` en `backtest.py`, aditiva: no cambia nada de lo existente.
- `scripts_run_timing_v2.py` con salidas propias (`timing_v2_*`). No sobrescribe nada de poc-01.
- `reports/poc-02-v2-report.md` con el veredicto, declarado como **exploratorio**.
</objective>

<context>
@.paul/phases/poc-factor-timing/poc-01-SUMMARY.md
@reports/poc-01-timing-report.md
@src/factorlab/timing/models.py
@src/factorlab/timing/backtest.py
@scripts_run_timing.py
</context>

<acceptance_criteria>

## AC-1: El freno es correcto y con δ=0 reproduce poc-01
```gherkin
Given las probabilidades walk-forward del MLP (mismos hiperparámetros y semilla que poc-01)
When se aplica el freno con δ = 0
Then los pesos son idénticos a top_k_picks y el Sharpe a 10 pb en 2005-01..2026-08 es 0.440 (±0.001)
And con δ grande (1.0) el portafolio nunca cambia después del primer mes (rotación 0)
And la rotación es no creciente al subir δ en la rejilla
```

## AC-2: δ se elige solo con datos anteriores a 2005
```gherkin
Given la rejilla δ ∈ {0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15}
When se elige δ
Then se usa el Sharpe neto a 10 pb en validación 1996-01..2004-12:
  1996-1999 con el modelo de ajuste de poc-01 (entrenado hasta 1995-11) y 2000-2004 con las predicciones walk-forward fuera de muestra
And en empate (diferencia < 0.005) gana el δ más chico
And δ queda congelado antes de calcular cualquier métrica de 2005+ (el script lo imprime y lo guarda antes de evaluar)
```

## AC-3: Veredicto contra criterio preregistrado
```gherkin
Given el criterio fijado en este plan: el v2 "funciona" si Sharpe(v2) > 0.440 (modelo principal de poc-01) a 10 pb en 2005-01..2026-08
When se corre la evaluación
Then el reporte da FUNCIONA o NO FUNCIONA sin reinterpretar el criterio
And se reportan como información (fuera del veredicto): Sharpe vs equal-weight (0.550), sensibilidad 0/10/25 pb, subperiodos, rotación, y bootstrap pareado en bloques de 12 meses de Sharpe(v2) − Sharpe(principal) y Sharpe(v2) − Sharpe(EW)
And el reporte declara que el experimento se diseñó después de ver los resultados de poc-01 (exploratorio, no confirmatorio)
And si el veredicto es NO FUNCIONA no se prueba otro δ ni otra regla
```

</acceptance_criteria>

<tasks>

<task type="auto">
  <name>Task 1: Freno de rotación, verificación y elección de δ</name>
  <files>src/factorlab/timing/backtest.py, scripts_run_timing_v2.py</files>
  <action>
    - `weights_hysteresis(probs, delta, k=2)`: recorre los meses en orden. El primer mes toma el top-k. Después, mientras exista un factor fuera del portafolio cuya probabilidad supere a la del más débil adentro por más de δ, los intercambia (máx. k cambios por mes). Pesos 1/k. Meses con probabilidades NaN conservan el portafolio previo.
    - En el script:
      1. Reconstruir el panel y correr `tune_primary("mlp")` + `walk_forward_primary` igual que poc-01, para obtener las probabilidades OOS 2000+.
      2. Para 1996-1999: ajustar el MLP con los parámetros elegidos sobre ≤1995-11 y predecir 1996-1999.
      3. Concatenar 1996-2004 como serie de validación.
    - Los chequeos de AC-1 van como `assert` en el script: δ=0 igual a top_k_picks; δ=1.0 da rotación 0; rotación monótona en la rejilla.
    - Elegir δ con validación y guardarlo en el summary ANTES de evaluar 2005+.
    - El freno se aplica de forma continua desde 2000-01 (estado inicial) y se evalúa desde 2005-01, para que el portafolio de enero 2005 tenga historia real.
  </action>
  <verify>El script imprime la tabla de validación δ → (Sharpe 10 pb, rotación), el δ elegido, y pasan los 3 asserts; el Sharpe del principal reproducido = 0.440</verify>
  <done>AC-1 y AC-2 satisfechos</done>
</task>

<task type="auto">
  <name>Task 2: Evaluación 2005-2026, veredicto y reporte</name>
  <files>scripts_run_timing_v2.py, data/interim/timing_v2_results.parquet, data/interim/timing_v2_summary.json, reports/poc-02-v2-report.md</files>
  <action>
    - Evaluar v2 junto a ew, mom12 y primary con `portfolio_returns` y `metrics` de poc-01, a costos 0/10/25 pb y en los subperiodos 2005-2009, 2010-2019 y 2020-2026.
    - Bootstrap pareado (bloque 12, 2000 réplicas, semilla de poc-01): v2 − primary y v2 − ew.
    - Guardar los retornos mensuales de v2 (bruto, neto, rotación, picks) en `timing_v2_results.parquet` para la demo, y el veredicto en `timing_v2_summary.json`.
    - Escribir el reporte en español, con el mismo estilo que el de poc-01: el veredicto primero, después las tablas, el diagnóstico (¿cuánta rotación se ahorró y cuánta ventaja bruta se perdió?) y las limitaciones, con el carácter exploratorio explícito.
  </action>
  <verify>`python scripts_run_timing_v2.py` corre sin errores; el JSON tiene veredicto, δ, métricas y bootstrap; el reporte cita las mismas cifras que el JSON</verify>
  <done>AC-3 satisfecho</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <what-built>Resultado del experimento v2 y su reporte</what-built>
  <how-to-verify>
    1. Leer el veredicto y la tabla principal de reports/poc-02-v2-report.md
    2. Confirmar cómo entra el v2 a la demo: curva extra en la pestaña de Backtest
  </how-to-verify>
  <resume-signal>"aprobado" o describir ajustes (que no sean cambiar δ ni el criterio)</resume-signal>
</task>

</tasks>

<boundaries>

## DO NOT CHANGE
- `src/factorlab/timing/{data,features,models}.py`: el motor de poc-01 queda congelado.
- Funciones existentes de `backtest.py`: solo se agrega una función nueva.
- Salidas de poc-01: `timing_results.parquet`, `timing_metrics.parquet`, `timing_summary.json`, `reports/poc-01-timing-report.md`.
- Todo archivo de la Fase 1 y la rama `main`.

## SCOPE LIMITS
- Sin gate en el v2.
- Sin otras reglas de rotación (pesos suaves, rebalanceo trimestral, etc.): una sola hipótesis.
- No se reentrena ni se cambia el modelo principal.
- No se elige nada con datos de 2005+.

</boundaries>

<verification>
- [ ] Los asserts de AC-1 pasan
- [ ] δ elegido y guardado antes de evaluar
- [ ] Veredicto AC-3 en el JSON y en el reporte, consistentes
- [ ] `git diff` no toca archivos protegidos
</verification>

<success_criteria>
- Veredicto preregistrado emitido sin ajustes posteriores
- Datos de v2 listos para la pestaña de Backtest de la demo (poc-03)
</success_criteria>

<output>
`.paul/phases/poc-factor-timing/poc-02-SUMMARY.md`
</output>
