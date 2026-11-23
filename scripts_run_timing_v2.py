"""Runner del experimento v2 (plan poc-02): freno de rotación sobre el modelo principal.

Exploratorio: se diseñó DESPUÉS de ver los resultados de poc-01. Para no contaminarlo:
- el tamaño del freno (delta) se elige solo con 1996-2004 y se congela antes de evaluar;
- el criterio se fijó en el plan: Sharpe(v2) > 0.440 a 10 pb en 2005-01..2026-08;
- un solo intento: si no funciona, no se prueba otro delta.

Salidas: data/interim/timing_v2_results.parquet, timing_v2_summary.json.
"""
import json
import warnings

import numpy as np
import pandas as pd

from factorlab.config import DATA_INTERIM
from factorlab.timing import backtest as bt, data, features as ft, models as md

warnings.filterwarnings("ignore")

EVAL_START, EVAL_END = "2005-01-31", "2026-08-31"
VALID = ("1996-01-31", "2004-12-31")
DELTAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]
COSTS = {"0pb": 0.0, "10pb": 0.0010, "25pb": 0.0025}
SUBPERIODS = {"2005-2009": ("2005-01-31", "2009-12-31"),
              "2010-2019": ("2010-01-31", "2019-12-31"),
              "2020-2026": ("2020-01-31", EVAL_END)}
CRITERION = 0.440   # Sharpe del modelo principal en poc-01, costo 10 pb
BASE_COST = COSTS["10pb"]


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std(ddof=1) * np.sqrt(12))


factors, fred = data.load_factors(), data.load_fred()
wide = ft.build_wide(factors, fred)
fwd = ft.targets(factors)
panel = ft.stack_panel(wide, fwd)
cols = ft.model_columns(panel)

# --- Probabilidades del MLP (mismo procedimiento que poc-01) ---------------------
params, _ = md.tune_primary(panel, fwd, "mlp")
probs_oos, _ = md.walk_forward_primary(panel, "mlp", params)          # 2000+
tr = panel[(panel.date <= md.TUNE_TRAIN_END) & panel.label.notna()]
va = panel[(panel.date >= "1996-01-31") & (panel.date <= "1999-12-31")]
tune_model = md.make_mlp(**params).fit(tr[cols], tr["label"])
probs_tune = md._to_wide_probs(va, tune_model.predict_proba(va[cols])[:, 1])  # 1996-1999
print(f"MLP params: {params}", flush=True)

# --- AC-1: chequeos del freno -----------------------------------------------------
w0 = bt.weights_hysteresis(probs_oos, 0.0)
assert w0.equals(md.top_k_picks(probs_oos)), "delta=0 debe coincidir con top_k_picks"
w_inf = bt.weights_hysteresis(probs_oos, 1.0)
assert bt.portfolio_returns(w_inf, fwd)["turnover"].iloc[1:].sum() == 0, "delta=1 debe congelar"

# --- AC-2: elegir delta con 1996-2004 ----------------------------------------------
probs_valid = pd.concat([probs_tune, probs_oos.loc[:VALID[1]]])
rows = []
for d in DELTAS:
    r = bt.portfolio_returns(bt.weights_hysteresis(probs_valid, d), fwd, BASE_COST)
    r = r.loc[VALID[0]:VALID[1]]
    rows.append({"delta": d, "sharpe_valid_10pb": sharpe(r["net"]),
                 "rotacion_valid": float(r["turnover"].mean())})
valid_table = pd.DataFrame(rows)
rot = valid_table["rotacion_valid"].to_numpy()
assert np.all(np.diff(rot) <= 1e-12), f"la rotación debe bajar con delta: {rot}"
best = valid_table["sharpe_valid_10pb"].max()
delta = float(valid_table.loc[valid_table["sharpe_valid_10pb"] >= best - 0.005, "delta"].min())
print(valid_table.round(4).to_string(index=False), flush=True)
print(f"DELTA CONGELADO (validación 1996-2004): {delta}", flush=True)
(DATA_INTERIM / "timing_v2_delta.json").write_text(json.dumps(
    {"delta": delta, "validacion": valid_table.to_dict("records")}, indent=2))

# --- Evaluación 2005-2026 (solo después de congelar delta) ------------------------
idx = probs_oos.index      # el freno corre desde 2000-01; se evalúa desde 2005-01
w = {
    "ew": bt.weights_equal(idx),
    "mom12": bt.weights_momentum(wide, idx),
    "primary": md.top_k_picks(probs_oos),
    "v2": bt.weights_hysteresis(probs_oos, delta),
}
metric_rows, rets10 = [], {}
for cost_name, c in COSTS.items():
    rets = {k: bt.portfolio_returns(v, fwd, cost=c) for k, v in w.items()}
    for k, r in rets.items():
        for period, (a, b) in {"2005-2026": (EVAL_START, EVAL_END), **SUBPERIODS}.items():
            m = bt.metrics(r["net"].loc[a:b], rets["ew"]["net"].loc[a:b], r["turnover"].loc[a:b])
            metric_rows.append({"estrategia": k, "costo": cost_name, "periodo": period, **m})
    if cost_name == "10pb":
        rets10 = rets
metrics = pd.DataFrame(metric_rows)
main = metrics[(metrics.costo == "10pb") & (metrics.periodo == "2005-2026")].set_index("estrategia")

s_primary = main.loc["primary", "sharpe"]
assert abs(s_primary - CRITERION) < 0.001, f"el principal no reproduce poc-01: {s_primary}"
s_v2 = float(main.loc["v2", "sharpe"])
works = bool(s_v2 > CRITERION)

ev = {k: r.loc[EVAL_START:EVAL_END] for k, r in rets10.items()}
boot_primary = bt.block_bootstrap_sharpe_diff(ev["v2"]["net"], ev["primary"]["net"])
boot_ew = bt.block_bootstrap_sharpe_diff(ev["v2"]["net"], ev["ew"]["net"])

# Diagnóstico: cuánto costo se ahorró y cuánta ventaja bruta se perdió.
diag = {k: {"retorno_bruto_mensual": float(ev[k]["gross"].mean()),
            "costo_mensual_10pb": float((ev[k]["turnover"] * BASE_COST).mean()),
            "rotacion_mensual": float(ev[k]["turnover"].mean())}
        for k in ("primary", "v2", "ew")}
same_picks = float((w["v2"].loc[EVAL_START:EVAL_END] == w["primary"].loc[EVAL_START:EVAL_END])
                   .all(axis=1).mean())

# --- Salidas -----------------------------------------------------------------------
res = pd.DataFrame({
    "picks_v2": w["v2"].apply(lambda r: "+".join(r.index[r > 0]), axis=1),
    "picks_primary": w["primary"].apply(lambda r: "+".join(r.index[r > 0]), axis=1),
    "gross_v2": rets10["v2"]["gross"], "turnover_v2": rets10["v2"]["turnover"],
    "net10_v2": rets10["v2"]["net"],
}).loc[EVAL_START:EVAL_END]
res.to_parquet(DATA_INTERIM / "timing_v2_results.parquet")
metrics.to_parquet(DATA_INTERIM / "timing_v2_metrics.parquet", index=False)

summary = {
    "experimento": "v2 exploratorio: freno de rotación (histéresis)",
    "delta": delta, "validacion": valid_table.to_dict("records"),
    "criterio": f"Sharpe(v2) > {CRITERION} a 10 pb, {EVAL_START}..{EVAL_END}",
    "sharpe_v2": s_v2, "sharpe_primary": float(s_primary),
    "sharpe_ew": float(main.loc["ew", "sharpe"]),
    "veredicto": "FUNCIONA" if works else "NO FUNCIONA",
    "bootstrap_v2_menos_primary": boot_primary, "bootstrap_v2_menos_ew": boot_ew,
    "diagnostico": diag, "meses_mismos_picks_que_primary": same_picks,
    "decision_vigente": {"mes": str(idx.max().date()),
                         "picks": "+".join(w["v2"].columns[w["v2"].iloc[-1] > 0])},
}
(DATA_INTERIM / "timing_v2_summary.json").write_text(json.dumps(summary, indent=2, default=str))

pd.set_option("display.width", 160)
print("\n=== 2005-2026, 10 pb ===")
print(main[["retorno_anual", "volatilidad", "sharpe", "max_drawdown", "hit_vs_ew",
            "rotacion_mensual"]].round(3).to_string())
print("\n=== Sharpe por costo ===")
print(metrics[metrics.periodo == "2005-2026"].pivot(index="estrategia", columns="costo",
                                                     values="sharpe").round(3).to_string())
print("\n=== Sharpe por subperiodo (10 pb) ===")
print(metrics[metrics.costo == "10pb"].pivot(index="estrategia", columns="periodo",
                                             values="sharpe").round(3).to_string())
print(f"\nbootstrap v2-primary: {boot_primary}\nbootstrap v2-ew: {boot_ew}")
print(f"diagnóstico: {json.dumps(diag, indent=1)}\nmeses con mismos picks: {same_picks:.3f}")
print(f"VEREDICTO poc-02: {summary['veredicto']}  (Sharpe v2 {s_v2:.3f} vs {CRITERION})")
print(f"decisión vigente v2: {summary['decision_vigente']}")
