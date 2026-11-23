"""Runner del PoC de factor timing (plan poc-01, Task 3).

Corre de punta a punta: datos -> features -> modelo principal (walk-forward desde 2000)
-> gate de confiabilidad (desde 2005) -> backtest de las 4 estrategias -> métricas,
bootstrap y veredicto contra el criterio preregistrado (AC-4).

Salidas en data/interim/: timing_results.parquet, timing_metrics.parquet,
timing_summary.json.
"""
import json
import warnings

import numpy as np
import pandas as pd

from factorlab.config import DATA_INTERIM
from factorlab.timing import backtest as bt, data, features as ft, models as md

warnings.filterwarnings("ignore")  # convergencia del MLP con early stopping

EVAL_START, EVAL_END = "2005-01-31", "2026-08-31"   # último mes con retorno t+1 conocido
COSTS = {"0pb": 0.0, "10pb": 0.0010, "25pb": 0.0025}
SUBPERIODS = {"2005-2009": ("2005-01-31", "2009-12-31"),
              "2010-2019": ("2010-01-31", "2019-12-31"),
              "2020-2026": ("2020-01-31", EVAL_END)}

factors, fred = data.load_factors(), data.load_fred()
wide = ft.build_wide(factors, fred)
fwd = ft.targets(factors)
panel = ft.stack_panel(wide, fwd)

# --- Modelo principal: ambas variantes, la elegida por validación 1996-1999 -----
primary, tuning = {}, {}
for kind in ("hgb", "mlp"):
    params, table = md.tune_primary(panel, fwd, kind)
    probs, _ = md.walk_forward_primary(panel, kind, params)
    primary[kind] = probs
    tuning[kind] = {"params": {k: str(v) for k, v in params.items()},
                    "valid_excess": float(table.iloc[0]["valid_excess_mensual"])}
chosen = max(tuning, key=lambda k: tuning[k]["valid_excess"])
print(f"variante elegida por validación 1996-1999: {chosen} {tuning}", flush=True)


def run_gate(kind_primary: str):
    probs = primary[kind_primary]
    meta = md.meta_frame(probs, fwd, wide)
    # El meta-modelo solo puede ver predicciones FUERA de muestra del principal.
    assert meta.index.min() >= pd.Timestamp(f"{md.PRIMARY_OOS_START}-01-01")
    meta_kind, meta_table = md.choose_meta_kind(meta)
    gate, gate_log = md.walk_forward_meta(meta, meta_kind)
    return probs, meta, meta_kind, meta_table, gate, gate_log


probs, meta, meta_kind, meta_table, gate, gate_log = run_gate(chosen)
print(f"meta-modelo elegido (Brier 2003-2004): {meta_kind}\n{meta_table}", flush=True)
print(pd.DataFrame(gate_log)[["year", "train_last", "threshold"]].to_string(index=False), flush=True)

# --- Pesos y retornos de las 4 estrategias -------------------------------------
idx = gate.index
w = {
    "ew": bt.weights_equal(idx),
    "mom12": bt.weights_momentum(wide, idx),
    "primary": md.top_k_picks(probs.reindex(idx)),
    "gated": bt.weights_gated(md.top_k_picks(probs.reindex(idx)), gate["abstain"]),
}

rows, returns = [], {}
for cost_name, c in COSTS.items():
    rets = {k: bt.portfolio_returns(v, fwd, cost=c) for k, v in w.items()}
    ew_net = rets["ew"]["net"]
    for k, r in rets.items():
        for period, (a, b) in {"2005-2026": (EVAL_START, EVAL_END), **SUBPERIODS}.items():
            m = bt.metrics(r["net"].loc[a:b], ew_net.loc[a:b], r["turnover"].loc[a:b])
            rows.append({"estrategia": k, "costo": cost_name, "periodo": period, **m})
    if cost_name == "10pb":
        returns = {k: r["net"].loc[EVAL_START:EVAL_END] for k, r in rets.items()}
        turnover = {k: r["turnover"] for k, r in rets.items()}
metrics = pd.DataFrame(rows)

# --- Criterio preregistrado (AC-4), con costo base de 10 pb ----------------------
ev = meta.loc[EVAL_START:EVAL_END].join(gate)
abst = ev[ev["abstain"]]
abst_precision = float((abst["pick_ret"] < abst["ew_ret"]).mean()) if len(abst) else float("nan")
main = metrics[(metrics.costo == "10pb") & (metrics.periodo == "2005-2026")].set_index("estrategia")
sharpe_gate, sharpe_primary = main.loc["gated", "sharpe"], main.loc["primary", "sharpe"]
boot = bt.block_bootstrap_sharpe_diff(returns["gated"], returns["primary"])
works = bool(sharpe_gate > sharpe_primary and abst_precision > 0.5)

# --- Robustez (NO entra al veredicto): el gate sobre la otra variante ----------
other = "hgb" if chosen == "mlp" else "mlp"
p2, m2, mk2, _, g2, _ = run_gate(other)
w2p = md.top_k_picks(p2.reindex(g2.index))
r2p = bt.portfolio_returns(w2p, fwd, 0.0010)["net"].loc[EVAL_START:EVAL_END]
r2g = bt.portfolio_returns(bt.weights_gated(w2p, g2["abstain"]), fwd, 0.0010)["net"].loc[EVAL_START:EVAL_END]
robust = {"variante": other, "meta": mk2,
          "sharpe_primary": float(r2p.mean() / r2p.std() * np.sqrt(12)),
          "sharpe_gated": float(r2g.mean() / r2g.std() * np.sqrt(12)),
          "meses_abstencion": int(g2.loc[EVAL_START:EVAL_END, "abstain"].sum())}

# --- Salidas ------------------------------------------------------------------
results = ev.copy()
results["picks"] = w["primary"].reindex(results.index).apply(
    lambda r: "+".join(r.index[r > 0]), axis=1)
for k, r in returns.items():
    results[f"ret_{k}"] = r
for f in ft.FACTORS:
    results[f"prob_{f}"] = probs[f].reindex(results.index)
results.to_parquet(DATA_INTERIM / "timing_results.parquet")
metrics.to_parquet(DATA_INTERIM / "timing_metrics.parquet", index=False)

summary = {
    "variante_principal": chosen, "tuning": tuning, "meta_kind": meta_kind,
    "meta_brier": meta_table.to_dict("records"),
    "umbrales_por_anio": {int(g["year"]): g["threshold"] for g in gate_log},
    "eval": [EVAL_START, EVAL_END],
    "meses_eval": int(len(ev)), "meses_abstencion": int(len(abst)),
    "precision_abstencion": abst_precision,
    "sharpe_gated": float(sharpe_gate), "sharpe_primary": float(sharpe_primary),
    "bootstrap_gated_menos_primary": boot,
    "veredicto": "FUNCIONA" if works else "NO FUNCIONA",
    "robustez_otra_variante": robust,
    "decision_vigente": {"mes": str(probs.index.max().date()),
                         "picks": "+".join(md.top_k_picks(probs).iloc[-1].pipe(lambda r: r.index[r > 0]))},
}
(DATA_INTERIM / "timing_summary.json").write_text(json.dumps(summary, indent=2, default=str))

print("\n=== MÉTRICAS 2005-2026, costo 10 pb ===", flush=True)
print(main[["retorno_anual", "volatilidad", "sharpe", "max_drawdown", "hit_vs_ew",
            "rotacion_mensual"]].round(3).to_string(), flush=True)
print(f"\nmeses de abstención: {len(abst)} de {len(ev)} | precisión: {abst_precision:.3f}", flush=True)
print(f"bootstrap Sharpe(gated) - Sharpe(primary): {boot}", flush=True)
print(f"VEREDICTO AC-4: {summary['veredicto']}", flush=True)
print(f"robustez ({other}): {robust}", flush=True)
