"""Runner del experimento v3 (plan poc-03).

Fase A: con EE.UU. 1996-2004 elige (lam, kappa) y los escribe a disco.
Fase B: SOLO después, carga los factores Developed ex-US (nunca vistos) y emite el
veredicto preregistrado: Sharpe(v3) > Sharpe(top-2 mensual de poc-01) a 10 pb en
2001-03..2026-08. EE.UU. 2005-2026 se reporta como exploratorio (periodo ya visto).
"""
import json
import warnings
from itertools import product

import numpy as np
import pandas as pd

from factorlab.config import DATA_INTERIM
from factorlab.timing import backtest as bt, data, features as ft, models as md, v3

warnings.filterwarnings("ignore")

PARAMS = {"hidden": (32, 16), "alpha": 1e-3}   # congelados de poc-01, sin re-tuning
COST = 0.0010
COSTS = {"0pb": 0.0, "10pb": 0.0010, "25pb": 0.0025}
VALID = ("1996-01-31", "2004-12-31")
PARAMS_FILE = DATA_INTERIM / "timing_v3_params.json"


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std(ddof=1) * np.sqrt(12))


def build(region: str):
    """Datos, panel mensual (poc-01) y panel trimestral (objetivo 3m) de una región."""
    factors = data.load_factors(region)
    fred = data.load_fred()
    wide = ft.build_wide(factors, fred)
    fwd1 = ft.targets(factors)
    p1 = ft.stack_panel(wide, fwd1)
    p3 = ft.stack_panel(wide, v3.targets_3m(factors))
    # Meses sin historia de 12m (inicio de la serie internacional) no entran.
    p1, p3 = (p[p["ret_12m"].notna()].reset_index(drop=True) for p in (p1, p3))
    return factors, wide, fwd1, p1, p3


# =============================== FASE A: solo EE.UU. ==============================
us_factors, us_wide, us_fwd, us_p1, us_p3 = build("us")
assert data.load_factors().equals(us_factors), "load_factors() por defecto cambió"

tr = us_p3[(us_p3.date <= v3.train_cutoff(1996, 3)) & us_p3.label.notna()]
va = us_p3[(us_p3.date >= "1996-01-31") & (us_p3.date <= "1999-12-31")]
probs3_tune = v3.fit_predict(tr, va, PARAMS)                       # 1996-1999
us_probs3 = v3.walk_forward(us_p3, PARAMS, 2000, embargo_months=3)  # 2000+
probs3_valid = pd.concat([probs3_tune, us_probs3.loc[:VALID[1]]])
assert tr.date.max() <= pd.Timestamp("1995-09-30")

# Asserts de mecánica (AC-2)
ew_w = v3.weights_tilt(probs3_valid, 0.0)
assert np.allclose(ew_w.to_numpy(), 1 / 6), "lam=0 debe dar equal-weight"
w_chk = v3.weights_v3(probs3_valid, us_wide, 3.0, 0.5)
assert (w_chk.to_numpy() >= -1e-12).all() and np.allclose(w_chk.sum(axis=1), 1)
to = bt.portfolio_returns(w_chk, us_fwd)["turnover"]
assert (to[~to.index.month.isin(v3.PHASES["cal"])].iloc[1:] < 1e-12).all(), "rota fuera de rebalanceo"
s_one = v3.scale_to_ew(v3.weights_tilt(probs3_valid, 3.0), v3.stress_flag(us_wide), 1.0)
assert np.allclose(s_one.to_numpy(), v3.weights_tilt(probs3_valid, 3.0).to_numpy())

rows = []
for lam, kappa in product(v3.LAMBDAS, v3.KAPPAS):
    sh = []
    for months in v3.PHASES.values():
        w = v3.weights_v3(probs3_valid, us_wide, lam, kappa, months)
        r = bt.portfolio_returns(w, us_fwd, COST)["net"].loc[VALID[0]:VALID[1]]
        sh.append(sharpe(r))
    rows.append({"lam": lam, "kappa": kappa, "sharpe_valid": float(np.mean(sh)),
                 "sharpe_fases": [round(x, 3) for x in sh]})
grid = pd.DataFrame(rows)
best = grid["sharpe_valid"].max()
cand = grid[grid["sharpe_valid"] >= best - 0.01].sort_values(["lam", "kappa"], ascending=[True, False])
LAM, KAPPA = float(cand.iloc[0]["lam"]), float(cand.iloc[0]["kappa"])
print(grid.pivot(index="lam", columns="kappa", values="sharpe_valid").round(3).to_string(), flush=True)
print(f"PARÁMETROS CONGELADOS: lam={LAM}, kappa={KAPPA}", flush=True)
PARAMS_FILE.write_text(json.dumps({"lam": LAM, "kappa": KAPPA, "grid": rows}, indent=2))
print(f"escrito {PARAMS_FILE.name} — a partir de aquí se cargan datos internacionales", flush=True)

# ============================ FASE B: evaluación =================================
frozen = json.loads(PARAMS_FILE.read_text())
LAM, KAPPA = frozen["lam"], frozen["kappa"]

REGIONS = {
    "dev_ex_us": {"first_year": 2001, "eval": ("2001-03-31", "2026-08-31"),
                  "sub": {"2001-2009": ("2001-03-31", "2009-12-31"),
                          "2010-2019": ("2010-01-31", "2019-12-31"),
                          "2020-2026": ("2020-01-31", "2026-08-31")}},
    "us": {"first_year": 2000, "eval": ("2005-01-31", "2026-08-31"),
           "sub": {"2005-2009": ("2005-01-31", "2009-12-31"),
                   "2010-2019": ("2010-01-31", "2019-12-31"),
                   "2020-2026": ("2020-01-31", "2026-08-31")}},
}

summary = {"lam": LAM, "kappa": KAPPA, "criterio":
           "Sharpe(v3) > Sharpe(top-2 mensual) a 10 pb en Developed ex-US 2001-03..2026-08"}
all_metrics, series = [], {}
for region, cfg in REGIONS.items():
    if region == "us":
        factors, wide, fwd, p1, p3, probs3 = us_factors, us_wide, us_fwd, us_p1, us_p3, us_probs3
    else:
        factors, wide, fwd, p1, p3 = build(region)
        probs3 = v3.walk_forward(p3, PARAMS, cfg["first_year"], embargo_months=3)
    probs1, _ = md.walk_forward_primary(p1, "mlp", PARAMS, first_year=cfg["first_year"])
    stress = v3.stress_flag(wide)
    idx = probs1.index
    w = {
        "ew": bt.weights_equal(idx),
        "mom12": bt.weights_momentum(wide, idx),
        "primary": md.top_k_picks(probs1),
        "v3": v3.weights_v3(probs3, wide, LAM, KAPPA),
        "abl_trimestral": v3.hold_periodic(md.top_k_picks(probs3.dropna()), v3.PHASES["cal"]),
        "abl_tilt": v3.weights_tilt(probs1, LAM),
        "abl_estres": v3.scale_to_ew(md.top_k_picks(probs1), stress, KAPPA),
    }
    a, b = cfg["eval"]
    rets10 = {}
    for cname, c in COSTS.items():
        rets = {k: bt.portfolio_returns(v, fwd, c) for k, v in w.items()}
        for k, r in rets.items():
            for per, (x, y) in {"total": (a, b), **cfg["sub"]}.items():
                m = bt.metrics(r["net"].loc[x:y], rets["ew"]["net"].loc[x:y], r["turnover"].loc[x:y])
                all_metrics.append({"region": region, "estrategia": k, "costo": cname,
                                    "periodo": per, **m})
        if cname == "10pb":
            rets10 = rets
    ev = {k: r.loc[a:b] for k, r in rets10.items()}
    s = {k: sharpe(r["net"]) for k, r in ev.items()}
    if region == "us":
        assert abs(s["primary"] - 0.440) < 0.001, f"principal EE.UU. no reproduce poc-01: {s['primary']}"
    summary[region] = {
        "ventana": [a, b], "sharpe_10pb": s,
        "bootstrap_v3_menos_primary": bt.block_bootstrap_sharpe_diff(ev["v3"]["net"], ev["primary"]["net"]),
        "bootstrap_v3_menos_ew": bt.block_bootstrap_sharpe_diff(ev["v3"]["net"], ev["ew"]["net"]),
        "rotacion_mensual": {k: float(r["turnover"].mean()) for k, r in ev.items()},
        "exceso_bruto_mensual_vs_ew": {k: float((r["gross"] - ev["ew"]["gross"]).mean()) for k, r in ev.items()},
        "meses_estres": int(stress.loc[a:b].sum()), "meses": int(len(ev["v3"])),
        "peso_max_promedio_v3": float(w["v3"].loc[a:b].max(axis=1).mean()),
        "factores_con_peso_promedio_v3": float((w["v3"].loc[a:b] > 0.01).sum(axis=1).mean()),
        "pesos_v3_promedio": w["v3"].loc[a:b].mean().round(3).to_dict(),
        "pesos_v3_vigentes": {"mes": str(w["v3"].index.max().date()),
                              **w["v3"].iloc[-1].round(3).to_dict()},
    }
    if region == "dev_ex_us":
        summary["veredicto"] = "FUNCIONA" if s["v3"] > s["primary"] else "NO FUNCIONA"
        print(f"\n*** VEREDICTO poc-03 (Developed ex-US): {summary['veredicto']} "
              f"— Sharpe v3 {s['v3']:.3f} vs top-2 mensual {s['primary']:.3f} ***", flush=True)
    for k in ("v3", "primary", "ew"):
        series[(region, k)] = rets10[k]["net"]
    w["v3"].assign(region=region).to_parquet(DATA_INTERIM / f"timing_v3_weights_{region}.parquet")

metrics = pd.DataFrame(all_metrics)
metrics.to_parquet(DATA_INTERIM / "timing_v3_metrics.parquet", index=False)
pd.DataFrame({f"{r}__{k}": v for (r, k), v in series.items()}).to_parquet(
    DATA_INTERIM / "timing_v3_returns.parquet")
(DATA_INTERIM / "timing_v3_summary.json").write_text(json.dumps(summary, indent=2, default=str))

pd.set_option("display.width", 180)
for region in REGIONS:
    m = metrics[(metrics.region == region)]
    print(f"\n===== {region} — total, 10 pb =====")
    print(m[(m.costo == "10pb") & (m.periodo == "total")].set_index("estrategia")[
        ["retorno_anual", "volatilidad", "sharpe", "max_drawdown", "rotacion_mensual"]].round(3).to_string())
    print(m[m.periodo == "total"].pivot(index="estrategia", columns="costo", values="sharpe").round(3).to_string())
    print(m[m.costo == "10pb"].pivot(index="estrategia", columns="periodo", values="sharpe").round(3).to_string())
    print(json.dumps({k: summary[region][k] for k in ("bootstrap_v3_menos_primary", "bootstrap_v3_menos_ew",
          "meses_estres", "meses", "peso_max_promedio_v3", "factores_con_peso_promedio_v3",
          "pesos_v3_promedio", "pesos_v3_vigentes", "exceso_bruto_mensual_vs_ew")}, indent=1))
