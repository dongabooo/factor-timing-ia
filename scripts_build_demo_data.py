"""Precalcula los datos de la demo (plan poc-04). No genera resultados nuevos.

Reutiliza el motor de poc-01 a poc-03 con sus parámetros congelados y guarda, por región
y estrategia, el retorno BRUTO y la rotación de cada mes. Así la app recalcula el
retorno neto para cualquier costo (neto = bruto − costo × rotación) sin entrenar nada.
Al final, los asserts verifican que se reproducen los Sharpe de los reportes.
"""
import json
import warnings

import numpy as np
import pandas as pd

from factorlab.config import DATA_INTERIM
from factorlab.timing import backtest as bt, data, features as ft, models as md, v3

warnings.filterwarnings("ignore")
OUT = DATA_INTERIM / "demo"
OUT.mkdir(exist_ok=True)

PARAMS = {"hidden": (32, 16), "alpha": 1e-3}
frozen = json.loads((DATA_INTERIM / "timing_v3_params.json").read_text())
LAM, KAPPA = frozen["lam"], frozen["kappa"]
WINDOWS = {"us": ("2005-01-31", "2026-08-31"), "dev_ex_us": ("2001-03-31", "2026-08-31")}
FIRST_YEAR = {"us": 2000, "dev_ex_us": 2001}
EXPECTED = {("us", "ew"): 0.550, ("us", "primary"): 0.440, ("us", "gated"): 0.361,
            ("us", "trimestral"): 0.614, ("us", "v3"): 0.550,
            ("dev_ex_us", "ew"): 1.434, ("dev_ex_us", "v3"): 1.434,
            ("dev_ex_us", "primary"): 1.128, ("dev_ex_us", "trimestral"): 0.987}


def sharpe(r):
    r = r.dropna()
    return float(r.mean() / r.std(ddof=1) * np.sqrt(12))


fred = data.load_fred()
today, ic_rows = {}, []
for region in ("us", "dev_ex_us"):
    factors = data.load_factors(region)
    factors.to_parquet(OUT / f"factors_{region}.parquet")
    wide = ft.build_wide(factors, fred)
    fwd = ft.targets(factors)
    t3 = v3.targets_3m(factors)
    p1 = ft.stack_panel(wide, fwd)
    p3 = ft.stack_panel(wide, t3)
    p1, p3 = (p[p["ret_12m"].notna()].reset_index(drop=True) for p in (p1, p3))
    fy = FIRST_YEAR[region]
    probs1, _ = md.walk_forward_primary(p1, "mlp", PARAMS, first_year=fy)
    probs3 = v3.walk_forward(p3, PARAMS, fy, embargo_months=3)
    idx = probs1.index
    w = {
        "ew": bt.weights_equal(idx),
        "mom12": bt.weights_momentum(wide, idx),
        "primary": md.top_k_picks(probs1),
        "trimestral": v3.hold_periodic(md.top_k_picks(probs3.dropna()), v3.PHASES["cal"]),
        "v3": v3.weights_v3(probs3, wide, LAM, KAPPA),
    }
    if region == "us":   # el gate se toma tal cual de poc-01 (índice 2005+)
        res = pd.read_parquet(DATA_INTERIM / "timing_results.parquet")
        pr = res[[f"prob_{f}" for f in ft.FACTORS]].set_axis(ft.FACTORS, axis=1)
        w["gated"] = bt.weights_gated(md.top_k_picks(pr), res["abstain"])
        w["primary_gate_base"] = md.top_k_picks(pr)
    a, b = WINDOWS[region]
    rows = []
    for k, wk in w.items():
        r = bt.portfolio_returns(wk, fwd).loc[a:b]
        rows.append(pd.DataFrame({"estrategia": k, "gross": r["gross"], "turnover": r["turnover"]}))
        if region == "us" and k == "gated":
            g = bt.portfolio_returns(wk, fwd, 0.001)["net"].loc[a:b]
            print("gate check", sharpe(g))
    strat = pd.concat(rows).rename_axis("date").reset_index()
    strat.to_parquet(OUT / f"strategies_{region}.parquet")
    for (rg, k), exp in EXPECTED.items():
        if rg != region:
            continue
        s = strat[strat.estrategia == k]
        got = sharpe(s["gross"] - 0.001 * s["turnover"])
        assert abs(got - exp) < 0.002, f"{region}/{k}: {got:.4f} vs {exp}"
        print(f"ok {region}/{k}: {got:.3f}")

    last = probs1.dropna().iloc[-1]
    today[region] = {"mes_decision": str(last.name.date()),
                     "probs": last.round(4).to_dict(),
                     "picks": list(last.sort_values(ascending=False).index[:2]),
                     "v3": w["v3"].iloc[-1].round(4).to_dict()}

    # IC del modelo trimestral en decisiones de fin de trimestre
    periods = [("2005", "2026-06", "EE.UU. 2005–2026 (ya visto)")] if region == "us" \
        else [("2001", "2026-06", "Desarrollados ex-EE.UU. 2001–2026 (nunca visto)")]
    if region == "us":
        tr = p3[(p3.date <= v3.train_cutoff(1996, 3)) & p3.label.notna()]
        va = p3[(p3.date >= "1996") & (p3.date < "2000")]
        probs3_cal = pd.concat([v3.fit_predict(tr, va, PARAMS), probs3])
        periods = [("1996", "2004", "EE.UU. 1996–2004 (calibración)")] + periods
    else:
        probs3_cal = probs3
    for x, y, label in periods:
        q = probs3_cal.loc[x:y]
        q = q[q.index.month.isin([3, 6, 9, 12])]
        T = t3.reindex(q.index)[ft.FACTORS]
        ex = (md.top_k_picks(q) * T).sum(1) - T.mean(1)
        ic = np.nanmean([q.loc[d].corr(T.loc[d], method="spearman") for d in q.index])
        ic_rows.append({"periodo": label, "trimestres": len(q), "ic": ic,
                        "acierto": float((ex > 0).mean()), "exceso_trim": float(ex.mean())})

    if region == "us":   # ventaja del modelo mensual por bloques contra el oráculo
        f = fwd.reindex(probs1.index)[ft.FACTORS].dropna()
        ex = (md.top_k_picks(probs1.loc[f.index]) * f).sum(1) - f.mean(1)
        srt = np.sort(f.values, 1)[:, ::-1]
        oracle = pd.Series(srt[:, :2].mean(1) - f.mean(1).values, index=f.index)
        blocks = []
        for name, x, y in [("2000–04", "2000", "2004"), ("2005–09", "2005", "2009"),
                           ("2010–14", "2010", "2014"), ("2015–19", "2015", "2019"),
                           ("2020–26", "2020", "2026")]:
            blocks.append({"bloque": name, "modelo": ex.loc[x:y].mean(), "oraculo": oracle.loc[x:y].mean()})
        pd.DataFrame(blocks).to_parquet(OUT / "blocks.parquet")

pd.DataFrame(ic_rows).to_parquet(OUT / "ic.parquet")
(OUT / "today.json").write_text(json.dumps(today, indent=2))
res = pd.read_parquet(DATA_INTERIM / "timing_results.parquet")
res[["p_meta", "threshold", "abstain", "pick_ret", "ew_ret", "meta_label", "picks"]].to_parquet(OUT / "gate_us.parquet")
print(json.dumps(today, indent=1))
print(pd.DataFrame(ic_rows).round(3).to_string())
