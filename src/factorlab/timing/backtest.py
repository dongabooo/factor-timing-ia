"""Backtest de las 4 estrategias, métricas, costos y bootstrap.

Convención: los pesos del mes t se aplican al retorno de t+1. Los factores de Ken
French ya son retornos en exceso (long-short, o mercado menos RF), así que el Sharpe se
calcula directamente sobre ellos sin restar la tasa libre de riesgo.

Costo: `c` por unidad de rotación, con rotación = suma de |cambio de pesos|. Cambiar
por completo de un par de factores a otro cuesta 2c. El equal-weight se trata como
rebalanceo sin costo (simplificación declarada en el reporte).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.timing.data import FACTORS
from factorlab.timing.models import top_k_picks

STRATEGIES = {
    "ew": "Equal-weight (siempre invertido)",
    "mom12": "Top-2 por momentum 12m (rotación ingenua)",
    "primary": "Modelo principal",
    "gated": "Modelo principal + gate",
}


def weights_equal(index) -> pd.DataFrame:
    return pd.DataFrame(1.0 / len(FACTORS), index=index, columns=FACTORS)


def weights_momentum(wide: pd.DataFrame, index) -> pd.DataFrame:
    trailing = wide[[f"{f}__ret_12m" for f in FACTORS]].reindex(index)
    trailing.columns = FACTORS
    return top_k_picks(trailing)


def weights_gated(primary_w: pd.DataFrame, abstain: pd.Series) -> pd.DataFrame:
    ew = weights_equal(primary_w.index)
    mask = abstain.reindex(primary_w.index).fillna(False).astype(bool)
    return primary_w.where(~mask, ew, axis=0)


def portfolio_returns(w: pd.DataFrame, fwd: pd.DataFrame, cost: float = 0.0) -> pd.DataFrame:
    """Retorno bruto, rotación y retorno neto, indexados en el mes de DECISIÓN t."""
    fwd = fwd.reindex(w.index)[FACTORS]
    gross = (w * fwd).sum(axis=1).where(fwd.notna().all(axis=1))
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = 0.0
    return pd.DataFrame({"gross": gross, "turnover": turnover,
                         "net": gross - cost * turnover})


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r.dropna()).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def metrics(r: pd.Series, ew: pd.Series, turnover: pd.Series) -> dict:
    r = r.dropna()
    ew = ew.reindex(r.index)
    n = len(r)
    cagr = float((1 + r).prod() ** (12 / n) - 1)
    vol = float(r.std(ddof=1) * np.sqrt(12))
    return {
        "meses": n,
        "retorno_anual": cagr,
        "volatilidad": vol,
        "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(12)),
        "max_drawdown": max_drawdown(r),
        "hit_vs_ew": float((r > ew).mean()),
        "rotacion_mensual": float(turnover.reindex(r.index).mean()),
    }


def _sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return x.mean() / sd * np.sqrt(12) if sd > 0 else np.nan


def block_bootstrap_sharpe_diff(a: pd.Series, b: pd.Series, block: int = 12,
                                reps: int = 2000, seed: int = 20261112,
                                ci: float = 0.90) -> dict:
    """IC de Sharpe(a) - Sharpe(b) por bootstrap en bloques móviles, pareado.

    Los bloques preservan la autocorrelación mensual; remuestrear a y b con los MISMOS
    índices preserva su correlación (las dos estrategias comparten casi todo el riesgo).
    """
    df = pd.concat([a, b], axis=1).dropna().to_numpy()
    n = len(df)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(reps)
    for i in range(reps):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        s = df[idx]
        diffs[i] = _sharpe(s[:, 0]) - _sharpe(s[:, 1])
    lo, hi = np.quantile(diffs, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    point = _sharpe(df[:, 0]) - _sharpe(df[:, 1])
    return {"diff": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "p_diff_le_0": float((diffs <= 0).mean())}


def weights_hysteresis(probs: pd.DataFrame, delta: float, k: int = 2) -> pd.DataFrame:
    """Top-k con freno de rotación (banda de histéresis, plan poc-02).

    El portafolio guarda sus k factores. Cada mes, mientras el mejor factor de afuera
    supere en probabilidad al más débil de adentro por MÁS de `delta`, se
    intercambian (a lo sumo k cambios por mes). Con delta = 0 coincide con
    top_k_picks; con delta >= 1 el portafolio nunca cambia tras el primer mes.
    Un mes con probabilidades faltantes conserva el portafolio previo.
    """
    cols = list(probs.columns)
    out = np.zeros((len(probs), len(cols)))
    held: list[str] | None = None
    for i, (_, row) in enumerate(probs.iterrows()):
        if row.isna().any():
            if held is None:
                continue
        elif held is None:
            held = list(row.sort_values(ascending=False, kind="stable").index[:k])
        else:
            for _ in range(k):
                inside = row[held]
                outside = row.drop(held)
                weakest, best = inside.idxmin(), outside.idxmax()
                if outside[best] - inside[weakest] > delta:
                    held[held.index(weakest)] = best
                else:
                    break
        for f in held:
            out[i, cols.index(f)] = 1.0 / k
    return pd.DataFrame(out, index=probs.index, columns=cols)
