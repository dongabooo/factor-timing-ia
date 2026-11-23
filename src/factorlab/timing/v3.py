"""Experimento v3 (plan poc-03): asignación trimestral con tilt y reducción en estrés.

Tres cambios sobre el modelo de poc-01, propuestos por el autor desde la práctica de
índices de factores:

1. Rebalanceo trimestral: el modelo predice el retorno de los 3 meses siguientes y la
   cartera se mantiene fija entre rebalanceos.
2. Tilt desde equal-weight: w_i = max(0, 1/6 + lam * (p_i - media(p))), renormalizado.
3. Menos apuesta en estrés: w = EW + s_t * (w_tilt - EW), con s_t = kappa si el
   mercado viene cayendo (retorno 12m < 0) con volatilidad alta (vol. realizada 3m
   sobre su mediana expansiva), y s_t = 1 en otro caso.

Nada de aquí modifica el motor de poc-01: reutiliza sus fábricas de modelos y
features, y parametriza solo el embargo del walk-forward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.timing.data import FACTORS
from factorlab.timing.features import model_columns
from factorlab.timing.models import make_mlp

#: Fases de rebalanceo trimestral (meses de decisión). La primera es la de calendario.
PHASES = {"cal": (3, 6, 9, 12), "ene": (1, 4, 7, 10), "feb": (2, 5, 8, 11)}
LAMBDAS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
KAPPAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def targets_3m(factors: pd.DataFrame) -> pd.DataFrame:
    """Retorno compuesto de t+1 a t+3 de cada factor, indexado en t."""
    g = 1 + factors[FACTORS]
    return (g.shift(-1) * g.shift(-2) * g.shift(-3)) - 1


def train_cutoff(year: int, embargo_months: int) -> pd.Timestamp:
    """Último mes t cuya etiqueta (que depende de t+embargo) se conocía al iniciar `year`."""
    return pd.Timestamp(year - 1, 12, 31) - pd.DateOffset(months=embargo_months) + pd.offsets.MonthEnd(0)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, params: dict) -> pd.DataFrame:
    cols = model_columns(train)
    model = make_mlp(**params).fit(train[cols], train["label"])
    out = test.assign(p=model.predict_proba(test[cols])[:, 1])
    return out.pivot(index="date", columns="factor", values="p")[FACTORS]


def walk_forward(panel: pd.DataFrame, params: dict, first_year: int,
                 embargo_months: int) -> pd.DataFrame:
    """Probabilidades OOS por (mes, factor), reentrenando cada enero con embargo dado."""
    out = []
    for year in range(first_year, panel.date.max().year + 1):
        train = panel[(panel.date <= train_cutoff(year, embargo_months)) & panel.label.notna()]
        test = panel[panel.date.dt.year == year]
        if not test.empty:
            out.append(fit_predict(train, test, params))
    return pd.concat(out)


def weights_tilt(probs: pd.DataFrame, lam: float) -> pd.DataFrame:
    """Tilt long-only desde equal-weight según la confianza relativa del modelo."""
    dev = probs.sub(probs.mean(axis=1), axis=0)
    w = (1.0 / len(FACTORS) + lam * dev).clip(lower=0)
    return w.div(w.sum(axis=1), axis=0)


def stress_flag(wide: pd.DataFrame) -> pd.Series:
    """True si el mercado viene de caer (12m < 0) con volatilidad 3m sobre su mediana.

    La mediana es expansiva (solo usa el pasado) y exige 36 meses de historia.
    """
    rv = wide["mkt_rvol_3m"]
    med = rv.expanding(min_periods=36).median()
    return ((wide["Mkt-RF__ret_12m"] < 0) & (rv > med)).fillna(False)


def scale_to_ew(w: pd.DataFrame, stress: pd.Series, kappa: float) -> pd.DataFrame:
    """w -> EW + s_t (w - EW): en estrés la apuesta activa se reduce a kappa."""
    s = pd.Series(np.where(stress.reindex(w.index).fillna(False), kappa, 1.0), index=w.index)
    ew = 1.0 / len(FACTORS)
    return (w - ew).mul(s, axis=0) + ew


def hold_periodic(target: pd.DataFrame, months: tuple[int, ...]) -> pd.DataFrame:
    """Fija los pesos en los meses de decisión y los mantiene hasta el siguiente.

    Las filas anteriores a la primera decisión se descartan.
    """
    mask = target.index.month.isin(months)
    return target[mask].reindex(target.index).ffill().dropna(how="any")


def weights_v3(probs3: pd.DataFrame, wide: pd.DataFrame, lam: float, kappa: float,
               months: tuple[int, ...] = PHASES["cal"]) -> pd.DataFrame:
    target = scale_to_ew(weights_tilt(probs3.dropna(), lam), stress_flag(wide), kappa)
    return hold_periodic(target, months)
