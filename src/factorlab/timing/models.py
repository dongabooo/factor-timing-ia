"""Modelo principal (ranking de factores) y modelo de confiabilidad (meta-labeling).

Todo se evalúa en walk-forward con ventana expansiva y reentrenamiento cada enero.
Regla de embargo: la etiqueta del mes t depende del retorno de t+1, así que el modelo
que predice el año Y solo puede entrenar con meses t <= noviembre de Y-1 (cuyo t+1
cae a más tardar en diciembre de Y-1).

Las decisiones de diseño (hiperparámetros, variante de modelo) se congelan con datos
ANTERIORES al periodo de evaluación. Nada se elige mirando 2000+ (modelo principal)
ni 2005+ (gate).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from factorlab.timing.data import FACTORS
from factorlab.timing.features import MACRO_FEATURES, model_columns

SEED = 20261112
TOP_K = 2

PRIMARY_OOS_START = 2000   # primer año fuera de muestra del modelo principal
META_OOS_START = 2005      # primer año de evaluación del gate
TUNE_TRAIN_END = "1995-11-30"   # ajuste de hiperparámetros del principal:
TUNE_VALID = ("1996-01-31", "1999-12-31")  # entrena 1986-1995, valida 1996-1999


# --- Fábricas de modelos ------------------------------------------------------

def make_hgb(max_depth=3, learning_rate=0.05, max_iter=150, l2=1.0):
    return HistGradientBoostingClassifier(
        max_depth=max_depth, learning_rate=learning_rate, max_iter=max_iter,
        l2_regularization=l2, min_samples_leaf=40, random_state=SEED,
    )


def make_mlp(hidden=(16,), alpha=1e-2):
    # Imputación y escalado DENTRO del pipeline: se ajustan solo con la ventana de
    # entrenamiento de cada año, nunca con estadísticas de toda la muestra.
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=600,
                      early_stopping=True, validation_fraction=0.2,
                      n_iter_no_change=20, random_state=SEED),
    )


def make_logit(C=0.5):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000))


PRIMARY_GRIDS = {
    "hgb": [dict(max_depth=d, learning_rate=lr, max_iter=it)
            for d, lr, it in product([2, 3], [0.03, 0.08], [100, 250])],
    "mlp": [dict(hidden=h, alpha=a)
            for h, a in product([(8,), (16,), (32, 16)], [1e-3, 1e-2, 1e-1])],
}
PRIMARY_FACTORIES = {"hgb": make_hgb, "mlp": make_mlp}


# --- Utilidades ---------------------------------------------------------------

def _train_mask(dates: pd.Series, year: int) -> pd.Series:
    """Meses cuya etiqueta ya se conocía al iniciar `year` (embargo de 1 mes)."""
    return dates <= pd.Timestamp(year - 1, 11, 30)


def top_k_picks(probs: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    """Pesos equal-weight en los k factores con mayor probabilidad, por mes."""
    ranks = probs.rank(axis=1, ascending=False, method="first")
    return (ranks <= k).astype(float) / k


def selection_excess(probs: pd.DataFrame, fwd: pd.DataFrame) -> float:
    """Retorno promedio de la elección top-k menos el equal-weight (criterio de ajuste)."""
    w = top_k_picks(probs)
    fwd = fwd.reindex(probs.index)
    return float(((w * fwd).sum(axis=1) - fwd.mean(axis=1)).mean())


def _to_wide_probs(panel: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    return (panel.assign(p=p).pivot(index="date", columns="factor", values="p")[FACTORS])


# --- Modelo principal ---------------------------------------------------------

def tune_primary(panel: pd.DataFrame, fwd: pd.DataFrame, kind: str) -> tuple[dict, pd.DataFrame]:
    """Elige hiperparámetros con 1986-1995 -> 1996-1999. Devuelve (mejores, tabla)."""
    cols = model_columns(panel)
    tr = panel[(panel.date <= TUNE_TRAIN_END) & panel.label.notna()]
    va = panel[(panel.date >= TUNE_VALID[0]) & (panel.date <= TUNE_VALID[1])]
    rows = []
    for params in PRIMARY_GRIDS[kind]:
        model = PRIMARY_FACTORIES[kind](**params).fit(tr[cols], tr["label"])
        probs = _to_wide_probs(va, model.predict_proba(va[cols])[:, 1])
        rows.append({**{k: str(v) for k, v in params.items()},
                     "valid_excess_mensual": selection_excess(probs, fwd)})
    table = pd.DataFrame(rows).sort_values("valid_excess_mensual", ascending=False)
    best = PRIMARY_GRIDS[kind][int(table.index[0])]
    return best, table


def walk_forward_primary(panel: pd.DataFrame, kind: str, params: dict,
                         first_year: int = PRIMARY_OOS_START) -> tuple[pd.DataFrame, list]:
    """Probabilidades fuera de muestra por (mes, factor), reentrenando cada enero.

    Returns:
        (probs ancho mes x factor, bitácora de entrenamiento por año)
    """
    cols = model_columns(panel)
    last_year = panel.date.max().year
    out, log = [], []
    for year in range(first_year, last_year + 1):
        train = panel[_train_mask(panel.date, year) & panel.label.notna()]
        test = panel[panel.date.dt.year == year]
        if test.empty:
            continue
        model = PRIMARY_FACTORIES[kind](**params).fit(train[cols], train["label"])
        out.append(test.assign(p=model.predict_proba(test[cols])[:, 1]))
        log.append({"year": year, "train_last_month": train.date.max(),
                    "n_train": len(train), "n_test": len(test)})
    res = pd.concat(out)
    return res.pivot(index="date", columns="factor", values="p")[FACTORS], log


# --- Gate de confiabilidad (meta-labeling) ------------------------------------

def meta_frame(probs: pd.DataFrame, fwd: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """Una fila por mes con los insumos del meta-modelo y su etiqueta.

    meta_label = 1 si la elección top-2 del principal superó al equal-weight en t+1.
    El hit rate reciente del principal usa SOLO meses ya cerrados: el resultado del
    mes s se conoce en s+1, así que en t se promedian los resultados hasta t-1.
    """
    fwd = fwd.reindex(probs.index)
    w = top_k_picks(probs)
    pick_ret = (w * fwd).sum(axis=1).where(fwd.notna().all(axis=1))
    ew_ret = fwd.mean(axis=1).where(fwd.notna().all(axis=1))
    label = (pick_ret > ew_ret).astype(float).where(pick_ret.notna())

    srt = np.sort(probs.to_numpy(), axis=1)[:, ::-1]
    df = pd.DataFrame(index=probs.index)
    df["p_top1"] = srt[:, 0]
    df["p_top2"] = srt[:, 1]
    df["gap_top2_vs_3"] = srt[:, :2].mean(axis=1) - srt[:, 2]
    df["p_dispersion"] = srt.std(axis=1)
    df["primary_hit_12m"] = label.shift(1).rolling(12, min_periods=6).mean()
    df = df.join(wide[MACRO_FEATURES])
    df["pick_ret"] = pick_ret
    df["ew_ret"] = ew_ret
    df["meta_label"] = label
    return df


META_FEATURES = ["p_top1", "p_top2", "gap_top2_vs_3", "p_dispersion",
                 "primary_hit_12m"] + MACRO_FEATURES
META_FACTORIES = {
    "logit": lambda: make_logit(C=0.5),
    "hgb": lambda: make_hgb(max_depth=2, learning_rate=0.05, max_iter=100, l2=5.0),
}
THRESHOLDS = np.round(np.arange(0.30, 0.701, 0.025), 3)


def _brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def choose_meta_kind(meta: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Elige la variante del meta-modelo con datos ANTERIORES a la evaluación.

    Entrena con 2000-2002 y valida en 2003-2004 (Brier score); ambos periodos son
    predicciones fuera de muestra del principal y anteriores a 2005.
    """
    tr = meta[(meta.index <= "2002-11-30") & meta.meta_label.notna()]
    va = meta[(meta.index >= "2003-01-31") & (meta.index <= "2004-12-31")
              & meta.meta_label.notna()]
    rows = []
    for kind, factory in META_FACTORIES.items():
        m = factory().fit(tr[META_FEATURES], tr["meta_label"])
        rows.append({"kind": kind,
                     "brier_valid": _brier(va["meta_label"],
                                           m.predict_proba(va[META_FEATURES])[:, 1])})
    table = pd.DataFrame(rows).sort_values("brier_valid")
    return str(table.iloc[0]["kind"]), table


def _gated_sharpe(p_meta, pick_ret, ew_ret, thr):
    r = np.where(p_meta >= thr, pick_ret, ew_ret)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else -np.inf


def choose_threshold(train: pd.DataFrame, kind: str, holdout_months: int = 36) -> float:
    """Umbral de abstención elegido SOLO con la ventana de entrenamiento.

    Se reserva el final de la ventana (36 meses, o 40% si es más corta), se ajusta el
    meta-modelo con el resto, y en ese tramo reservado se elige el umbral que maximiza
    el Sharpe del portafolio con gate. Si ningún umbral supera a "nunca abstenerse",
    el umbral es 0: por defecto el gate NO interviene.
    """
    n = len(train)
    h = min(holdout_months, max(12, int(n * 0.4)))
    fit, hold = train.iloc[:-h], train.iloc[-h:]
    if fit.meta_label.nunique() < 2:
        return 0.0
    m = META_FACTORIES[kind]().fit(fit[META_FEATURES], fit["meta_label"])
    p = m.predict_proba(hold[META_FEATURES])[:, 1]
    base = _gated_sharpe(p, hold.pick_ret.to_numpy(), hold.ew_ret.to_numpy(), 0.0)
    best_thr, best = 0.0, base
    for thr in THRESHOLDS:
        s = _gated_sharpe(p, hold.pick_ret.to_numpy(), hold.ew_ret.to_numpy(), thr)
        if s > best + 1e-9:
            best_thr, best = float(thr), s
    return best_thr


def walk_forward_meta(meta: pd.DataFrame, kind: str,
                      first_year: int = META_OOS_START) -> tuple[pd.DataFrame, list]:
    """p_meta fuera de muestra y decisión de abstención, reentrenando cada enero.

    El meta-modelo solo ve predicciones FUERA de muestra del principal (meta arranca
    en 2000, que es el primer año OOS del principal), con el mismo embargo de 1 mes.
    """
    out, log = [], []
    for year in range(first_year, meta.index.max().year + 1):
        train = meta[(meta.index <= pd.Timestamp(year - 1, 11, 30)) & meta.meta_label.notna()]
        test = meta[meta.index.year == year]
        if test.empty:
            continue
        thr = choose_threshold(train, kind)
        m = META_FACTORIES[kind]().fit(train[META_FEATURES], train["meta_label"])
        p = m.predict_proba(test[META_FEATURES])[:, 1]
        out.append(pd.DataFrame({"p_meta": p, "threshold": thr,
                                 "abstain": p < thr}, index=test.index))
        log.append({"year": year, "train_first": train.index.min(),
                    "train_last": train.index.max(), "n_train": len(train),
                    "threshold": thr})
    return pd.concat(out), log
