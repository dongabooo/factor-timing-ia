"""Features del PoC: todo lo que se sabe al CIERRE del mes t, y nada más.

Regla de no-fuga: un feature del mes t solo puede usar información publicada a más
tardar al cierre de t. Tres consecuencias concretas:

- Las ventanas móviles (retornos acumulados, volatilidad, drawdown) son causales por
  construcción: solo miran hacia atrás.
- UNRATE y CPI de un mes se publican a inicios del mes siguiente. Por eso se desplazan
  1 mes: el desempleo de mayo solo es usable al cierre de junio.
- Las series diarias (curva, crédito, VIX) se toman en su último dato del mes, que sí
  es conocido al cierre de ese mes.

El objetivo del mes t es el retorno de cada factor en t+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.timing.data import FACTORS

#: Primer mes de la muestra de modelado: BAA10Y empieza en 1986-01.
SAMPLE_START = "1986-01-31"

#: Features de mercado/macro que entran a los modelos. El VIX queda FUERA: empieza en
#: 1990 y dejaría hoyos en 1986-1989; la volatilidad realizada del mercado lo sustituye.
MACRO_FEATURES = [
    "T10Y3M", "BAA10Y", "T10Y3M_d3", "BAA10Y_d3",
    "mkt_rvol_3m", "unrate_d3_lag1", "cpi_yoy_lag1",
]

#: Features propios de cada factor (se calculan para los 6).
OWN_FEATURES = ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_12m", "dd_12m"]


def _compound(r: pd.Series, n: int) -> pd.Series:
    return (1 + r).rolling(n, min_periods=n).apply(np.prod, raw=True) - 1


def factor_features(factors: pd.DataFrame) -> pd.DataFrame:
    """Features por factor en formato ancho: columnas '<factor>__<feature>'."""
    cols = {}
    for f in FACTORS:
        r = factors[f]
        cols[f"{f}__ret_1m"] = r
        cols[f"{f}__ret_3m"] = _compound(r, 3)
        cols[f"{f}__ret_6m"] = _compound(r, 6)
        cols[f"{f}__ret_12m"] = _compound(r, 12)
        cols[f"{f}__vol_12m"] = r.rolling(12, min_periods=12).std() * np.sqrt(12)
        wealth = (1 + r).cumprod()
        cols[f"{f}__dd_12m"] = wealth / wealth.rolling(12, min_periods=12).max() - 1
    return pd.DataFrame(cols, index=factors.index)


def macro_features(factors: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
    """Features de mercado y macro, alineados al calendario de los factores."""
    fred = fred.reindex(factors.index)
    out = pd.DataFrame(index=factors.index)
    out["T10Y3M"] = fred["T10Y3M"]
    out["BAA10Y"] = fred["BAA10Y"]
    out["T10Y3M_d3"] = fred["T10Y3M"].diff(3)
    out["BAA10Y_d3"] = fred["BAA10Y"].diff(3)
    out["VIX"] = fred["VIXCLS"]  # solo para mostrar; no entra a los modelos
    out["mkt_rvol_3m"] = factors["Mkt-RF"].rolling(3, min_periods=3).std() * np.sqrt(12)
    # Rezago de publicación: el dato del mes m se conoce hasta el mes m+1.
    out["unrate_d3_lag1"] = fred["UNRATE"].diff(3).shift(1)
    out["cpi_yoy_lag1"] = fred["CPIAUCSL"].pct_change(12, fill_method=None).shift(1)
    return out


def build_wide(factors: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
    """Panel ancho por mes: features de los 6 factores + mercado/macro.

    Las filas anteriores a SAMPLE_START se calculan (alimentan las ventanas) pero se
    recortan al final.
    """
    wide = factor_features(factors).join(macro_features(factors, fred))
    return wide.loc[SAMPLE_START:]


def targets(factors: pd.DataFrame) -> pd.DataFrame:
    """Retorno de cada factor en t+1, indexado en t. La última fila queda NaN."""
    return factors[FACTORS].shift(-1)


def stack_panel(wide: pd.DataFrame, fwd: pd.DataFrame) -> pd.DataFrame:
    """Panel apilado (mes x factor) para el modelo de ranking.

    Por fila: los features propios del factor, su rango transversal entre los 6 factores
    (comparar factores entre sí es lo que importa para elegir), los features de
    mercado/macro, un one-hot del factor, el retorno de t+1 y la etiqueta:
    1 si ese retorno supera la mediana de los 6 factores en t+1.
    """
    own = {}
    for feat in OWN_FEATURES:
        block = wide[[f"{f}__{feat}" for f in FACTORS]]
        block.columns = FACTORS
        own[feat] = block
        own[f"{feat}_rank"] = block.rank(axis=1, pct=True)

    fwd = fwd.reindex(wide.index)
    median_next = fwd.median(axis=1)
    parts = []
    for f in FACTORS:
        df = pd.DataFrame({k: v[f] for k, v in own.items()}, index=wide.index)
        df = df.join(wide[MACRO_FEATURES])
        for g in FACTORS:
            df[f"is_{g}"] = float(f == g)
        df["factor"] = f
        df["fwd_ret"] = fwd[f]
        df["label"] = (fwd[f] > median_next).astype(float).where(fwd[f].notna())
        parts.append(df)

    panel = pd.concat(parts).rename_axis("date").reset_index()
    return panel.sort_values(["date", "factor"]).reset_index(drop=True)


def model_columns(panel: pd.DataFrame) -> list[str]:
    """Columnas que entran a los modelos (excluye llaves, objetivo y etiqueta)."""
    return [c for c in panel.columns if c not in ("date", "factor", "fwd_ret", "label")]


def no_leak_check(factors: pd.DataFrame, fred: pd.DataFrame, n: int = 20,
                  seed: int = 20261112) -> pd.DataFrame:
    """Recalcula los features truncando los datos crudos en el mes t y compara.

    Si un feature del mes t cambia al quitar la información posterior a t, ese feature
    está mirando al futuro. La versión truncada reproduce lo que de verdad se conocía al
    cierre de t: las series diarias hasta t, pero UNRATE y CPI solo hasta t-1, porque
    el dato del mes t se publica en t+1. Así, si se omitiera el rezago, el test falla.
    """
    full = build_wide(factors, fred)
    rng = np.random.default_rng(seed)
    months = rng.choice(full.index[24:-1], size=n, replace=False)
    rows = []
    for t in sorted(months):
        t = pd.Timestamp(t)
        known = fred.loc[:t].copy()
        known.loc[known.index > t - pd.offsets.MonthEnd(1), ["UNRATE", "CPIAUCSL"]] = np.nan
        trunc = build_wide(factors.loc[:t], known)
        a, b = full.loc[t], trunc.loc[t]
        same = np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                           equal_nan=True, rtol=0, atol=1e-12)
        rows.append({"month": t, "identical": bool(same)})
    return pd.DataFrame(rows)
