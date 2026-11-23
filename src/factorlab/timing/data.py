"""Descarga y cacheo de los datos del PoC: factores de Ken French y series de FRED.

Los factores de Ken French se construyen con CRSP, así que no cargan el sesgo de
supervivencia que limita a la Fase 1. Las series de FRED se bajan en su vintage
ACTUAL, no histórico: las revisiones de UNRATE y CPI introducen un sesgo menor que se
declara en el reporte (corregirlo exigiría ALFRED, fuera de alcance del PoC).
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests

from factorlab.config import DATA_RAW

TIMING_RAW = DATA_RAW / "timing"
TIMING_RAW.mkdir(parents=True, exist_ok=True)

_FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {
    "ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "mom": "F-F_Momentum_Factor_CSV.zip",
}
#: Archivos por región (poc-03). "us" es el de poc-01; "dev_ex_us" es el examen
#: fuera de muestra del v3: mercados desarrollados sin EE.UU., desde 1990.
REGION_FILES = {
    "us": FRENCH_FILES,
    "dev_ex_us": {"ff5": "Developed_ex_US_5_Factors_CSV.zip",
                  "mom": "Developed_ex_US_Mom_Factor_CSV.zip"},
}

#: Los 6 factores sobre los que el PoC rota. RF se conserva aparte.
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]

#: serie FRED -> descripción. Diarias se llevan a fin de mes con el último dato.
FRED_SERIES = {
    "T10Y3M": "Pendiente de la curva 10 años - 3 meses (diaria, desde 1982)",
    "BAA10Y": "Spread de crédito Baa - Tesoro 10 años (diaria, desde 1986)",
    "VIXCLS": "VIX (diaria, desde 1990)",
    "UNRATE": "Tasa de desempleo (mensual, se publica con rezago)",
    "CPIAUCSL": "Índice de precios al consumidor (mensual, se publica con rezago)",
}

_TIMEOUT = 60


def _download(url: str, dest) -> bytes:
    """Devuelve el contenido cacheado; descarga solo si no existe en disco."""
    if dest.exists():
        return dest.read_bytes()
    resp = requests.get(url, timeout=_TIMEOUT)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  descargado: {dest.name} ({len(resp.content):,} bytes)", flush=True)
    return resp.content


def _parse_french_monthly(raw_zip: bytes) -> pd.DataFrame:
    """Extrae SOLO la tabla mensual de un CSV de Ken French.

    El archivo trae primero la tabla mensual (fechas AAAAMM, 6 dígitos) y después una
    tabla anual (fechas AAAA, 4 dígitos) con los mismos encabezados. Leer el CSV entero
    mezclaría ambas; por eso se filtra por la longitud de la fecha.
    """
    z = zipfile.ZipFile(io.BytesIO(raw_zip))
    text = z.read(z.namelist()[0]).decode("latin1")
    lines = text.splitlines()

    header = None
    rows = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if header is None and len(parts) > 1 and parts[0] == "" and parts[1]:
            header = ["date"] + parts[1:]
            continue
        if header is not None and len(parts[0]) == 6 and parts[0].isdigit():
            rows.append(parts[: len(header)])
        elif header is not None and rows and not (len(parts[0]) == 6 and parts[0].isdigit()):
            break  # terminó la tabla mensual

    df = pd.DataFrame(rows, columns=header)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.set_index("date").apply(pd.to_numeric, errors="coerce")
    df.columns = [c.strip() for c in df.columns]
    return df / 100.0  # de porcentaje a decimal


def load_factors(region: str = "us") -> pd.DataFrame:
    """Retornos mensuales de los 6 factores + RF, en decimales, índice a fin de mes.

    Los archivos internacionales marcan faltantes con -99.99 (en porcentaje); se
    convierten a NaN y esas filas se descartan.
    """
    files = REGION_FILES[region]
    ff5 = _parse_french_monthly(_download(_FRENCH_BASE + files["ff5"], TIMING_RAW / files["ff5"]))
    mom = _parse_french_monthly(_download(_FRENCH_BASE + files["mom"], TIMING_RAW / files["mom"]))
    mom = mom.rename(columns={mom.columns[0]: "Mom"})
    out = ff5.join(mom[["Mom"]], how="inner")
    out = out.mask(out <= -0.9998)  # -99.99% = dato faltante
    return out[FACTORS + ["RF"]].dropna()


def load_fred() -> pd.DataFrame:
    """Series de FRED llevadas a fin de mes (último dato disponible de cada mes).

    No aplica el rezago de publicación: eso lo hace features.py, que es donde se
    decide qué es "conocible" al cierre de cada mes.
    """
    frames = {}
    for sid in FRED_SERIES:
        dest = TIMING_RAW / f"fred_{sid}.csv"
        raw = _download(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", dest)
        s = pd.read_csv(io.BytesIO(raw))
        s.columns = ["date", sid]
        s["date"] = pd.to_datetime(s["date"])
        s[sid] = pd.to_numeric(s[sid], errors="coerce")  # FRED marca faltantes con "."
        s = s.dropna().set_index("date")[sid]
        frames[sid] = s.resample("ME").last()
    return pd.DataFrame(frames)
