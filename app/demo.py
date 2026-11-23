"""Demo del PoC: ¿puede la IA elegir qué factores de inversión ganarán el próximo mes?

Ejecutar desde la raíz del proyecto:  streamlit run app/demo.py
Solo presenta resultados ya calculados (scripts_build_demo_data.py); no entrena nada.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score

DEMO = Path(__file__).resolve().parents[1] / "data" / "interim" / "demo"

st.set_page_config(page_title="IA y factores de inversión", layout="wide")

# Paleta categórica fija (orden validado; el color sigue a la entidad en todas las pestañas)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GRAY = "#a3a29c"
INK, INK2 = "#0b0b0b", "#52514e"

FACTOR_NAMES = {"Mkt-RF": "Mercado", "SMB": "Tamaño (chicas)", "HML": "Valor (baratas)",
                "RMW": "Rentabilidad", "CMA": "Inversión conservadora", "Mom": "Momentum"}
FACTOR_COLOR = dict(zip(FACTOR_NAMES, C))
STRATS = {  # clave -> (nombre, color)
    "ew": ("Repartir parejo (equal-weight) = v3", C[0]),
    "primary": ("IA: elige 2 cada mes", C[1]),
    "trimestral": ("IA: elige 2 cada trimestre", C[2]),
    "mom12": ("Regla simple: los 2 que más subieron en 12 meses", C[3]),
    "gated": ("IA + gate de confianza", C[4]),
}
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
         "octubre", "noviembre", "diciembre"]
REGIONS = {"us": "Estados Unidos (2005–2026)", "dev_ex_us": "Desarrollados sin EE.UU. (2001–2026) — examen nunca visto"}


def style(fig, height=430, ytitle=None, yfmt=None):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0), hovermode="x unified",
                      font=dict(color=INK2), yaxis_title=ytitle)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#ececea", tickformat=yfmt)
    return fig


@st.cache_data
def load(name):
    return pd.read_parquet(DEMO / name)


@st.cache_data
def load_json(name):
    return json.loads((DEMO / name).read_text())


def metrics(net, turnover):
    n = len(net)
    return {"Retorno anual": (1 + net).prod() ** (12 / n) - 1,
            "Volatilidad": net.std(ddof=1) * np.sqrt(12),
            "Sharpe": net.mean() / net.std(ddof=1) * np.sqrt(12),
            "Peor caída": ((1 + net).cumprod() / (1 + net).cumprod().cummax() - 1).min(),
            "Rotación mensual": turnover.mean()}


st.title("¿Puede la IA elegir qué factores de inversión ganarán?")
st.caption("Proof-of-concept · Datos: Ken French (factores) y FRED (macro) · Todo evaluado fuera de muestra, "
           "con reglas fijadas antes de ver los resultados.")

tabs = st.tabs(["0 · Cómo funciona", "1 · Los datos", "2 · La decisión de hoy", "3 · Backtest", "4 · El gate", "5 · Lo que aprendimos"])

# ---------------------------------------------------------------- 0. Cómo funciona
with tabs[0]:
    st.markdown("**El problema:** cada mes, a unos estilos de inversión (factores) les va mejor que a otros. "
                "Este proyecto construye un **agente** que decide cómo repartir el dinero entre 6 factores y lo "
                "evalúa con honestidad: sin mirar al futuro y con reglas de éxito fijadas antes de ver resultados.")
    st.subheader("¿Qué tipo de IA es? Los 4 enfoques de Russell y Norvig")
    a1, a2 = st.columns(2)
    with a1:
        with st.container(border=True):
            st.markdown("**Pensar como humano** · modelado cognitivo")
            st.caption("No modela cómo razona un inversionista.")
        with st.container(border=True):
            st.markdown("**Actuar como humano** · prueba de Turing")
            st.caption("No busca parecer una persona.")
    with a2:
        with st.container(border=True):
            st.markdown("**Pensar racionalmente** · leyes del pensamiento / lógica")
            st.caption("No deduce con reglas lógicas.")
        st.success("**Actuar racionalmente · agente racional** ← este proyecto  \n"
                   "Percibe su entorno y elige la acción que maximiza el desempeño esperado bajo "
                   "incertidumbre. El gate es el agente evaluando su propia confianza.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**El agente en formato PEAS**")
        st.table(pd.DataFrame({
            "Qué es": ["Medida de desempeño", "Entorno", "Actuadores", "Sensores"],
            "En este proyecto": [
                "Retorno por unidad de riesgo (Sharpe), neto de costos",
                "Mercado de factores: incierto, cambiante, parcialmente observable y con otros agentes "
                "(inversionistas) que explotan las mismas señales",
                "Los pesos del portafolio entre los 6 factores, cada mes o trimestre",
                "Retornos pasados de los factores y datos macro ya publicados (tasas, crédito, desempleo, "
                "inflación)",
            ]}).set_index("Qué es"))
    with c2:
        st.markdown("**Cómo decide**")
        st.graphviz_chart('''
        digraph {
          rankdir=TB; bgcolor="transparent";
          node [shape=box, style="rounded,filled", fillcolor="#f4f4f2", color="#c3c2b7",
                fontname="Helvetica", fontsize=11];
          d [label="Datos: Ken French + FRED\\n(solo lo ya publicado)"];
          f [label="Señales por factor y del mercado\\n(retornos, volatilidad, tasas, crédito)"];
          m [label="Red neuronal (MLP)\\nP(factor en la mitad ganadora)", fillcolor="#fde3d7", color="#eb6834"];
          p [label="Decisión: los 2 más probables"];
          g [label="Gate (meta-labeling)\\n¿confío en esta decisión?", fillcolor="#fbe4ee", color="#e87ba4"];
          w [label="Portafolio del mes"];
          d -> f -> m -> p -> g;
          g -> w [label=" sí: los 2 elegidos\\n no: repartir parejo", fontsize=9, fontcolor="#52514e"];
        }''')
    st.markdown("**¿Por qué estas técnicas?**")
    st.markdown("""
- **Red neuronal (perceptrón multicapa):** capta relaciones no lineales entre señales (por ejemplo, el momentum funciona distinto con volatilidad alta). Se comparó contra *gradient boosting* y se eligió la que ganó en validación **1996–1999**, antes del periodo de evaluación.
- **Formulación de ranking:** importa si un factor le gana a los demás, no cuánto rinde. Eso es lo que decide el portafolio.
- **Meta-labeling (López de Prado, 2018):** un segundo modelo aprende cuándo el primero se equivoca. Es la capa de confiabilidad que estudia la tesina.
- **Evaluación walk-forward:** el modelo se reentrena cada año solo con el pasado, y las reglas de éxito se escribieron antes de correr para no engañarnos.
""")

# ---------------------------------------------------------------- 1. Datos
with tabs[1]:
    st.markdown("Un **factor** es un estilo de inversión: comprar empresas baratas, rentables, que vienen "
                "subiendo, etc. Cada mes a unos les va mejor que a otros. **¿Se puede anticipar cuáles?**")
    region = st.radio("Región", list(REGIONS), format_func=lambda r: {"us": "Estados Unidos (desde 1963)",
                      "dev_ex_us": "Desarrollados sin EE.UU. (desde 1990)"}[r], horizontal=True, key="r1")
    f = load(f"factors_{region}.parquet")
    fig = go.Figure()
    for k, name in FACTOR_NAMES.items():
        fig.add_scatter(x=f.index, y=(1 + f[k]).cumprod(), name=name, line=dict(width=2, color=FACTOR_COLOR[k]),
                        hovertemplate="%{y:$,.2f}")
    fig.update_yaxes(type="log")
    st.plotly_chart(style(fig, 470, "Valor de $1 invertido (escala log)"), width="stretch")
    st.caption("Retornos en exceso de la tasa libre de riesgo; los factores distintos al mercado son portafolios largo-corto.")

# ---------------------------------------------------------------- 2. Decisión de hoy
with tabs[2]:
    today = load_json("today.json")
    region = st.radio("Región", list(REGIONS), format_func=lambda r: REGIONS[r].split(" (")[0],
                      horizontal=True, key="r2")
    t = today[region]
    probs = pd.Series(t["probs"]).sort_values()
    st.markdown(f"Con datos al cierre de **{MESES[pd.Timestamp(t['mes_decision']).month - 1]} de {pd.Timestamp(t['mes_decision']).year}**, la red neuronal estima "
                "para cada factor la probabilidad de quedar **en la mitad ganadora** el próximo mes.")
    colors = [C[1] if k in t["picks"] else GRAY for k in probs.index]
    fig = go.Figure(go.Bar(x=probs.values, y=[FACTOR_NAMES[k] for k in probs.index], orientation="h",
                           marker=dict(color=colors, cornerradius=4), text=[f"{v:.0%}" for v in probs.values],
                           textposition="outside", hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.add_vline(x=0.5, line=dict(color=INK2, dash="dot", width=1))
    style(fig, 360, None, None).update_xaxes(tickformat=".0%", range=[0, 1])
    fig.update_layout(hovermode="closest", showlegend=False)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig, width="stretch")
    with c2:
        st.metric("La IA elige", " + ".join(FACTOR_NAMES[k] for k in t["picks"]))
        st.markdown("**Lo que diría el v3:** repartir parejo, 1/6 en cada factor. La calibración con datos "
                    "anteriores a 2005 concluyó que inclinar los pesos no ayudaba.")
        st.caption("En naranja, los 2 factores elegidos. La línea punteada marca 50%.")

# ---------------------------------------------------------------- 3. Backtest
with tabs[3]:
    st.markdown("Cuánto habría crecido **$1** con cada estrategia. Mueve el **costo por operación**: la IA cambia "
                "de factores casi cada mes, y cada cambio cuesta.")
    c1, c2 = st.columns([2, 1])
    region = c1.radio("Región", list(REGIONS), format_func=lambda r: REGIONS[r], key="r3")
    cost_bp = c2.slider("Costo por unidad de rotación (puntos base)", 0, 25, 10, 1)
    s = load(f"strategies_{region}.parquet")
    keys = [k for k in STRATS if k in s.estrategia.unique()]
    chosen = st.multiselect("Estrategias", keys, default=keys, format_func=lambda k: STRATS[k][0])
    fig, rows = go.Figure(), {}
    for k in keys:
        if k not in chosen:
            continue
        d = s[s.estrategia == k].set_index("date")
        net = d["gross"] - cost_bp / 1e4 * d["turnover"]
        name, color = STRATS[k]
        fig.add_scatter(x=net.index, y=(1 + net).cumprod(), name=name, line=dict(width=2, color=color),
                        hovertemplate="%{y:$,.2f}")
        rows[name] = metrics(net, d["turnover"])
    st.plotly_chart(style(fig, 450, "Valor de $1"), width="stretch")
    if rows:
        tbl = pd.DataFrame(rows).T.sort_values("Sharpe", ascending=False)
        st.dataframe(tbl.style.format({"Retorno anual": "{:.1%}", "Volatilidad": "{:.1%}", "Sharpe": "{:.2f}",
                                       "Peor caída": "{:.1%}", "Rotación mensual": "{:.2f}"}),
                     width="stretch")
    st.caption("Sharpe = retorno por unidad de riesgo (más alto es mejor). El v3 (trimestral + pesos + freno en "
               "crisis) quedó calibrado como 'repartir parejo', por eso comparte línea con equal-weight.")

# ---------------------------------------------------------------- 4. Gate
with tabs[4]:
    st.markdown("Le pusimos a la IA un **segundo modelo** que decide cuándo **no** hacerle caso y quedarse en "
                "'repartir parejo'. Solo Estados Unidos, 2005–2026.")
    g = load("gate_us.parquet")
    s = load("strategies_us.parquet")
    auc = roc_auc_score(g["meta_label"], g["p_meta"])
    abst = g[g["abstain"]]
    prec = (abst["pick_ret"] < abst["ew_ret"]).mean()
    m1, m2, m3 = st.columns(3)
    m1.metric("Meses en que se abstuvo", f"{len(abst)} de {len(g)}")
    m2.metric("Acertó al abstenerse", f"{prec:.0%}")
    m2.caption("Necesitaba más de 50%")
    m3.metric("Capacidad de distinguir (AUC)", f"{auc:.2f}")
    m3.caption("0.50 equivale a una moneda al aire")

    c1, c2 = st.columns(2)
    fig = go.Figure()
    for k in ("primary", "gated", "ew"):
        d = s[s.estrategia == k].set_index("date")
        net = d["gross"] - 0.001 * d["turnover"]
        fig.add_scatter(x=net.index, y=(1 + net).cumprod(), name=STRATS[k][0],
                        line=dict(width=2, color=STRATS[k][1]), hovertemplate="%{y:$,.2f}")
    c1.markdown("**Valor de $1, costo de 10 pb**")
    c1.plotly_chart(style(fig, 380), width="stretch")

    g["quintil"] = pd.qcut(g["p_meta"], 5, labels=False)
    cal = g.groupby("quintil").agg(conf=("p_meta", "mean"), real=("meta_label", "mean"), n=("p_meta", "size"))
    fig = go.Figure()
    fig.add_scatter(x=[0.35, 0.75], y=[0.35, 0.75], mode="lines", name="Calibración perfecta",
                    line=dict(color=GRAY, dash="dot", width=1), hoverinfo="skip")
    fig.add_scatter(x=cal["conf"], y=cal["real"], mode="lines+markers", name="Gate",
                    line=dict(color=C[4], width=2), marker=dict(size=10),
                    hovertemplate="confianza %{x:.0%} → la IA acertó %{y:.0%}<extra></extra>")
    style(fig, 380).update_xaxes(tickformat=".0%", title="Confianza del gate en la IA")
    fig.update_yaxes(tickformat=".0%", title="Qué tan seguido acertó la IA")
    fig.update_layout(hovermode="closest")
    c2.markdown("**¿Su confianza significaba algo?**")
    c2.plotly_chart(fig, width="stretch")
    st.info("Cuando el gate menos confiaba, la IA acertaba **más**. Dudaba en los momentos equivocados.")

# ---------------------------------------------------------------- 5. Lo que aprendimos
with tabs[5]:
    b = load("blocks.parquet")
    ic = load("ic.parquet")
    c1, c2 = st.columns(2)
    fig = go.Figure(go.Bar(x=b["bloque"], y=b["modelo"], marker=dict(color=C[1], cornerradius=4),
                           text=[f"{v:+.2%}" for v in b["modelo"]], textposition="outside",
                           customdata=b["oraculo"],
                           hovertemplate="%{x}: IA %{y:+.2%} al mes · máximo posible %{customdata:.1%}<extra></extra>"))
    style(fig, 360).update_yaxes(tickformat=".1%", title="Ganancia mensual sobre repartir parejo")
    fig.update_layout(hovermode="closest", showlegend=False)
    c1.markdown("**La ventaja de la IA cambia con el tiempo** (EE.UU., antes de costos)")
    c1.plotly_chart(fig, width="stretch")

    ic_labels = ["EE.UU. 1996–2004<br>(calibración)", "EE.UU. 2005–2026<br>(ya visto)",
                 "Fuera de EE.UU. 2001–2026<br>(nunca visto)"]
    fig = go.Figure(go.Bar(x=ic_labels, y=ic["ic"], marker=dict(color=C[2], cornerradius=4),
                           text=[f"{v:+.2f}" for v in ic["ic"]], textposition="outside",
                           customdata=np.c_[ic["acierto"], ic["trimestres"]],
                           hovertemplate="%{x}<br>IC %{y:+.2f} · acierto %{customdata[0]:.0%} · "
                                         "%{customdata[1]} trimestres<extra></extra>"))
    style(fig, 360).update_yaxes(title="¿Ordena bien los factores? (IC)")
    fig.update_layout(hovermode="closest", showlegend=False)
    c2.markdown("**La señal trimestral existe y viaja a otro mercado**")
    c2.plotly_chart(fig, width="stretch")

    st.markdown("""
### La historia en cinco pasos
1. **La IA aprende a elegir factores**: acierta más de la mitad de las veces, pero cambia tanto de opinión que los costos se comen la ganancia.
2. **Le pusimos un gate para dudar**: dudaba justo cuando la IA tenía razón.
3. **Le pusimos un freno a la rotación**: los años de calibración le enseñaron que no lo necesitaba.
4. **Rebalanceo trimestral, pesos y freno en crisis (v3)**: la calibración eligió repartir parejo. En el examen limpio ganó, pero por no hacer timing.
5. **Lo valioso**: la señal trimestral **sí ordena bien los factores en un mercado que nunca vio**. El reto no es encontrarla, sino convertirla en mejor riesgo-retorno que la diversificación.

**Siguiente paso (tesina):** calibrar con muchos mercados a la vez, usar pesos moderados en lugar de elegir 2 y reservar los mercados emergentes como examen limpio.
""")
    st.caption("Los experimentos 2 a 4 son exploratorios: se diseñaron después de ver los resultados del primero.")
