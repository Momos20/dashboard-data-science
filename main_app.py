import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
st.set_page_config(page_title="EDA - Monitoreo Ambiental", layout="wide", page_icon="📊")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      div[data-testid="stMetric"] { background: rgba(255,255,255,0.05); padding: 10px 12px; border-radius: 12px; }
      .card { background: rgba(255,255,255,0.04); padding: 14px 16px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.06); }
      .small-note { opacity: 0.75; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 EDA - Monitoreo Ambiental")
st.caption("EDA organizado en 3 bloques: Cuantitativo, Cualitativo y Gráfico (dinámico).")

# ------------------------------------------------------------
# Sidebar: upload + read options
# ------------------------------------------------------------
st.sidebar.header("⚙️ Carga y configuración")
uploaded = st.sidebar.file_uploader("Suba su archivo CSV", type=["csv"])

with st.sidebar.expander("Opciones de lectura", expanded=False):
    sep = st.text_input("Separador (opcional)", value="")
    encoding = st.text_input("Encoding (opcional)", value="")
    decimal = st.text_input("Decimal (opcional)", value="")
    na_values = st.text_input("NA values (coma-separado)", value="")

sep = sep.strip() or None
encoding = encoding.strip() or None
decimal = decimal.strip() or None
na_values = [x.strip() for x in na_values.split(",") if x.strip()] or None

if uploaded is None:
    st.info("⬅️ Suba un archivo CSV desde el panel izquierdo para comenzar.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_csv(file, sep, encoding, decimal, na_values):
    return pd.read_csv(file, sep=sep, encoding=encoding, decimal=decimal, na_values=na_values)

df = load_csv(uploaded, sep, encoding, decimal, na_values)

# ------------------------------------------------------------
# Datetime detection
# ------------------------------------------------------------
DATETIME_HINTS = ["fecha", "date", "hora", "time", "timestamp", "datetime", "created", "updated"]

@st.cache_data(show_spinner=False)
def try_parse_datetime_columns(data: pd.DataFrame, threshold=0.6):
    data = data.copy()
    dt_cols = []
    for c in data.columns:
        cname = str(c).lower()
        if any(k in cname for k in DATETIME_HINTS) or data[c].dtype == "object":
            try:
                parsed = pd.to_datetime(data[c], errors="coerce", infer_datetime_format=True)
                if parsed.notna().mean() >= threshold:
                    data[c] = parsed
                    dt_cols.append(c)
            except Exception:
                pass
    return data, dt_cols

df, dt_cols = try_parse_datetime_columns(df)

def split_columns(data: pd.DataFrame):
    num = data.select_dtypes(include=[np.number]).columns.tolist()
    dt = data.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    boo = data.select_dtypes(include=["bool"]).columns.tolist()
    cat = data.select_dtypes(include=["object", "category"]).columns.tolist()
    return num, cat, dt, boo

num_cols, cat_cols, dt_cols2, bool_cols = split_columns(df)
dt_cols = list(dict.fromkeys(dt_cols + dt_cols2))

# ------------------------------------------------------------
# Downsample for plots
# ------------------------------------------------------------
st.sidebar.header("🧰 Rendimiento")
use_downsample = st.sidebar.checkbox("Downsample para gráficas", value=True)
downsample_n = st.sidebar.slider("Tamaño downsample", 500, 20000, 5000, step=500) if use_downsample else None

def maybe_downsample(data: pd.DataFrame):
    if not use_downsample:
        return data
    if len(data) > downsample_n:
        return data.sample(downsample_n, random_state=42)
    return data

df_plot = maybe_downsample(df)

# ------------------------------------------------------------
# Overview metrics
# ------------------------------------------------------------
n_rows, n_cols = df.shape
total_missing = int(df.isna().sum().sum())
dup_rows = int(df.duplicated().sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Filas", f"{n_rows:,}")
c2.metric("Columnas", f"{n_cols:,}")
c3.metric("Nulos", f"{total_missing:,}")
c4.metric("Duplicadas", f"{dup_rows:,}")
c5.metric("Numéricas", f"{len(num_cols):,}")

with st.expander("Vista rápida (primeras filas)", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)

# ------------------------------------------------------------
# 3 BLOQUES
# ------------------------------------------------------------
tab_qt, tab_ql, tab_g = st.tabs(["📐 Cuantitativo", "🗂️ Cualitativo", "🎛️ Gráfico (dinámico)"])

# ============================================================
# 1) CUANTITATIVO
# ============================================================
with tab_qt:
    st.subheader("📐 Bloque Cuantitativo")

    if not num_cols:
        st.warning("No hay columnas numéricas.")
    else:
        st.markdown("### Estadísticas descriptivas")
        st.dataframe(df[num_cols].describe().T, use_container_width=True)

        st.markdown("### Ceros y outliers (IQR)")
        out = []
        zeros = []
        for c in num_cols:
            s = df[c]
            zeros.append({"columna": c, "ceros": int((s == 0).sum()), "%": round((s == 0).mean() * 100, 2)})

            s2 = s.dropna()
            if len(s2) < 8:
                out.append({"columna": c, "outliers(IQR)": 0, "%": 0.0})
                continue
            q1, q3 = np.percentile(s2, 25), np.percentile(s2, 75)
            iqr = q3 - q1
            if iqr == 0:
                out.append({"columna": c, "outliers(IQR)": 0, "%": 0.0})
                continue
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            k = int(((s < lo) | (s > hi)).sum())
            out.append({"columna": c, "outliers(IQR)": k, "%": round(k / len(df) * 100, 2)})

        cA, cB = st.columns(2)
        with cA:
            st.dataframe(pd.DataFrame(zeros).sort_values("ceros", ascending=False), use_container_width=True, height=320)
        with cB:
            st.dataframe(pd.DataFrame(out).sort_values("outliers(IQR)", ascending=False), use_container_width=True, height=320)

        st.markdown("### Correlación")
        if len(num_cols) < 2:
            st.info("Se requieren al menos 2 columnas numéricas.")
        else:
            method = st.radio("Método", ["pearson", "spearman"], horizontal=True, key="corr_method")
            corr = df[num_cols].corr(method=method)

            # Triangular (menos ruido)
            corr_tri = corr.copy()
            corr_tri.values[np.triu_indices_from(corr_tri.values, k=1)] = np.nan

            fig = px.imshow(
                corr_tri,
                text_auto=True,
                aspect="auto",
                title=f"Matriz de correlación ({method}) – triangular",
            )
            st.plotly_chart(fig, use_container_width=True)

            pairs = (
                corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))
                    .stack()
                    .reset_index()
            )
            pairs.columns = ["var1", "var2", "corr"]
            pairs["abs"] = pairs["corr"].abs()
            top = pairs.sort_values("abs", ascending=False).head(15).drop(columns="abs")

            st.markdown("**Top correlaciones (magnitud)**")
            st.dataframe(top, use_container_width=True)

# ============================================================
# 2) CUALITATIVO
# ============================================================
with tab_ql:
    st.subheader("🗂️ Bloque Cualitativo")

    st.markdown("### Tipos, cardinalidad y ejemplos")
    prof = []
    for c in df.columns:
        s = df[c]
        prof.append({
            "columna": c,
            "tipo": str(s.dtype),
            "nulos": int(s.isna().sum()),
            "% nulos": round(float(s.isna().mean() * 100), 2),
            "únicos": int(s.nunique(dropna=True)),
            "ejemplos": ", ".join([str(x) for x in s.dropna().astype(str).head(3).tolist()])
        })
    prof_df = pd.DataFrame(prof).sort_values(["% nulos", "únicos"], ascending=[False, False])
    st.dataframe(prof_df, use_container_width=True)

    st.markdown("### Nulos (detalle)")
    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0]

    if miss.empty:
        st.success("✅ No hay valores nulos.")
    else:
        miss_df = pd.DataFrame({
            "columna": miss.index,
            "nulos": miss.values,
            "%": (miss.values / len(df) * 100).round(2)
        })
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(miss_df, use_container_width=True, height=350)
        with c2:
            fig = px.bar(miss_df, x="columna", y="%", title="Porcentaje de nulos por columna", text="%")
            fig.update_layout(xaxis_title="", yaxis_title="% nulos")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Duplicados")
    st.metric("Filas duplicadas", f"{dup_rows:,}")
    if dup_rows > 0:
        with st.expander("Ver ejemplo de duplicados", expanded=False):
            st.dataframe(df[df.duplicated(keep=False)].head(50), use_container_width=True)

    st.markdown("### Distribución de categorías (top)")
    if not cat_cols:
        st.info("No hay columnas categóricas (object/category).")
    else:
        col = st.selectbox("Variable categórica", cat_cols, key="cat_col")
        topn = st.slider("Top N categorías", 5, 50, 15, key="cat_topn")

        vc = df[col].astype("string").value_counts(dropna=False).head(topn).reset_index()
        vc.columns = [col, "conteo"]

        fig = px.bar(vc, x="conteo", y=col, orientation="h", title=f"Top {topn}: {col}")
        fig.update_layout(yaxis_title="", xaxis_title="conteo")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(vc, use_container_width=True)

# ============================================================
# 3) GRÁFICO (DINÁMICO)
# ============================================================
with tab_g:
    st.subheader("🎛️ Bloque Gráfico (dinámico)")
    st.caption("Elija el tipo de visualización y las variables. El gráfico se arma dinámicamente.")

    chart_type = st.selectbox(
        "Tipo de gráfico",
        [
            "Histograma (numérica)",
            "Boxplot (numérica)",
            "Scatter (num vs num)",
            "Serie temporal (datetime vs num)",
            "Barras (categórica)",
        ],
        key="chart_type"
    )

    if chart_type == "Histograma (numérica)":
        if not num_cols:
            st.warning("No hay columnas numéricas.")
        else:
            col = st.selectbox("Variable", num_cols, key="g_hist_col")
            bins = st.slider("Bins", 10, 150, 40, key="g_hist_bins")
            log_x = st.checkbox("Escala log (x)", value=False, key="g_hist_log")

            fig = px.histogram(df_plot, x=col, nbins=bins, marginal="rug", title=f"Histograma: {col}")
            if log_x:
                fig.update_xaxes(type="log")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Boxplot (numérica)":
        if not num_cols:
            st.warning("No hay columnas numéricas.")
        else:
            col = st.selectbox("Variable", num_cols, key="g_box_col")
            fig = px.box(df_plot, y=col, points="outliers", title=f"Boxplot: {col}")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter (num vs num)":
        if len(num_cols) < 2:
            st.info("Se requieren al menos 2 columnas numéricas.")
        else:
            x = st.selectbox("X", num_cols, key="g_scatter_x")
            y = st.selectbox("Y", [c for c in num_cols if c != x], key="g_scatter_y")
            trend = st.checkbox("Trendline", value=True, key="g_scatter_trend")
            color = st.selectbox("Color (opcional)", ["(ninguno)"] + cat_cols, key="g_scatter_color")

            fig = px.scatter(
                df_plot,
                x=x, y=y,
                color=None if color == "(ninguno)" else color,
                trendline="ols" if trend else None,
                opacity=0.65,
                title=f"{y} vs {x}"
            )
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Serie temporal (datetime vs num)":
        if not dt_cols:
            st.info("No se detectaron columnas datetime.")
        elif not num_cols:
            st.info("No hay columnas numéricas para graficar en el tiempo.")
        else:
            dt = st.selectbox("Datetime", dt_cols, key="g_time_dt")
            y = st.selectbox("Y (numérica)", num_cols, key="g_time_y")

            c1, c2, c3 = st.columns(3)
            with c1:
                freq = st.selectbox("Resample", ["(sin)", "D", "W", "M"], index=1, key="g_time_freq")
            with c2:
                agg = st.selectbox("Agregación", ["mean", "median", "sum", "min", "max"], key="g_time_agg")
            with c3:
                roll = st.slider("Rolling window", 1, 30, 7, key="g_time_roll")

            tmp = df[[dt, y]].dropna().sort_values(dt)
            if tmp.empty:
                st.warning("No hay datos suficientes (todo es NA en fecha o variable).")
            else:
                if freq != "(sin)":
                    tmp = tmp.set_index(dt).resample(freq).agg({y: agg}).reset_index()

                tmp["rolling"] = tmp[y].rolling(roll, min_periods=max(1, roll // 2)).mean()

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=tmp[dt], y=tmp[y], mode="lines", name=y))
                fig.add_trace(go.Scatter(x=tmp[dt], y=tmp["rolling"], mode="lines", name=f"rolling({roll})"))
                fig.update_layout(title=f"Serie temporal: {y}", xaxis_title=str(dt), yaxis_title=y)
                st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Barras (categórica)":
        if not cat_cols:
            st.info("No hay columnas categóricas.")
        else:
            col = st.selectbox("Categoría", cat_cols, key="g_bar_col")
            topn = st.slider("Top N", 5, 50, 15, key="g_bar_topn")

            vc = df[col].astype("string").value_counts(dropna=False).head(topn).reset_index()
            vc.columns = [col, "conteo"]

            fig = px.bar(vc, x="conteo", y=col, orientation="h", title=f"Top {topn}: {col}")
            fig.update_layout(yaxis_title="", xaxis_title="conteo")
            st.plotly_chart(fig, use_container_width=True)

st.caption("EDA interactivo – 3 bloques (Cuantitativo / Cualitativo / Gráfico dinámico) ✅")
