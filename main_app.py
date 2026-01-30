import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# Page config + small UI helpers
# ------------------------------------------------------------
st.set_page_config(page_title="EDA - Monitoreo Ambiental", layout="wide", page_icon="📊")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      div[data-testid="stMetric"] { background: rgba(255,255,255,0.05); padding: 10px 12px; border-radius: 12px; }
      .small-note { opacity: 0.75; font-size: 0.9rem; }
      .card { background: rgba(255,255,255,0.04); padding: 14px 16px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.06); }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 EDA - Monitoreo Ambiental")
st.caption("Cargue un CSV y explore calidad, distribución, relaciones, correlación y series temporales.")

# ------------------------------------------------------------
# Sidebar – Upload + options
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
# Type inference: datetime / numeric / categorical / boolean
# ------------------------------------------------------------
DATETIME_HINTS = ["fecha", "date", "hora", "time", "timestamp", "datetime", "created", "updated"]

def try_parse_datetime_columns(data: pd.DataFrame, threshold=0.6):
    data = data.copy()
    dt_cols = []
    for c in data.columns:
        if any(k in str(c).lower() for k in DATETIME_HINTS) or data[c].dtype == "object":
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

    # "categorical": objects + category, excluding datetime already parsed
    cat = data.select_dtypes(include=["object", "category"]).columns.tolist()
    return num, cat, dt, boo

num_cols, cat_cols, dt_cols2, bool_cols = split_columns(df)
dt_cols = list(dict.fromkeys(dt_cols + dt_cols2))  # unique preserving order

# ------------------------------------------------------------
# Global filters
# ------------------------------------------------------------
st.sidebar.header("🧰 Filtros")
max_rows_show = st.sidebar.slider("Filas a mostrar (vista)", 10, 500, 50, step=10)

use_downsample = st.sidebar.checkbox("Downsample para gráficas (rápido)", value=True)
downsample_n = st.sidebar.slider("Tamaño downsample", 500, 20000, 5000, step=500) if use_downsample else None

def maybe_downsample(data: pd.DataFrame):
    if not use_downsample:
        return data
    if len(data) > downsample_n:
        return data.sample(downsample_n, random_state=42)
    return data

df_plot = maybe_downsample(df)

# ------------------------------------------------------------
# Dataset overview metrics
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

with st.expander("📌 Contexto rápido", expanded=False):
    st.markdown(
        f"""
        <div class="card">
        <div class="small-note">
        - Columnas: <b>{n_cols}</b> (numéricas: <b>{len(num_cols)}</b>, categóricas: <b>{len(cat_cols)}</b>, datetime: <b>{len(dt_cols)}</b>, boolean: <b>{len(bool_cols)}</b>)<br/>
        - Downsample activo: <b>{'sí' if use_downsample else 'no'}</b> (para gráficas: <b>{len(df_plot):,}</b> filas)<br/>
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "👀 Vista", "🧼 Calidad", "📈 Univariado", "🔎 Bivariado", "🧠 Correlación", "⏱️ Tiempo"
])

# ------------------------------------------------------------
# Tab 1 – Data view + schema
# ------------------------------------------------------------
with tab1:
    st.subheader("Vista del dataset")
    st.dataframe(df.head(max_rows_show), use_container_width=True)

    st.subheader("Esquema (tipos + cardinalidad)")
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

# ------------------------------------------------------------
# Tab 2 – Data quality
# ------------------------------------------------------------
with tab2:
    st.subheader("Nulos")
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
        cA, cB = st.columns([1, 2])
        with cA:
            st.dataframe(miss_df, use_container_width=True, height=350)
        with cB:
            fig = px.bar(
                miss_df,
                x="columna",
                y="%",
                title="Porcentaje de nulos por columna",
                text="%",
            )
            fig.update_layout(xaxis_title="", yaxis_title="% nulos")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Duplicados, ceros y outliers (numéricas)")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Filas duplicadas", f"{dup_rows:,}")
    with c2:
        st.metric("Porcentaje duplicadas", f"{(dup_rows / max(n_rows,1) * 100):.2f}%")

    if num_cols:
        # zeros
        zero_counts = {c: int((df[c] == 0).sum()) for c in num_cols}
        zeros_df = pd.DataFrame({"columna": list(zero_counts.keys()), "ceros": list(zero_counts.values())})
        zeros_df["%"] = (zeros_df["ceros"] / len(df) * 100).round(2)
        zeros_df = zeros_df.sort_values("ceros", ascending=False)

        # outliers by IQR
        out = []
        for c in num_cols:
            s = df[c].dropna()
            if len(s) < 8:
                out.append({"columna": c, "outliers(IQR)": 0, "%": 0.0})
                continue
            q1, q3 = np.percentile(s, 25), np.percentile(s, 75)
            iqr = q3 - q1
            if iqr == 0:
                out.append({"columna": c, "outliers(IQR)": 0, "%": 0.0})
                continue
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            k = int(((df[c] < lo) | (df[c] > hi)).sum())
            out.append({"columna": c, "outliers(IQR)": k, "%": round(k / len(df) * 100, 2)})
        out_df = pd.DataFrame(out).sort_values("outliers(IQR)", ascending=False)

        cA, cB = st.columns(2)
        with cA:
            st.markdown("**Ceros por variable**")
            st.dataframe(zeros_df.head(20), use_container_width=True, height=350)
        with cB:
            st.markdown("**Outliers (IQR) por variable**")
            st.dataframe(out_df.head(20), use_container_width=True, height=350)
    else:
        st.info("No hay columnas numéricas para evaluar ceros/outliers.")

# ------------------------------------------------------------
# Tab 3 – Univariate analysis
# ------------------------------------------------------------
with tab3:
    st.subheader("Análisis univariado")
    mode = st.radio("Tipo de variable", ["Numérica", "Categórica"], horizontal=True)

    if mode == "Numérica":
        if not num_cols:
            st.warning("No hay columnas numéricas.")
        else:
            col = st.selectbox("Variable numérica", num_cols, key="uni_num_col")
            c1, c2, c3 = st.columns(3)
            with c1:
                bins = st.slider("Bins", 10, 150, 40, key="uni_bins")
            with c2:
                log_x = st.checkbox("Escala log (x)", value=False, key="uni_log")
            with c3:
                show_box = st.checkbox("Boxplot", value=True, key="uni_box")

            s = df_plot[col]

            fig = px.histogram(df_plot, x=col, nbins=bins, marginal="rug")
            fig.update_layout(title=f"Histograma: {col}", xaxis_title=col, yaxis_title="conteo")
            if log_x:
                fig.update_xaxes(type="log")
            st.plotly_chart(fig, use_container_width=True)

            if show_box:
                fig2 = px.box(df_plot, y=col, points="outliers")
                fig2.update_layout(title=f"Boxplot: {col}", yaxis_title=col)
                st.plotly_chart(fig2, use_container_width=True)

            # stats card
            s_clean = df[col].dropna()
            st.markdown(
                f"""
                <div class="card">
                <b>Resumen estadístico</b><br/>
                media: <b>{s_clean.mean():.4g}</b> · mediana: <b>{s_clean.median():.4g}</b> · std: <b>{s_clean.std():.4g}</b><br/>
                min: <b>{s_clean.min():.4g}</b> · p25: <b>{s_clean.quantile(0.25):.4g}</b> · p75: <b>{s_clean.quantile(0.75):.4g}</b> · max: <b>{s_clean.max():.4g}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        if not cat_cols:
            st.warning("No hay columnas categóricas (object/category).")
        else:
            col = st.selectbox("Variable categórica", cat_cols, key="uni_cat_col")
            topn = st.slider("Top N categorías", 5, 50, 15, key="uni_cat_topn")

            vc = df[col].astype("string").value_counts(dropna=False).head(topn).reset_index()
            vc.columns = [col, "conteo"]

            fig = px.bar(vc, x="conteo", y=col, orientation="h", title=f"Top {topn}: {col}")
            fig.update_layout(yaxis_title="", xaxis_title="conteo")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(vc, use_container_width=True)

# ------------------------------------------------------------
# Tab 4 – Bivariate analysis
# ------------------------------------------------------------
with tab4:
    st.subheader("Relaciones entre variables")

    if len(num_cols) >= 2:
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            x = st.selectbox("X (numérica)", num_cols, key="bi_x")
        with c2:
            y = st.selectbox("Y (numérica)", [c for c in num_cols if c != x], key="bi_y")
        with c3:
            trend = st.checkbox("Trendline", value=True, key="bi_trend")

        fig = px.scatter(df_plot, x=x, y=y, trendline="ols" if trend else None, opacity=0.65)
        fig.update_layout(title=f"{y} vs {x}", xaxis_title=x, yaxis_title=y)
        st.plotly_chart(fig, use_container_width=True)

        if cat_cols:
            st.markdown("**Segmentación por categoría (opcional)**")
            hue = st.selectbox("Color por categoría", ["(ninguna)"] + cat_cols, key="bi_hue")
            if hue != "(ninguna)":
                fig2 = px.scatter(df_plot, x=x, y=y, color=hue, opacity=0.65)
                fig2.update_layout(title=f"{y} vs {x} (color: {hue})")
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Se requieren al menos 2 columnas numéricas para análisis bivariado.")

# ------------------------------------------------------------
# Tab 5 – Correlation
# ------------------------------------------------------------
with tab5:
    st.subheader("Correlación (numéricas)")

    if len(num_cols) < 2:
        st.info("Se requieren al menos 2 columnas numéricas.")
    else:
        method = st.radio("Método", ["pearson", "spearman"], horizontal=True)
        corr = df[num_cols].corr(method=method)

        # Triangular mask via Plotly (show full but hide upper with NaNs)
        corr_tri = corr.copy()
        corr_tri.values[np.triu_indices_from(corr_tri.values, k=1)] = np.nan

        fig = px.imshow(
            corr_tri,
            text_auto=True,
            aspect="auto",
            title=f"Matriz de correlación ({method}) – triangular",
        )
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Top correlaciones (magnitud)**")
        pairs = (
            corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))
                .stack()
                .reset_index()
        )
        pairs.columns = ["var1", "var2", "corr"]
        pairs["abs"] = pairs["corr"].abs()
        top = pairs.sort_values("abs", ascending=False).head(15).drop(columns="abs")
        st.dataframe(top, use_container_width=True)

# ------------------------------------------------------------
# Tab 6 – Time series
# ------------------------------------------------------------
with tab6:
    st.subheader("Análisis temporal")

    if not dt_cols:
        st.info("No se detectaron columnas datetime. (Sugerencia: renombre a 'fecha', 'timestamp', etc.)")
    else:
        dt = st.selectbox("Columna datetime", dt_cols, key="t_dt")
        if not num_cols:
            st.info("No hay columnas numéricas para graficar en el tiempo.")
        else:
            y = st.selectbox("Variable numérica", num_cols, key="t_y")

            c1, c2, c3 = st.columns(3)
            with c1:
                freq = st.selectbox("Resample", ["(sin)", "D", "W", "M"], index=1, key="t_freq")
            with c2:
                agg = st.selectbox("Agregación", ["mean", "median", "sum", "min", "max"], key="t_agg")
            with c3:
                roll = st.slider("Rolling window", 1, 30, 7, key="t_roll")

            tmp = df[[dt, y]].dropna().sort_values(dt)
            if tmp.empty:
                st.warning("No hay datos suficientes (todo es NA en fecha o variable).")
            else:
                if freq != "(sin)":
                    tmp = tmp.set_index(dt).resample(freq).agg({y: agg}).reset_index()

                # rolling
                tmp["rolling"] = tmp[y].rolling(roll, min_periods=max(1, roll//2)).mean()

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=tmp[dt], y=tmp[y], mode="lines", name=y))
                fig.add_trace(go.Scatter(x=tmp[dt], y=tmp["rolling"], mode="lines", name=f"rolling({roll})"))
                fig.update_layout(title=f"Serie temporal: {y}", xaxis_title=str(dt), yaxis_title=y)
                st.plotly_chart(fig, use_container_width=True)

                # missingness over time (optional quick diagnostic)
                st.markdown("**Diagnóstico rápido: densidad temporal**")
                if freq == "(sin)":
                    freq2 = "D"
                else:
                    freq2 = freq
                dens = df[[dt]].dropna().set_index(dt).resample(freq2).size().reset_index(name="conteo_registros")
                fig2 = px.bar(dens, x=dt, y="conteo_registros", title=f"Registros por periodo ({freq2})")
                st.plotly_chart(fig2, use_container_width=True)

st.caption("EDA interactivo – listo para Streamlit Cloud ✅")

