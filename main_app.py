import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="EDA - Monitoreo Ambiental", layout="wide")

st.title("📊 EDA - Monitoreo Ambiental")
st.caption("Suba un archivo CSV para comenzar el análisis exploratorio")

# -----------------------------
# Sidebar – Upload
# -----------------------------
st.sidebar.title("Carga de datos")

uploaded = st.sidebar.file_uploader("Suba su archivo CSV", type=["csv"])

sep = st.sidebar.text_input("Separador (opcional)", value="")
encoding = st.sidebar.text_input("Encoding (opcional)", value="")

sep = sep if sep.strip() else None
encoding = encoding if encoding.strip() else None

if uploaded is None:
    st.info("⬅️ Suba un archivo CSV desde el panel izquierdo para comenzar.")
    st.stop()

# -----------------------------
# Load data
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv(file, sep, encoding):
    return pd.read_csv(file, sep=sep, encoding=encoding)

df = load_csv(uploaded, sep, encoding)

# -----------------------------
# Try detect datetime
# -----------------------------
def try_parse_datetime(df):
    for c in df.columns:
        if any(k in c.lower() for k in ["fecha", "date", "hora", "time", "timestamp"]):
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().mean() > 0.6:
                    df[c] = parsed
                    return df, c
            except:
                pass
    return df, None

df, dt_col = try_parse_datetime(df)

# -----------------------------
# Overview
# -----------------------------
st.subheader("📌 Resumen del dataset")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Filas", f"{len(df):,}")
c2.metric("Columnas", df.shape[1])
c3.metric("Nulos", int(df.isna().sum().sum()))
c4.metric("Numéricas", len(df.select_dtypes(include=np.number).columns))

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Vista", "Calidad", "Estadísticas", "Distribuciones", "Correlación & Tiempo"
])

# -----------------------------
# Tab 1 – Data
# -----------------------------
with tab1:
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Tipos de datos")
    st.dataframe(pd.DataFrame({
        "columna": df.columns,
        "tipo": df.dtypes.astype(str)
    }), use_container_width=True)

# -----------------------------
# Tab 2 – Data quality
# -----------------------------
with tab2:
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)

    if miss.empty:
        st.success("No hay valores nulos")
    else:
        miss_df = pd.DataFrame({
            "Nulos": miss,
            "%": (miss / len(df) * 100).round(2)
        })
        st.dataframe(miss_df, use_container_width=True)
        st.plotly_chart(
            px.bar(miss_df.reset_index(), x="index", y="%", title="Porcentaje de nulos"),
            use_container_width=True
        )

    st.write("Filas duplicadas:", df.duplicated().sum())

# -----------------------------
# Tab 3 – Stats
# -----------------------------
with tab3:
    num = df.select_dtypes(include=np.number)
    if num.empty:
        st.warning("No hay columnas numéricas")
    else:
        st.dataframe(num.describe().T, use_container_width=True)

# -----------------------------
# Tab 4 – Distributions
# -----------------------------
with tab4:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not num_cols:
        st.warning("No hay columnas numéricas")
    else:
        col = st.selectbox("Variable", num_cols)
        bins = st.slider("Bins", 10, 100, 40)

        st.plotly_chart(px.histogram(df, x=col, nbins=bins), use_container_width=True)
        st.plotly_chart(px.box(df, y=col), use_container_width=True)

# -----------------------------
# Tab 5 – Correlation & Time
# -----------------------------
with tab5:
    num_cols = df.select_dtypes(include=np.number).columns

    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    if dt_col:
        st.subheader("Serie temporal")
        y = st.selectbox("Variable", num_cols)
        st.plotly_chart(px.line(df, x=dt_col, y=y), use_container_width=True)
    else:
        st.info("No se detectó columna de fecha")

st.caption("EDA interactivo – listo para Streamlit Cloud")

