# main_app.py
import os
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="EDA - Monitoreo Ambiental", layout="wide")

DEFAULT_CSV_PATH = "monitoreo_ambiental.csv"

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str, sep: str | None = None, encoding: str | None = None) -> pd.DataFrame:
    # pyarrow acelera read_csv si está disponible (pandas lo usa internamente en algunos casos)
    return pd.read_csv(path, sep=sep, encoding=encoding)

def try_parse_datetime(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """
    Intenta detectar una columna de fecha/hora y convertirla a datetime.
    Retorna (df, datetime_col or None)
    """
    candidates = []
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["fecha", "date", "datetime", "time", "timestamp", "hora"]):
            candidates.append(c)

    # Si no hay candidatos por nombre, intenta con columnas object que parezcan fecha
    if not candidates:
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        candidates = obj_cols[:5]  # no probar demasiadas

    best_col = None
    best_score = 0

    for c in candidates:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            score = parsed.notna().mean()
            if score > best_score and score >= 0.6:  # umbral razonable
                best_score = score
                best_col = c
        except Exception:
            continue

    if best_col is not None:
        df = df.copy()
        df[best_col] = pd.to_datetime(df[best_col], errors="coerce", infer_datetime_format=True)
        return df, best_col

    return df, None

def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()

def top_missing(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    miss = df.isna().sum().sort_values(ascending=False)
    pct = (miss / len(df)).replace([np.inf, np.nan], 0)
    out = pd.DataFrame({"missing": miss, "missing_%": (pct * 100).round(2)})
    out = out[out["missing"] > 0].head(top_n)
    return out

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Configuración")

source_mode = st.sidebar.radio(
    "Fuente de datos",
    ["Usar archivo local (monitoreo_ambiental.csv)", "Subir CSV"],
    index=0
)

sep = st.sidebar.text_input("Separador (opcional)", value="")
encoding = st.sidebar.text_input("Encoding (opcional)", value="")

sep = sep if sep.strip() else None
encoding = encoding if encoding.strip() else None

df = None
data_path = None

if source_mode == "Subir CSV":
    uploaded = st.sidebar.file_uploader("Cargar CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded, sep=sep, encoding=encoding)
        data_path = "uploaded_file.csv"
else:
    if os.path.exists(DEFAULT_CSV_PATH):
        df = load_csv(DEFAULT_CSV_PATH, sep=sep, encoding=encoding)
        data_path = DEFAULT_CSV_PATH
    else:
        st.error(
            f"No se encontró el archivo '{DEFAULT_CSV_PATH}' en el directorio actual.\n\n"
            "Opciones:\n"
            "1) Ponga el CSV junto a este main_app.py, o\n"
            "2) Use la opción 'Subir CSV' en el panel izquierdo."
        )
        st.stop()

# -----------------------------
# Pre-procesamiento ligero
# -----------------------------
df, dt_col = try_parse_datetime(df)

st.title("EDA - Monitoreo Ambiental")
st.caption(f"Fuente: {data_path} | Filas: {len(df):,} | Columnas: {df.shape[1]}")

# -----------------------------
# KPIs / Overview
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Filas", f"{len(df):,}")
c2.metric("Columnas", f"{df.shape[1]}")
c3.metric("Nulos totales", f"{int(df.isna().sum().sum()):,}")
c4.metric("Columnas numéricas", f"{len(numeric_columns(df))}")

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Vista de datos",
    "Calidad (nulos/duplicados)",
    "Estadísticas",
    "Distribuciones",
    "Correlación y tiempo"
])

with tab1:
    st.subheader("Vista previa")
    n = st.slider("Filas a mostrar", 5, 200, 25)
    st.dataframe(df.head(n), use_container_width=True)

    st.subheader("Tipos de datos")
    dtypes_df = pd.DataFrame({
        "columna": df.columns,
        "dtype": [str(t) for t in df.dtypes]
    })
    st.dataframe(dtypes_df, use_container_width=True)

with tab2:
    st.subheader("Nulos por columna")
    miss_df = top_missing(df, top_n=50)
    if miss_df.empty:
        st.success("No se detectaron valores nulos.")
    else:
        st.dataframe(miss_df, use_container_width=True)
        fig = px.bar(
            miss_df.reset_index().rename(columns={"index": "columna"}),
            x="columna", y="missing_%", title="Porcentaje de nulos por columna"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Duplicados")
    dup_count = int(df.duplicated().sum())
    st.write(f"Filas duplicadas: **{dup_count:,}** ({(dup_count/len(df)*100):.2f}%)" if len(df) else "Sin datos.")
    if dup_count > 0:
        st.info("Puede considerar eliminar duplicados si no tienen significado (lecturas repetidas).")

with tab3:
    st.subheader("Resumen estadístico (numérico)")
    num_cols = numeric_columns(df)
    if not num_cols:
        st.warning("No hay columnas numéricas para describir.")
    else:
        st.dataframe(df[num_cols].describe().T, use_container_width=True)

    st.subheader("Cardinalidad (categóricas)")
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if not cat_cols:
        st.info("No se detectaron columnas categóricas.")
    else:
        card = pd.DataFrame({
            "columna": cat_cols,
            "unique": [df[c].nunique(dropna=True) for c in cat_cols],
            "top": [df[c].mode(dropna=True).iloc[0] if df[c].notna().any() else None for c in cat_cols]
        }).sort_values("unique", ascending=False)
        st.dataframe(card, use_container_width=True)

with tab4:
    st.subheader("Filtros rápidos")
    # filtro simple por columnas categóricas de baja cardinalidad
    filtered_df = df.copy()

    low_card_cols = []
    for c in df.select_dtypes(include=["object", "category", "bool"]).columns:
        nunq = df[c].nunique(dropna=True)
        if 2 <= nunq <= 30:
            low_card_cols.append(c)

    if low_card_cols:
        col_a, col_b = st.columns(2)
        with col_a:
            cat_filter_col = st.selectbox("Filtrar por columna categórica", ["(sin filtro)"] + low_card_cols)
        with col_b:
            if cat_filter_col != "(sin filtro)":
                options = sorted([x for x in df[cat_filter_col].dropna().unique().tolist()])
                selected = st.multiselect("Valores", options, default=options[: min(5, len(options))])
                if selected:
                    filtered_df = filtered_df[filtered_df[cat_filter_col].isin(selected)]
    else:
        st.info("No se encontraron columnas categóricas de baja cardinalidad para filtros rápidos.")

    st.caption(f"Filas después de filtros: {len(filtered_df):,}")

    st.subheader("Histogramas / Boxplots")
    num_cols = numeric_columns(filtered_df)
    if not num_cols:
        st.warning("No hay columnas numéricas para graficar.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            x_col = st.selectbox("Columna numérica", num_cols)
        with col2:
            bins = st.slider("Bins (histograma)", 10, 200, 40)

        # Histograma (Plotly)
        fig_h = px.histogram(filtered_df, x=x_col, nbins=bins, title=f"Histograma: {x_col}")
        st.plotly_chart(fig_h, use_container_width=True)

        # Boxplot (Plotly)
        fig_b = px.box(filtered_df, y=x_col, title=f"Boxplot: {x_col}", points="outliers")
        st.plotly_chart(fig_b, use_container_width=True)

        # Outliers por IQR
        q1 = filtered_df[x_col].quantile(0.25)
        q3 = filtered_df[x_col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        outlier_rate = ((filtered_df[x_col] < lo) | (filtered_df[x_col] > hi)).mean() * 100
        st.write(f"Outliers (regla IQR) estimados: **{outlier_rate:.2f}%**")

with tab5:
    st.subheader("Correlación (numéricas)")
    num_cols = numeric_columns(df)
    if len(num_cols) < 2:
        st.warning("Se requieren al menos 2 columnas numéricas para correlación.")
    else:
        method = st.selectbox("Método", ["pearson", "spearman"], index=0)
        corr = df[num_cols].corr(method=method)

        # Heatmap con seaborn/matplotlib (mejor para matrices grandes)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, linewidths=0.5)
        ax.set_title(f"Matriz de correlación ({method})")
        st.pyplot(fig)

        st.caption("Sugerencia: correlaciones altas pueden indicar variables redundantes o relaciones físicas esperables.")

    st.divider()

    st.subheader("Análisis temporal (si hay columna fecha)")
    if dt_col is None:
        st.info("No se detectó una columna de fecha/hora automáticamente.")
    else:
        st.write(f"Columna de fecha detectada: **{dt_col}**")
        df_time = df.dropna(subset=[dt_col]).sort_values(dt_col).copy()

        if df_time.empty:
            st.warning("La columna de fecha existe, pero no hay valores válidos para graficar.")
        else:
            # Selector de variable numérica
            num_cols_time = numeric_columns(df_time)
            if not num_cols_time:
                st.warning("No hay columnas numéricas para graficar contra el tiempo.")
            else:
                y = st.selectbox("Variable (numérica)", num_cols_time)

                # Resample opcional si hay mucha densidad
                with st.expander("Opciones de agregación temporal"):
                    freq = st.selectbox("Frecuencia", ["(sin agregación)", "H", "D", "W", "M"], index=2)
                    agg = st.selectbox("Agregación", ["mean", "median", "min", "max"], index=0)

                plot_df = df_time[[dt_col, y]].dropna()
                if freq != "(sin agregación)":
                    plot_df = (
                        plot_df.set_index(dt_col)
                        .resample(freq)
                        .agg({y: agg})
                        .reset_index()
                    )

                fig_ts = px.line(plot_df, x=dt_col, y=y, title=f"Serie de tiempo: {y}")
                st.plotly_chart(fig_ts, use_container_width=True)

st.divider()
st.caption("EDA básico en Streamlit: vista general, calidad de datos, estadísticos, distribuciones, correlación y tiempo.")
