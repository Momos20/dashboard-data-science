import re
import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# UI / CONFIG
# ============================================================
st.set_page_config(page_title="EDA Multinodal", layout="wide", page_icon="📊")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
      div[data-testid="stMetric"] { background: rgba(255,255,255,0.05); padding: 10px 12px; border-radius: 12px; }
      .card { background: rgba(255,255,255,0.04); padding: 14px 16px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.06); }
      .muted { opacity: 0.75; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 EDA Multinodal")
st.caption("Funciona con cualquier CSV. 4 pestañas: Cuantitativo, Cualitativo, Gráfico dinámico, Asistente (Groq).")

# ============================================================
# SIDEBAR - CONTROLES
# ============================================================
st.sidebar.header("⚙️ Ingesta")
uploaded = st.sidebar.file_uploader("Suba su CSV", type=["csv"])

if st.sidebar.button("🧹 Limpiar caché y recargar"):
    st.cache_data.clear()
    st.rerun()

with st.sidebar.expander("Opciones de lectura", expanded=False):
    sep_in = st.text_input("Separador (opcional)", value="")
    encoding_in = st.text_input("Encoding (opcional)", value="")
    decimal_in = st.text_input("Decimal (opcional: '.' o ',')", value="")
    na_values_in = st.text_input("NA values (coma-separado)", value="")

def clean_optional_str(x: str):
    x = (x or "").strip()
    return x if x else None

sep = clean_optional_str(sep_in)
encoding = clean_optional_str(encoding_in)

decimal_raw = (decimal_in or "").strip()
if decimal_raw == "":
    decimal = None
elif decimal_raw in [".", ","]:
    decimal = decimal_raw
else:
    st.sidebar.warning("Decimal inválido. Use '.' o ','. Se usará '.'.")
    decimal = "."

na_values = [t.strip() for t in (na_values_in or "").split(",") if t.strip()] or None
auto_fix = st.sidebar.checkbox("Auto-detectar sep/decimal", value=True)

st.sidebar.header("🧠 Tipado")
coerce_threshold = st.sidebar.slider("Umbral coerción numérica", 0.50, 0.99, 0.75, 0.01)
dt_threshold = st.sidebar.slider("Umbral detección datetime", 0.40, 0.99, 0.60, 0.01)

st.sidebar.header("⚡ Rendimiento")
use_downsample = st.sidebar.checkbox("Downsample para gráficas", value=True)
downsample_n = st.sidebar.slider("Tamaño downsample", 500, 200000, 5000, step=500) if use_downsample else None

# ============================================================
# NODO 1 - LOADER ROBUSTO (y arregla caso 1-columna)
# ============================================================
@st.cache_data(show_spinner=False)
def read_csv_try(file, sep, encoding, decimal, na_values):
    file.seek(0)
    return pd.read_csv(file, sep=sep, encoding=encoding, decimal=decimal, na_values=na_values)

@st.cache_data(show_spinner=False)
def load_csv_smart(file, sep, encoding, decimal, na_values, auto_fix=True):
    # intento 1: lo que el usuario puso
    try:
        df = read_csv_try(file, sep, encoding, decimal, na_values)
    except Exception:
        if not auto_fix:
            raise
        df = None

    # si falló o vino raro, probar combinaciones
    if df is None and auto_fix:
        candidates = [
            {"sep": ";", "decimal": ","},
            {"sep": ";", "decimal": "."},
            {"sep": ",", "decimal": "."},
            {"sep": ",", "decimal": ","},
            {"sep": "\t", "decimal": "."},
        ]
        last_err = None
        for cfg in candidates:
            try:
                df = read_csv_try(file, cfg["sep"], encoding, cfg["decimal"], na_values)
                break
            except Exception as e:
                last_err = e
        if df is None:
            raise last_err

    # ✅ Caso típico: separador incorrecto => 1 columna gigante
    if auto_fix and df is not None and df.shape[1] == 1:
        col0 = df.columns[0]
        sample = df[col0].astype("string").head(30).fillna("")

        semicolons = sample.str.count(";").mean()
        commas = sample.str.count(",").mean()
        tabs = sample.str.count("\t").mean()

        # Heurística: el separador más frecuente gana
        best_sep = None
        if max(semicolons, commas, tabs) >= 2:
            if semicolons >= commas and semicolons >= tabs:
                best_sep = ";"
            elif commas >= semicolons and commas >= tabs:
                best_sep = ","
            else:
                best_sep = "\t"

        if best_sep is not None:
            # reintento con decimal probable
            for dec in [decimal, ",", "."]:
                try:
                    df2 = read_csv_try(file, best_sep, encoding, dec, na_values)
                    if df2.shape[1] > 1:
                        df = df2
                        break
                except Exception:
                    pass

    return df

if uploaded is None:
    st.info("⬅️ Suba un CSV desde la barra lateral para comenzar.")
    st.stop()

df_raw = load_csv_smart(uploaded, sep, encoding, decimal, na_values, auto_fix=auto_fix)

# ============================================================
# NODO 2 - COERCIÓN NUMÉRICA (mejorada)
# ============================================================
def _normalize_numeric_text(s: pd.Series) -> pd.Series:
    """
    Normaliza texto numérico:
    - elimina moneda/unidades/espacios
    - soporta porcentajes
    - soporta negativos con paréntesis: (123) => -123
    - soporta miles/decimales tanto EU como LATAM
    """
    x = s.astype("string").str.strip()

    # negativos en paréntesis
    x = x.str.replace(r"^\((.*)\)$", r"-\1", regex=True)

    # quitar símbolos no numéricos comunes (mantener dígitos, signo, coma, punto)
    x = x.str.replace(r"[^\d\-\.,]", "", regex=True)

    return x

@st.cache_data(show_spinner=False)
def coerce_numeric_like_columns(data: pd.DataFrame, threshold=0.75):
    data = data.copy()

    for c in data.columns:
        if pd.api.types.is_numeric_dtype(data[c]) or pd.api.types.is_datetime64_any_dtype(data[c]) or pd.api.types.is_bool_dtype(data[c]):
            continue
        if not (pd.api.types.is_object_dtype(data[c]) or str(data[c].dtype) == "category"):
            continue

        s = data[c]
        x = _normalize_numeric_text(s)

        if x.dropna().empty:
            continue

        # variante A: formato LATAM: 1.234,56 -> 1234.56
        a = x.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        na = pd.to_numeric(a, errors="coerce")

        # variante B: formato US: 1,234.56 -> 1234.56
        b = x.str.replace(",", "", regex=False)
        nb = pd.to_numeric(b, errors="coerce")

        ok_a = na.notna().mean()
        ok_b = nb.notna().mean()

        best = na if ok_a >= ok_b else nb
        ok = max(ok_a, ok_b)

        # regla: si convierte bien y no destruye demasiados valores, aceptar
        if ok >= threshold:
            data[c] = best

    return data

# ============================================================
# NODO 3 - DATETIME DETECTION
# ============================================================
DATETIME_HINTS = ["fecha", "date", "hora", "time", "timestamp", "datetime", "created", "updated"]

@st.cache_data(show_spinner=False)
def try_parse_datetime_columns(data: pd.DataFrame, threshold=0.6):
    data = data.copy()
    dt_cols = []
    for c in data.columns:
        cname = str(c).lower()
        if any(k in cname for k in DATETIME_HINTS) or pd.api.types.is_object_dtype(data[c]):
            try:
                parsed = pd.to_datetime(data[c], errors="coerce", infer_datetime_format=True)
                if parsed.notna().mean() >= threshold:
                    data[c] = parsed
                    dt_cols.append(c)
            except Exception:
                pass
    return data, dt_cols

def split_columns(data: pd.DataFrame):
    num = data.select_dtypes(include=[np.number]).columns.tolist()
    dt = data.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    boo = data.select_dtypes(include=["bool"]).columns.tolist()
    cat = data.select_dtypes(include=["object", "category"]).columns.tolist()
    return num, cat, dt, boo

df = coerce_numeric_like_columns(df_raw, threshold=coerce_threshold)
df, dt_cols = try_parse_datetime_columns(df, threshold=dt_threshold)
num_cols, cat_cols, dt_cols2, bool_cols = split_columns(df)
dt_cols = list(dict.fromkeys(dt_cols + dt_cols2))

# ============================================================
# NODO 4 - RENDIMIENTO
# ============================================================
def maybe_downsample(data: pd.DataFrame):
    if not use_downsample:
        return data
    if downsample_n and len(data) > downsample_n:
        return data.sample(downsample_n, random_state=42)
    return data

df_plot = maybe_downsample(df)

# ============================================================
# DEBUG
# ============================================================
with st.expander("🧪 Debug: detección de tipos (abrir si algo falla)", expanded=False):
    st.write("Shape:", df.shape)
    st.write("Numéricas detectadas:", num_cols)
    st.write("Categóricas detectadas:", cat_cols[:30], "..." if len(cat_cols) > 30 else "")
    st.write("Datetime detectadas:", dt_cols)
    st.dataframe(pd.DataFrame({"columna": df.columns, "dtype": df.dtypes.astype(str)}), use_container_width=True)

# ============================================================
# OVERVIEW
# ============================================================
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

# ============================================================
# FUNCIONES PARA IA (GROQ)
# ============================================================
def build_eda_profile_for_llm(df: pd.DataFrame, num_cols, cat_cols, dt_cols, max_rows=1000) -> dict:
    prof = {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "duplicates": int(df.duplicated().sum()),
        "missing_total": int(df.isna().sum().sum()),
        "columns": [],
        "numeric_summary": {},
        "categorical_summary": {},
        "datetime_summary": {},
        "correlation_top_pairs": [],
    }

    for c in df.columns:
        s = df[c]
        prof["columns"].append({
            "name": str(c),
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": float(round(s.isna().mean() * 100, 2)),
            "nunique": int(s.nunique(dropna=True)),
        })

    if num_cols:
        num = df[num_cols].copy()
        desc = num.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
        desc = desc.round(4)
        prof["numeric_summary"]["describe"] = desc.to_dict()

        zeros = {}
        outliers_iqr = {}
        for c in num_cols:
            s = df[c]
            zeros[c] = int((s == 0).sum())

            s2 = s.dropna()
            if len(s2) >= 8:
                q1, q3 = np.percentile(s2, 25), np.percentile(s2, 75)
                iqr = q3 - q1
                if iqr > 0:
                    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    outliers_iqr[c] = int(((s < lo) | (s > hi)).sum())
                else:
                    outliers_iqr[c] = 0
            else:
                outliers_iqr[c] = 0

        prof["numeric_summary"]["zeros"] = zeros
        prof["numeric_summary"]["outliers_iqr"] = outliers_iqr

        if len(num_cols) >= 2:
            corr = df[num_cols].corr(method="pearson")
            pairs = (
                corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))
                    .stack()
                    .reset_index()
            )
            pairs.columns = ["var1", "var2", "corr"]
            pairs["abs"] = pairs["corr"].abs()
            top = pairs.sort_values("abs", ascending=False).head(15)
            prof["correlation_top_pairs"] = top[["var1", "var2", "corr"]].round(4).to_dict(orient="records")

    if cat_cols:
        cat_summary = {}
        for c in cat_cols[:30]:
            vc = df[c].astype("string").value_counts(dropna=False).head(10)
            cat_summary[str(c)] = vc.to_dict()
        prof["categorical_summary"]["top_values"] = cat_summary

    if dt_cols:
        dt_summary = {}
        for c in dt_cols[:10]:
            s = df[c].dropna()
            if not s.empty:
                dt_summary[str(c)] = {
                    "min": str(s.min()),
                    "max": str(s.max()),
                    "non_null": int(s.shape[0]),
                }
        prof["datetime_summary"]["ranges"] = dt_summary

    sample = df.head(max_rows).copy()
    prof["sample_head"] = sample.head(8).astype("string").to_dict(orient="records")

    return prof

def groq_chat_completion(api_key: str, model: str, system: str, user: str) -> str:
    from groq import Groq
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=900,
    )
    return completion.choices[0].message.content

# ============================================================
# 4 PESTAÑAS
# ============================================================
tab_qt, tab_ql, tab_g, tab_ai = st.tabs(
    ["📐 Cuantitativo", "🗂️ Cualitativo", "🎛️ Gráfico (dinámico)", "🤖 Asistente (Groq)"]
)

# ============================================================
# CUANTITATIVO + CORRELACIÓN
# ============================================================
with tab_qt:
    st.subheader("📐 Cuantitativo")
    if not num_cols:
        st.error(
            "No se detectaron columnas numéricas. "
            "Abra el panel Debug arriba para ver dtypes y si el CSV quedó en 1 sola columna. "
            "También pruebe 'Limpiar caché y recargar'."
        )
    else:
        st.markdown("### Estadísticas descriptivas")
        st.dataframe(df[num_cols].describe().T, use_container_width=True)

        st.markdown("### Correlación")
        if len(num_cols) < 2:
            st.info("Se requieren al menos 2 columnas numéricas.")
        else:
            method = st.radio("Método", ["pearson", "spearman"], horizontal=True, key="corr_method")
            corr = df[num_cols].corr(method=method)

            corr_tri = corr.copy()
            corr_tri.values[np.triu_indices_from(corr_tri.values, k=1)] = np.nan
            fig = px.imshow(corr_tri, text_auto=True, aspect="auto",
                            title=f"Matriz de correlación ({method}) – triangular")
            st.plotly_chart(fig, use_container_width=True)

            pairs = (
                corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))
                    .stack()
                    .reset_index()
            )
            pairs.columns = ["var1", "var2", "corr"]
            pairs["abs"] = pairs["corr"].abs()
            top = pairs.sort_values("abs", ascending=False).head(25).drop(columns="abs")

            st.markdown("**Top pares por |correlación|**")
            st.dataframe(top, use_container_width=True)

            st.markdown("**Explorar par**")
            cA, cB = st.columns(2)
            with cA:
                x = st.selectbox("X", num_cols, key="pair_x")
            with cB:
                y = st.selectbox("Y", [c for c in num_cols if c != x], key="pair_y")

            rho = corr.loc[x, y]
            st.markdown(f"<div class='card'><b>{method}({x}, {y})</b> = <b>{rho:.4f}</b></div>", unsafe_allow_html=True)

            fig2 = px.scatter(df_plot, x=x, y=y, trendline="ols", opacity=0.65,
                              title=f"{y} vs {x} (corr={rho:.4f})")
            st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# CUALITATIVO
# ============================================================
with tab_ql:
    st.subheader("🗂️ Cualitativo")

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

    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        st.success("✅ No hay valores nulos.")
    else:
        miss_df = pd.DataFrame({"columna": miss.index, "nulos": miss.values,
                                "%": (miss.values / len(df) * 100).round(2)})
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(miss_df, use_container_width=True, height=350)
        with c2:
            st.plotly_chart(px.bar(miss_df, x="columna", y="%", text="%",
                                   title="Porcentaje de nulos por columna"), use_container_width=True)

    st.metric("Filas duplicadas", f"{int(df.duplicated().sum()):,}")

    if cat_cols:
        col = st.selectbox("Variable categórica", cat_cols, key="cat_col")
        topn = st.slider("Top N categorías", 5, 80, 15, key="cat_topn")
        vc = df[col].astype("string").value_counts(dropna=False).head(topn).reset_index()
        vc.columns = [col, "conteo"]
        st.plotly_chart(px.bar(vc, x="conteo", y=col, orientation="h", title=f"Top {topn}: {col}"),
                        use_container_width=True)

# ============================================================
# GRÁFICO DINÁMICO (sin asistente aquí)
# ============================================================
with tab_g:
    st.subheader("🎛️ Gráfico (dinámico)")

    chart_type = st.selectbox(
        "Tipo de gráfico",
        ["Histograma (numérica)", "Boxplot (numérica)", "Scatter (num vs num)",
         "Serie temporal (datetime vs num)", "Barras (categórica)", "Heatmap correlación (numéricas)"],
        key="chart_type"
    )

    chart_ctx = {"chart_type": chart_type}

    if chart_type == "Histograma (numérica)":
        if not num_cols:
            st.warning("No hay columnas numéricas.")
        else:
            col = st.selectbox("Variable", num_cols, key="g_hist_col")
            bins = st.slider("Bins", 10, 200, 40, key="g_hist_bins")
            chart_ctx.update({"x": col, "bins": int(bins)})
            st.plotly_chart(px.histogram(df_plot, x=col, nbins=bins, title=f"Histograma: {col}"),
                            use_container_width=True)

    elif chart_type == "Boxplot (numérica)":
        if not num_cols:
            st.warning("No hay columnas numéricas.")
        else:
            col = st.selectbox("Variable", num_cols, key="g_box_col")
            chart_ctx.update({"y": col})
            st.plotly_chart(px.box(df_plot, y=col, points="outliers", title=f"Boxplot: {col}"),
                            use_container_width=True)

    elif chart_type == "Scatter (num vs num)":
        if len(num_cols) < 2:
            st.info("Se requieren al menos 2 columnas numéricas.")
        else:
            x = st.selectbox("X", num_cols, key="g_scatter_x")
            y = st.selectbox("Y", [c for c in num_cols if c != x], key="g_scatter_y")
            chart_ctx.update({"x": x, "y": y})
            st.plotly_chart(px.scatter(df_plot, x=x, y=y, trendline="ols", opacity=0.65, title=f"{y} vs {x}"),
                            use_container_width=True)

    elif chart_type == "Serie temporal (datetime vs num)":
        if not dt_cols:
            st.info("No se detectaron columnas datetime.")
        elif not num_cols:
            st.info("No hay columnas numéricas.")
        else:
            dt = st.selectbox("Datetime", dt_cols, key="g_time_dt")
            y = st.selectbox("Y", num_cols, key="g_time_y")
            chart_ctx.update({"dt": dt, "y": y})
            tmp = df[[dt, y]].dropna().sort_values(dt)
            st.plotly_chart(px.line(tmp, x=dt, y=y, title=f"Serie temporal: {y}"),
                            use_container_width=True)

    elif chart_type == "Barras (categórica)":
        if not cat_cols:
            st.info("No hay columnas categóricas.")
        else:
            col = st.selectbox("Categoría", cat_cols, key="g_bar_col")
            topn = st.slider("Top N", 5, 80, 15, key="g_bar_topn")
            chart_ctx.update({"cat": col, "topn": int(topn)})
            vc = df[col].astype("string").value_counts(dropna=False).head(topn).reset_index()
            vc.columns = [col, "conteo"]
            st.plotly_chart(px.bar(vc, x="conteo", y=col, orientation="h", title=f"Top {topn}: {col}"),
                            use_container_width=True)

    elif chart_type == "Heatmap correlación (numéricas)":
        if len(num_cols) < 2:
            st.info("Se requieren al menos 2 columnas numéricas.")
        else:
            method = st.radio("Método", ["pearson", "spearman"], horizontal=True, key="g_corr_method")
            chart_ctx.update({"corr_method": method})
            corr = df[num_cols].corr(method=method)
            st.plotly_chart(px.imshow(corr, aspect="auto", title=f"Heatmap correlación ({method})"),
                            use_container_width=True)

    # Guardar contexto del gráfico (para que el asistente lo use)
    st.session_state["chart_context"] = chart_ctx

# ============================================================
# ASISTENTE (GROQ) EN SU PROPIA PESTAÑA
# ============================================================
with tab_ai:
    st.subheader("🤖 Asistente de análisis (Groq)")
    st.caption("Genera un EDA accionable usando un perfil compacto del dataset y el contexto del gráfico dinámico actual.")

    if "groq_analysis_text" not in st.session_state:
        st.session_state["groq_analysis_text"] = ""

    with st.expander("Configurar y ejecutar (API Key + análisis)", expanded=True):
        groq_api_key = st.text_input(
            "GROQ API Key",
            type="password",
            help="Pegue aquí su API Key de Groq. Se usa solo para esta sesión.",
            key="groq_api_key"
        )

        model_name = st.selectbox(
            "Modelo",
            options=["llama-3.3-70b-versatile"],
            index=0,
            help="Modelo recomendado para análisis general.",
            key="groq_model"
        )

        max_rows_profile = st.slider(
            "Muestras para perfilado (para limitar tamaño del prompt)",
            200, 5000, 1000, 100,
            key="groq_max_rows"
        )

        extra_focus = st.text_area(
            "Enfoque adicional (opcional)",
            placeholder="Ej: 'enfocarse en calidad de datos, outliers y variables clave para predicción'.",
            height=100,
            key="groq_extra_focus"
        )

        run_btn = st.button("🧠 Generar análisis descriptivo", type="primary", use_container_width=True, key="groq_run")

    if run_btn:
        if not groq_api_key or len(groq_api_key.strip()) < 10:
            st.error("Ingrese una GROQ API Key válida para ejecutar el análisis.")
        else:
            with st.spinner("Generando análisis con Groq..."):
                profile = build_eda_profile_for_llm(
                    df=df,
                    num_cols=num_cols,
                    cat_cols=cat_cols,
                    dt_cols=dt_cols,
                    max_rows=max_rows_profile
                )

                chart_context = st.session_state.get("chart_context", {})

                system_prompt = (
                    "Eres un analista senior de datos. Tu tarea es redactar un análisis exploratorio (EDA) "
                    "de forma profesional y accionable, basado en un perfil del dataset y el contexto del gráfico actual. "
                    "No inventes valores: usa únicamente lo que está en el perfil/contexto. "
                    "Estructura la respuesta con encabezados y bullets claros."
                )

                user_prompt = f"""
Además del EDA general, tenga en cuenta el gráfico actual seleccionado por el usuario:

Contexto del gráfico:
{chart_context}

Con base en este perfil del dataset (en formato dict/JSON), genere:

1) Resumen ejecutivo (3-6 bullets).
2) Estadística descriptiva relevante (numéricas: rangos, dispersión, sesgos; categóricas: concentración).
3) Hallazgos importantes: variables dominantes, correlaciones fuertes (si existen), outliers/ceros, calidad de datos.
4) Riesgos y advertencias: nulos, duplicados, tipos sospechosos, variables casi constantes, posibles errores de captura.
5) Recomendaciones prácticas: qué limpiar, qué transformar (log/escala), qué variables revisar para modelado.
6) Si hay datetime: sugerir análisis temporal posible.
7) Recomendaciones específicas según el gráfico actual (qué mirar, qué validaciones hacer, hipótesis).

Enfoque adicional del usuario (si aplica): {extra_focus if extra_focus.strip() else "N/A"}

Perfil:
{profile}
"""

                try:
                    st.session_state["groq_analysis_text"] = groq_chat_completion(
                        api_key=groq_api_key.strip(),
                        model=model_name,
                        system=system_prompt,
                        user=user_prompt
                    )
                except Exception as e:
                    st.error(f"Error llamando a Groq: {e}")

    if st.session_state["groq_analysis_text"]:
        st.markdown("### 🧾 Análisis generado")
        st.write(st.session_state["groq_analysis_text"])
        st.download_button(
            "⬇️ Descargar análisis (.txt)",
            data=st.session_state["groq_analysis_text"].encode("utf-8"),
            file_name="analisis_eda_groq.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.info("Configure la API Key y presione **Generar análisis descriptivo** para obtener hallazgos y conclusiones.")

st.caption("EDA multinodal y agnóstico al dataset ✅")
