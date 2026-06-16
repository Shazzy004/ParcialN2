#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modulo: app.py
Descripción: Interfaz de usuario interactiva y Dashboard en Streamlit.
             Carga los datos modelados y proporciona visualizaciones
             dinámicas con filtros interactivos, mapas de dispersión PCA
             de clusters de K-Means, tendencias con regresión y un generador
             de resúmenes ejecutivos con IA (Gemini).
Autor: Grupo 4 - Gestión de la Información (Semestre I, 2026)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env (si existe) para que la
# función de "Conclusiones con IA" pueda leer GEMINI_API_KEY. Sin esto, la key
# del .env no estaría disponible en el dashboard (solo en el pipeline).
load_dotenv()

# Configurar el diseño de la página de Streamlit de forma premium
st.set_page_config(
    page_title="Mercado Laboral IT Panamá - UTP",
    page_icon="🇵🇦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_JOBS_PATH = os.path.join(DATA_PROC_DIR, "vacantes_modeladas.csv")
MODEL_TRENDS_PATH = os.path.join(DATA_PROC_DIR, "tendencias_skills.csv")

# Estilos CSS personalizados para WOW al usuario (Estética Premium, fuentes limpias y tarjetas con sombras)
st.markdown("""
    <style>
    /* Estilo del contenedor principal */
    .reportview-container {
        background: var(--background-color, #f7f9fc);
    }
    /* Encabezado Principal */
    .main-header {
        font-family: 'Inter', 'Outfit', sans-serif;
        color: #1E3A8A;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        color: var(--text-color, #4B5563);
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    /* Tarjetas de Métricas Premium (Gradientes y Efectos Hover) */
    .metric-card {
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
        overflow: hidden;
        text-overflow: ellipsis;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border: 1px solid rgba(128, 128, 128, 0.08);
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.1);
    }
    
    /* Variaciones de Gradientes (Modo Claro) */
    .card-blue {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-left: 6px solid #2563EB;
        color: #1E40AF !important;
    }
    .card-green {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border-left: 6px solid #059669;
        color: #065F46 !important;
    }
    .card-amber {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border-left: 6px solid #D97706;
        color: #92400E !important;
    }
    .card-purple {
        background: linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 100%);
        border-left: 6px solid #7C3AED;
        color: #5B21B6 !important;
    }
    
    .metric-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.8;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 0.25rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Variaciones de Gradientes (Modo Oscuro mediante detector de esquema de color) */
    @media (prefers-color-scheme: dark) {
        .card-blue {
            background: linear-gradient(135deg, #1E3A8A 0%, #0F172A 100%) !important;
            border-left: 6px solid #3B82F6 !important;
            color: #EFF6FF !important;
        }
        .card-green {
            background: linear-gradient(135deg, #064E3B 0%, #0F172A 100%) !important;
            border-left: 6px solid #10B981 !important;
            color: #ECFDF5 !important;
        }
        .card-amber {
            background: linear-gradient(135deg, #78350F 0%, #0F172A 100%) !important;
            border-left: 6px solid #F59E0B !important;
            color: #FFFBEB !important;
        }
        .card-purple {
            background: linear-gradient(135deg, #4C1D95 0%, #0F172A 100%) !important;
            border-left: 6px solid #8B5CF6 !important;
            color: #F5F3FF !important;
        }
    }
    /* Contenedor de Insights IA (Adaptable a Modo Oscuro/Claro) */
    .insight-box {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 1px solid #BFDBFE;
        color: #1F2937 !important;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    .insight-box h3 {
        color: #1E3A8A !important;
        margin-top: 0;
    }
    .insight-box div, .insight-box p, .insight-box li, .insight-box span, .insight-box b {
        color: #1F2937 !important;
    }
    .insight-box ul, .insight-box ol, .insight-box li, .insight-box table, .insight-box tr, .insight-box td, .insight-box th {
        white-space: normal !important;
    }
    
    /* Tarjetas de Clusters: fondo claro fijo + texto NEGRO siempre, para
       garantizar legibilidad independientemente del tema (claro/oscuro) del
       sistema o de Streamlit. Se usa !important para ganarle a las variables
       de tema y a la media query de modo oscuro del navegador. */
    .cluster-card {
        background-color: #F3F4F6 !important;
        color: #111827 !important;
        border-radius: 8px;
        padding: 1rem;
        border-top: 4px solid #10B981;
        min-height: 220px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .cluster-card h4 {
        margin: 0;
        color: #111827 !important;
    }
    .cluster-card p, .cluster-card b {
        margin: 5px 0;
        font-size: 0.9rem;
        color: #1F2937 !important;
        opacity: 1;
    }
    
    /* Ajustes específicos de tema oscuro usando media query del sistema */
    @media (prefers-color-scheme: dark) {
        .insight-box {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
            border: 1px solid #334155 !important;
        }
        .insight-box h3 {
            color: #38bdf8 !important;
        }
        .insight-box div, .insight-box p, .insight-box li, .insight-box span, .insight-box b {
            color: #f1f5f9 !important;
        }
    }
    </style>
""", unsafe_allow_html=True)


# =====================================================================
# Cargar Datos y Manejar Fallbacks
# =====================================================================
@st.cache_data
def cargar_datos_dashboard():
    """
    Carga los archivos generados por la fase de ML.
    Si no existen, retorna None para activar el disparador del pipeline en UI.
    """
    if not os.path.exists(MODEL_JOBS_PATH) or not os.path.exists(MODEL_TRENDS_PATH):
        return None, None
        
    df_jobs = pd.read_csv(MODEL_JOBS_PATH)
    df_trends = pd.read_csv(MODEL_TRENDS_PATH)
    return df_jobs, df_trends


# Interfaz para cuando los datos no existen en el despliegue inicial
df_jobs, df_trends = cargar_datos_dashboard()

if df_jobs is None or df_trends is None:
    st.title("🇵🇦 Mercado Laboral IT Panamá")
    st.warning("⚠️ No se encontraron datos procesados ni modelos entrenados en el sistema.")
    
    st.info("Para que los estudiantes de la UTP o el profesor puedan iniciar la aplicación, presione el botón de abajo. Esto ejecutará de forma automática el pipeline completo: Scraping/Simulación -> Extracción por LLM -> Guardado en SQLite -> Modelos de Machine Learning (K-Means y Regresión Lineal).")
    
    if st.button("🚀 Ejecutar Pipeline de Ingesta y Entrenar Modelos ML", type="primary"):
        with st.spinner("Procesando datos y ejecutando modelos... (Esto tomará unos segundos)"):
            try:
                # Ejecutar pipeline y modelo programáticamente
                from pipeline import ejecutar_pipeline
                from modelo import ejecutar_modelado
                
                # Ejecutar
                ejecutar_pipeline(modo_simulado=True, num_simulados=200)
                ejecutar_modelado()
                
                st.success("🎉 ¡Proceso finalizado con éxito! Cargando el Dashboard...")
                st.rerun()
            except Exception as e:
                st.error(f"Error al inicializar el proyecto: {e}")
    st.stop()


def asegurar_logo_local():
    logo_path = os.path.join(BASE_DIR, "src", "logo_utp.png")
    if not os.path.exists(logo_path):
        try:
            import requests
            url = "https://utp.ac.pa/sites/default/files/logo_utp.png"
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                with open(logo_path, "wb") as f:
                    f.write(r.content)
        except Exception as e:
            pass
    return logo_path if os.path.exists(logo_path) else None


# =====================================================================
# Estructuración de la Barra Lateral (Filtros)
# =====================================================================
logo_local = asegurar_logo_local()
if logo_local:
    st.sidebar.image(logo_local, width=90)
else:
    st.sidebar.markdown("🏫 **UTP - FISC**")
st.sidebar.title("Filtros del Mercado")
st.sidebar.markdown("Use los controles inferiores para filtrar las ofertas de empleo e interactuar con los modelos.")

# 1. Búsqueda de Texto
search_query = st.sidebar.text_input("Buscar puesto (Ej: React, Backend)", "")

# 2. Filtro por Categorías
categorias_disponibles = sorted(df_jobs["categoria_rol"].unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Categorías de Rol",
    options=categorias_disponibles,
    default=categorias_disponibles
)

# 3. Filtro por Portales de Empleo
portales_disponibles = sorted(df_jobs["portal"].unique().tolist())
selected_portals = st.sidebar.multiselect(
    "Portales de Origen",
    options=portales_disponibles,
    default=portales_disponibles
)

# 4. Filtro por Habilidades Técnicas
# Obtener todas las habilidades únicas
all_skills = set()
for s_list in df_jobs["habilidades"].dropna().str.split(","):
    for s in s_list:
        if s.strip():
            all_skills.add(s.strip())
skills_disponibles = sorted(list(all_skills))

selected_skills = st.sidebar.multiselect(
    "Habilidades Técnicas Requeridas",
    options=skills_disponibles,
    default=[]
)

# 5. Filtro de Salario Máximo y Mínimo
min_sal = float(df_jobs["salario_min"].min())
max_sal = float(df_jobs["salario_max"].max())

salario_range = st.sidebar.slider(
    "Rango Salarial Mensual (USD $)",
    min_value=500.0,
    max_value=6000.0,
    value=(1000.0, 4500.0),
    step=100.0
)

# 6. Años de Experiencia Requeridos
experiencia_max = int(df_jobs["experiencia_anios"].max())
selected_exp = st.sidebar.slider(
    "Años de Experiencia Requerida",
    min_value=0,
    max_value=10,
    value=(0, 6)
)


# 7. Información de los Integrantes del Grupo 4
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 👥 Grupo 4 - Integrantes:
* **Bryan Law** (8-1011-2459)
* **Evaristo Alvarez** (8-1011-177)
* **Fernando Jimenez** (20-24-7669)
* **Manuel Campos** (8-1022-1118)
* **Diego Gordon** (8-1017-349)

*Gestión de la Información — Semestre I, 2026*
""")


# =====================================================================
# Lógica de Filtrado de Datos
# =====================================================================
df_filtrado = df_jobs.copy()

# Filtro por búsqueda de texto
if search_query:
    df_filtrado = df_filtrado[
        df_filtrado["puesto"].str.contains(search_query, case=False, na=False) |
        df_filtrado["titulo_original"].str.contains(search_query, case=False, na=False)
    ]

# Filtro por categorías
if selected_categories:
    df_filtrado = df_filtrado[df_filtrado["categoria_rol"].isin(selected_categories)]
else:
    df_filtrado = df_filtrado.iloc[0:0] # Tabla vacía si no selecciona ninguna

# Filtro por portales
if selected_portals:
    df_filtrado = df_filtrado[df_filtrado["portal"].isin(selected_portals)]
else:
    df_filtrado = df_filtrado.iloc[0:0]

# Filtro por habilidades (debe cumplir con al menos una de las seleccionadas si el filtro no está vacío)
if selected_skills:
    def tiene_habilidades(skills_str):
        if not skills_str:
            return False
        skills_oferta = [s.strip() for s in skills_str.split(",")]
        return any(skill in skills_oferta for skill in selected_skills)
    df_filtrado = df_filtrado[df_filtrado["habilidades"].apply(tiene_habilidades)]

# Filtro por rango salarial — lógica de SOLAPAMIENTO de intervalos.
# Bug anterior: exigía salario_min >= tope_inf Y salario_max <= tope_sup, lo que
# descartaba cualquier oferta cuyo rango se saliera aunque fuera 1$ del slider
# (p. ej. una vacante 1.000–5.000 desaparecía si el tope era 4.500). Lo correcto
# es mostrar toda oferta cuyo rango [min, max] se SOLAPE con el rango elegido:
#   oferta.min <= filtro.max  AND  oferta.max >= filtro.min
df_filtrado = df_filtrado[
    (df_filtrado["salario_min"] <= salario_range[1]) &
    (df_filtrado["salario_max"] >= salario_range[0])
]

# Filtro por experiencia
df_filtrado = df_filtrado[
    (df_filtrado["experiencia_anios"] >= selected_exp[0]) &
    (df_filtrado["experiencia_anios"] <= selected_exp[1])
]


# =====================================================================
# Diseño de la Interfaz Principal (Dashboard)
# =====================================================================

# Fila de Encabezado
st.markdown('<div class="main-header">Análisis del Mercado Laboral IT en Panamá</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Segundo Parcial - Gestión de la Información | Facultad de Ingeniería de Sistemas Computacionales (UTP)</div>', unsafe_allow_html=True)

# Fila 1: Tarjetas de Métricas Clave
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_ofertas = len(df_filtrado)
    st.markdown(f"""
        <div class="metric-card card-blue">
            <div class="metric-title">Ofertas Activas</div>
            <div class="metric-value">{total_ofertas}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    sal_promedio_min = df_filtrado["salario_min"].mean() if len(df_filtrado) > 0 else 0
    st.markdown(f"""
        <div class="metric-card card-green">
            <div class="metric-title">Salario Mínimo Promedio</div>
            <div class="metric-value">B/. {sal_promedio_min:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    sal_promedio_max = df_filtrado["salario_max"].mean() if len(df_filtrado) > 0 else 0
    st.markdown(f"""
        <div class="metric-card card-amber">
            <div class="metric-title">Salario Máximo Promedio</div>
            <div class="metric-value">B/. {sal_promedio_max:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    # Habilidad más buscada en el subset filtrado
    lista_skills_filtradas = []
    for s_list in df_filtrado["habilidades"].dropna().str.split(","):
        lista_skills_filtradas.extend([s.strip() for s in s_list if s.strip()])
    
    top_skill = "N/A"
    if lista_skills_filtradas:
        top_skill = pd.Series(lista_skills_filtradas).value_counts().index[0]
        
    st.markdown(f"""
        <div class="metric-card card-purple">
            <div class="metric-title">Habilidad Más Solicitada</div>
            <div class="metric-value">{top_skill}</div>
        </div>
    """, unsafe_allow_html=True)


# =====================================================================
# Estructura de Pestañas (Tabs)
# =====================================================================
tab_mercado, tab_clusters, tab_emergentes, tab_ia = st.tabs([
    "📊 Visión General del Mercado",
    "🤖 Clustering de Perfiles (ML)",
    "📈 Habilidades Emergentes (Tendencias)",
    "💡 Conclusiones con IA"
])

# ---------------------------------------------------------------------
# Pestaña 1: Visión General del Mercado
# ---------------------------------------------------------------------
with tab_mercado:
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Top 10 Habilidades Técnicas Más Demandadas")
        if lista_skills_filtradas:
            counts = pd.Series(lista_skills_filtradas).value_counts().head(10).reset_index()
            counts.columns = ["Habilidad", "Cantidad"]
            fig_skills = px.bar(
                counts,
                x="Cantidad",
                y="Habilidad",
                orientation='h',
                color="Cantidad",
                color_continuous_scale="Blues",
                labels={"Cantidad": "Número de Vacantes", "Habilidad": "Tecnología"}
            )
            fig_skills.update_layout(
                yaxis={'categoryorder':'total ascending'}, 
                height=400, 
                margin=dict(l=0, r=0, t=10, b=10),
                modebar=dict(color='#3B82F6', activecolor='#10B981')
            )
            st.plotly_chart(fig_skills, use_container_width=True)
        else:
            st.info("No hay suficientes datos para generar el gráfico.")
            
    with col_chart2:
        st.subheader("Distribución de Salarios por Categoría de Rol")
        if len(df_filtrado) > 0:
            fig_salarios = px.box(
                df_filtrado,
                x="categoria_rol",
                y="salario_max",
                color="categoria_rol",
                labels={"categoria_rol": "Categoría", "salario_max": "Salario Máximo (USD)"},
                title="Rangos Salariales Máximos por Categoría"
            )
            fig_salarios.update_layout(
                showlegend=False, 
                height=400, 
                margin=dict(l=0, r=0, t=30, b=10),
                modebar=dict(color='#3B82F6', activecolor='#10B981')
            )
            st.plotly_chart(fig_salarios, use_container_width=True)
        else:
            st.info("No hay suficientes datos para generar el gráfico.")
            
    st.subheader("Detalle de Ofertas Laborales IT Filtradas")
    if len(df_filtrado) > 0:
        st.dataframe(
            df_filtrado[["puesto", "empresa", "portal", "fecha_publicacion", "salario_min", "salario_max", "experiencia_anios", "habilidades", "categoria_rol"]],
            use_container_width=True,
            column_config={
                "puesto": "Puesto",
                "empresa": "Empresa",
                "portal": "Portal",
                "fecha_publicacion": "Publicado",
                "salario_min": st.column_config.NumberColumn("Salario Mín", format="$ %.2f"),
                "salario_max": st.column_config.NumberColumn("Salario Máx", format="$ %.2f"),
                "experiencia_anios": "Experiencia (años)",
                "habilidades": "Habilidades",
                "categoria_rol": "Categoría"
            }
        )
    else:
        st.info("No se encontraron vacantes con los filtros seleccionados.")

# ---------------------------------------------------------------------
# Pestaña 2: Clustering de Perfiles (ML - K-Means)
# ---------------------------------------------------------------------
with tab_clusters:
    st.subheader("Clustering de Perfiles IT con K-Means y PCA (2D)")
    st.markdown("""
        Esta sección aplica la técnica no supervisada de **K-Means Clustering** en base a la similitud de habilidades requeridas en las ofertas de empleo.
        Los datos de alta dimensionalidad (habilidades vectorizadas por TF-IDF) se proyectan en dos dimensiones utilizando **PCA (Análisis de Componentes Principales)** para poder visualizarlos en un plano cartesiano interactivo.
    """)
    
    if len(df_filtrado) > 0:
        # Gráfico interactivo de dispersión PCA
        fig_pca = px.scatter(
            df_filtrado,
            x="pca_x",
            y="pca_y",
            color="nombre_cluster",
            hover_name="puesto",
            hover_data=["empresa", "salario_max", "habilidades"],
            labels={"pca_x": "Componente Principal 1", "pca_y": "Componente Principal 2", "nombre_cluster": "Perfil Identificado"},
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="Distribución Geoespacial de Perfiles IT por Habilidades Técnicas"
        )
        
        fig_pca.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
        fig_pca.update_layout(
            height=550,
            modebar=dict(color='#3B82F6', activecolor='#10B981')
        )
        st.plotly_chart(fig_pca, use_container_width=True)
        
        # Desglose de cada cluster
        st.subheader("Análisis de los Clusters Encontrados")
        cols_cluster = st.columns(4)
        
        unique_clusters = df_filtrado["nombre_cluster"].dropna().unique()
        for idx, cl_name in enumerate(unique_clusters[:4]):
            with cols_cluster[idx]:
                df_cl = df_filtrado[df_filtrado["nombre_cluster"] == cl_name]
                conteo_cl = len(df_cl)
                sal_cl = df_cl["salario_max"].mean()
                
                # Obtener top skills del cluster
                skills_cl = []
                for s_list in df_cl["habilidades"].dropna().str.split(","):
                    skills_cl.extend([s.strip() for s in s_list if s.strip()])
                top_skills_cl = pd.Series(skills_cl).value_counts().head(3).index.tolist() if skills_cl else ["N/A"]
                
                st.markdown(f"""
                <div class="cluster-card">
                    <h4>{cl_name}</h4>
                    <p><b>Ofertas:</b> {conteo_cl} ({conteo_cl/len(df_filtrado)*100:.1f}%)</p>
                    <p><b>Salario Promedio:</b> B/. {sal_cl:,.2f}</p>
                    <p><b>Habilidades Clave:</b> {', '.join(top_skills_cl)}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Filtra menos datos para analizar los clusters.")

# ---------------------------------------------------------------------
# Pestaña 3: Habilidades Emergentes (Tendencias Temporales - Regresión)
# ---------------------------------------------------------------------
with tab_emergentes:
    st.subheader("Predicción y Análisis de Habilidades Emergentes")
    st.markdown("""
        Esta sección analiza la demanda temporal de las tecnologías en los últimos 6 meses.
        Ajustamos un modelo de **Regresión Lineal** ($y = mx + b$) para cada habilidad, donde la pendiente ($m$) representa la tasa de crecimiento quincenal de su demanda.
        Una **pendiente positiva alta** revela habilidades emergentes (con fuerte tendencia de adopción en Panamá).
    """)
    
    # Mostrar tabla de tendencias estimadas por el modelo de Regresión
    col_t1, col_t2 = st.columns([1, 1])
    
    with col_t1:
        st.markdown("### Clasificación de Tendencias Predictivas")
        st.dataframe(
            df_trends[["habilidad", "pendiente", "porcentaje_actual", "porcentaje_predicho_futuro", "tendencia"]],
            use_container_width=True,
            column_config={
                "habilidad": "Habilidad",
                "pendiente": st.column_config.NumberColumn("Tasa de Crecimiento (m)", format="%.3f"),
                "porcentaje_actual": st.column_config.NumberColumn("Frecuencia Actual (%)", format="%.1f %%"),
                "porcentaje_predicho_futuro": st.column_config.NumberColumn("Demanda Proyectada (%)", format="%.1f %%"),
                "tendencia": "Clasificación de Tendencia"
            }
        )
        
    with col_t2:
        st.markdown("### Proyección de las 5 Habilidades con Mayor Crecimiento")
        # Mostrar el top 5 de habilidades emergentes según la pendiente positiva
        top_emergentes = df_trends[df_trends["pendiente"] > 0].head(5)
        
        if len(top_emergentes) > 0:
            fig_trends = px.bar(
                top_emergentes,
                x="habilidad",
                y="pendiente",
                color="pendiente",
                color_continuous_scale="Viridis",
                labels={"pendiente": "Pendiente de Regresión (Crecimiento)", "habilidad": "Tecnología"},
                title="Pendiente de Crecimiento del Top Tecnologías Emergentes"
            )
            fig_trends.update_layout(
                height=385, 
                margin=dict(l=10, r=10, t=40, b=10),
                modebar=dict(color='#3B82F6', activecolor='#10B981')
            )
            st.plotly_chart(fig_trends, use_container_width=True)
        else:
            st.info("No se detectaron habilidades emergentes con tendencia positiva en la base de datos actual.")


# ---------------------------------------------------------------------
# Pestaña 4: Conclusiones con IA (Gemini API Integration)
# ---------------------------------------------------------------------
with tab_ia:
    st.subheader("Conclusiones del Mercado IT en Panamá Generadas por Inteligencia Artificial")
    
    # Datos condensados para pasar al Prompt
    top_skills_str = ", ".join(lista_skills_filtradas[:15]) if lista_skills_filtradas else "N/A"
    categoria_counts = df_filtrado["categoria_rol"].value_counts().to_dict()
    sal_promedio_general = df_filtrado["salario_max"].mean() if len(df_filtrado) > 0 else 0
    
    # Botón para generar conclusiones dinámicas
    generar_ia = st.button("✨ Generar/Actualizar Informe del Mercado con IA")

    # Vacío o el placeholder de la plantilla .env se tratan como "sin key"
    # (el informe cae al generador heurístico en vez de fallar contra la API).
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key in (None, "", "PEGA_TU_KEY_AQUI"):
        gemini_key = None

    if generar_ia:
        with st.spinner("Conectando con el LLM para redactar el informe técnico..."):
            if gemini_key:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=gemini_key)
                    
                    # Generar la fecha actual dinámicamente en español
                    meses_es = [
                        "enero", "febrero", "marzo", "abril", "mayo", "junio", 
                        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
                    ]
                    ahora = datetime.now()
                    fecha_actual = f"{ahora.day} de {meses_es[ahora.month - 1]} de {ahora.year}"
                    
                    prompt_analisis = f"""
                    Como un experto Científico de Datos y Consultor del Mercado Laboral IT en Centroamérica, redacta un informe ejecutivo dirigido a los directivos de la Universidad Tecnológica de Panamá (UTP) analizando la demanda del mercado laboral tecnológico según las estadísticas actuales de nuestra base de datos.
                    
                    ESTADÍSTICAS DEL PORTAL:
                    - Cantidad de vacantes analizadas: {total_ofertas}
                    - Salario máximo promedio en Panamá: B/. {sal_promedio_max:,.2f}
                    - Salario mínimo promedio en Panamá: B/. {sal_promedio_min:,.2f}
                    - Habilidad técnica líder: {top_skill}
                    - Distribución de vacantes por categorías de rol: {categoria_counts}
                    - Muestra de habilidades del mercado: {top_skills_str}
                    
                    Requisitos del informe y formato:
                    1. Divide el análisis en 3 secciones: "Diagnóstico del Mercado IT en Panamá", "Brecha de Habilidades en la Academia" y "Recomendaciones Curriculares para la UTP".
                    2. Sé extremadamente profesional, usa terminología técnica pero legible, y proporciona datos cuantitativos concretos adaptados a Panamá.
                    3. Mantén un tono optimista sobre las habilidades emergentes pero realista sobre las brechas salariales.
                    4. Cabecera del informe: Comienza el informe redactando obligatoriamente una cabecera estructurada con exactamente los siguientes campos al principio, sin corchetes ni placeholders:
                       **Para**: Directivos de la Universidad Tecnológica de Panamá (UTP)
                       **De**: Analistas del Grupo 4 (FISC-UTP)
                       **Fecha**: {fecha_actual}
                       **Asunto**: Análisis de la Demanda del Mercado Laboral IT en Panamá y Recomendaciones Curriculares
                    5. Formato de Listas y Viñetas: Para listas o puntos clave, utiliza un formato de un solo nivel (sin anidamientos complejos). Escribe listas simples usando un único guion '-' o asterisco '*' al inicio de la línea seguido directamente del texto (ej: `- **Nombre**: Detalle`). Evita dejar líneas vacías entre elementos de la lista y no utilices sub-viñetas o sangrías adicionales, para garantizar la alineación vertical perfecta del texto y los marcadores.
                    6. Firma final: Firma el reporte de forma profesional indicando únicamente:
                       
                       Atentamente,
                       **Grupo 4 (FISC-UTP)**
                       
                       E incluye la lista con viñetas de los integrantes:
                       - Bryan Law
                       - Evaristo Alvarez
                       - Fernando Jimenez
                       - Manuel Campos
                       - Diego Gordon
                    """
                    
                    # Probar múltiples modelos por compatibilidad en la nube de Google
                    modelos_a_probar = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-pro", "gemini-1.5-flash-latest"]
                    response = None
                    last_err = None
                    
                    for model_name in modelos_a_probar:
                        try:
                            model = genai.GenerativeModel(model_name)
                            response = model.generate_content(prompt_analisis)
                            if response and response.text:
                                break
                        except Exception as e:
                            last_err = e
                            continue
                            
                    if not response:
                        raise last_err
                        
                    texto_informe = response.text
                    
                    st.session_state["informe_ia"] = texto_informe
                    st.success("¡Informe generado con éxito por Gemini!")
                except Exception as e:
                    st.error(f"Error al llamar a Gemini API: {e}. Se mostrará el informe estadístico base.")
                    st.session_state["informe_ia"] = None
            else:
                st.session_state["informe_ia"] = None
                
    # Mostrar el informe en pantalla (usar session_state para persistirlo entre interacciones de filtros)
    if "informe_ia" in st.session_state and st.session_state["informe_ia"] is not None:
        st.markdown(f"""
        <div class="insight-box">
            <h3>📝 Informe Estratégico Generado por IA</h3>
            <div style="line-height: 1.6; white-space: pre-wrap;">
{st.session_state["informe_ia"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback del informe (Generador heurístico estructurado y profesional)
        st.markdown(f"""
        <div class="insight-box">
            <h3>📝 Informe Estadístico de Tendencias IT (Análisis Base FISC-UTP)</h3>
            <p style="line-height: 1.6;">
                <b>Diagnóstico del Mercado IT en Panamá:</b><br>
                El análisis de las <b>{total_ofertas}</b> ofertas de trabajo tecnológicas vigentes en los portales activos revela que la categoría con mayor tracción es 
                {", ".join([f"<b>{k}</b> ({v} ofertas)" for k,v in list(categoria_counts.items())[:3]])}. 
                El salario promedio tope se sitúa en un rango aproximado de <b>B/. {sal_promedio_max:,.2f}</b> mensuales. Las habilidades basadas en 
                desarrollo de software y computación en la nube son las mejor remuneradas en el territorio panameño.
            </p>
            <p style="line-height: 1.6;">
                <b>Brecha de Habilidades en la Academia:</b><br>
                La tecnología de punta más cotizada actualmente es <b>{top_skill}</b>, la cual está altamente correlacionada con arquitecturas distribuidas y análisis de datos masivos. 
                Los perfiles requeridos por las corporaciones en Ciudad de Panamá demandan habilidades complementarias en metodologías ágiles (Scrum/Agile) y control de versiones distribuido (Git). 
                Esto señala la urgencia de fortalecer los laboratorios de desarrollo y las metodologías de proyectos grupales dentro de la carrera.
            </p>
            <p style="line-height: 1.6;">
                <b>Recomendaciones Curriculares para la UTP:</b><br>
                1. <b>Integración Temprana:</b> Introducir el uso de Git y metodologías Scrum desde los primeros semestres académicos.<br>
                2. <b>Enfoque en Datos y Cloud:</b> Implementar electivas específicas en Arquitecturas Cloud (AWS/Azure) y Pipelines de Ingesta de Datos (Data Engineering), alineándose con las tecnologías emergentes identificadas en el modelo predictivo.<br>
                3. <b>Talleres de Habilidades Técnicas:</b> Habilitar bootcamps intensivos sobre tecnologías de frontend moderno como <b>React</b> y backend en <b>Python</b> o <b>Java (Spring Boot)</b>, minimizando la brecha entre los planes de estudio tradicionales y el ecosistema empresarial panameño.
            </p>
        </div>
        """, unsafe_allow_html=True)
