#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modulo: modelo.py
Descripción: Script de Machine Learning.
             1. Clustering (K-Means) de ofertas de empleo basado en habilidades
                técnicas, agrupándolas en perfiles profesionales de TI.
             2. Análisis de tendencias y predicción de habilidades emergentes
                aplicando Regresión Lineal sobre la frecuencia temporal de las tecnologías.
Autor: Grupo 4 - Gestión de la Información (Semestre I, 2026)
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import json

# Configuración de Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
DB_PATH = os.path.join(DATA_PROC_DIR, "laboral_it.db")
CSV_PATH = os.path.join(DATA_PROC_DIR, "vacantes_limpias.csv")

# Archivos de salida del modelo
MODEL_JOBS_PATH = os.path.join(DATA_PROC_DIR, "vacantes_modeladas.csv")
MODEL_TRENDS_PATH = os.path.join(DATA_PROC_DIR, "tendencias_skills.csv")


def cargar_datos() -> pd.DataFrame:
    """
    Carga el dataset de vacantes procesadas desde el CSV limpio.
    Si no existe, notifica que se debe ejecutar primero el pipeline.
    """
    if not os.path.exists(CSV_PATH):
        print("[!] No se encontró el archivo de datos limpios. Ejecutando pipeline primero...")
        from pipeline import ejecutar_pipeline
        ejecutar_pipeline(modo_simulado=True, num_simulados=200)
        
    df = pd.read_csv(CSV_PATH)
    # Rellenar nulos
    df["habilidades"] = df["habilidades"].fillna("")
    df["salario_min"] = df["salario_min"].fillna(df["salario_min"].median() if not df["salario_min"].isna().all() else 1500.0)
    df["salario_max"] = df["salario_max"].fillna(df["salario_max"].median() if not df["salario_max"].isna().all() else 2500.0)
    df["experiencia_anios"] = df["experiencia_anios"].fillna(2)
    return df


# =====================================================================
# 1. Clustering de Ofertas IT (K-Means + PCA)
# =====================================================================
def ejecutar_clustering(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Agrupa las ofertas en base a sus habilidades requeridas.
    Usa TF-IDF para vectorizar el listado de habilidades de cada vacante,
    luego K-Means para agruparlas, y PCA para visualización 2D.
    """
    print(f"[*] Iniciando K-Means Clustering con k={n_clusters} clusters...")
    
    # Preparar el "texto" de habilidades (reemplazar comas por espacios para el vectorizador)
    skills_text = df["habilidades"].str.replace(",", " ")
    
    # Vectorización TF-IDF
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[\w\.\#\-]+\b') # Capturar C#, .NET, C++
    tfidf_matrix = vectorizer.fit_transform(skills_text)
    
    # Ajustar Modelo K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(tfidf_matrix)
    
    # Reducción de dimensionalidad con PCA para visualización en 2D en el Dashboard
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(tfidf_matrix.toarray())
    df["pca_x"] = pca_coords[:, 0]
    df["pca_y"] = pca_coords[:, 1]
    
    # Identificar palabras/habilidades clave de cada cluster para nombrarlos
    terms = vectorizer.get_feature_names_out()
    cluster_labels = {}
    
    print("\n[+] Análisis de Centroides (Top Habilidades por Cluster):")
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    for i in range(n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :4]]
        print(f"    Cluster {i}: Claves principales -> {', '.join(top_terms)}")
        # Asignar un nombre sugerido al cluster según sus palabras clave dominantes
        if any(w in ["react", "html", "css", "angular", "vue", "javascript"] for w in [t.lower() for t in top_terms]):
            label = "Frontend / UI Web"
        elif any(w in ["python", "sql", "power", "bi", "tableau", "excel", "data"] for w in [t.lower() for t in top_terms]):
            label = "Data & Analytics / BI"
        elif any(w in ["aws", "docker", "kubernetes", "cloud", "linux", "azure"] for w in [t.lower() for t in top_terms]):
            label = "DevOps & Cloud Infrastructure"
        else:
            label = "Backend / Java / Core Systems"
        cluster_labels[i] = label
        
    df["nombre_cluster"] = df["cluster"].map(cluster_labels)
    print(f"[+] Clustering finalizado. Coordenadas PCA y etiquetas generadas.\n")
    return df


# =====================================================================
# 2. Predicción de Habilidades Emergentes (Regresión Lineal Temporal)
# =====================================================================
def predecir_habilidades_emergentes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa las ofertas por periodos quincenales (15 días) y calcula la
    frecuencia porcentual de menciones de cada tecnología.
    Ajusta una regresión lineal para calcular la pendiente (crecimiento/tendencia)
    y proyecta la demanda futura para estimar las habilidades emergentes.
    """
    print("[*] Ejecutando análisis de tendencias y predicción con Regresión Lineal...")
    
    # Convertir fecha a datetime
    df["fecha"] = pd.to_datetime(df["fecha_publicacion"])
    
    # Desglosar habilidades en filas individuales
    habilidades_desglosadas = []
    for idx, row in df.iterrows():
        skills = [s.strip() for s in row["habilidades"].split(",") if s.strip()]
        for skill in skills:
            habilidades_desglosadas.append({
                "id_vacante": row["id"],
                "fecha": row["fecha"],
                "habilidad": skill
            })
            
    df_skills = pd.DataFrame(habilidades_desglosadas)
    
    # Agrupar las fechas en periodos de 15 días (bi-semanal) para tener suficientes puntos de datos
    # Mapear cada fecha al índice del periodo (0 a N)
    fecha_min = df_skills["fecha"].min()
    df_skills["periodo_id"] = ((df_skills["fecha"] - fecha_min).dt.days // 15).astype(int)
    
    # Obtener el número total de vacantes por periodo quincenal en la base completa
    df["periodo_id"] = ((df["fecha"] - fecha_min).dt.days // 15).astype(int)
    vacantes_por_periodo = df.groupby("periodo_id").size().to_dict()
    
    # Frecuencia absoluta de habilidades por periodo
    frecuencia_periodo = df_skills.groupby(["habilidad", "periodo_id"]).size().reset_index(name="conteo")
    
    # Calcular porcentaje de presencia de la habilidad respecto al total de vacantes del periodo
    def calcular_pct(row):
        total_vacantes = vacantes_por_periodo.get(row["periodo_id"], 1)
        return (row["conteo"] / total_vacantes) * 100
        
    frecuencia_periodo["porcentaje"] = frecuencia_periodo.apply(calcular_pct, axis=1)
    
    # Lista de habilidades únicas con suficiente presencia (aparecer al menos 5 veces)
    habilidades_populares = df_skills["habilidad"].value_counts()[df_skills["habilidad"].value_counts() >= 5].index.tolist()
    
    tendencias = []
    max_periodo = df_skills["periodo_id"].max()
    
    for skill in habilidades_populares:
        # Filtrar datos de la habilidad
        datos_skill = frecuencia_periodo[frecuencia_periodo["habilidad"] == skill]
        
        # Rellenar con 0 los periodos donde la habilidad no apareció
        todos_los_periodos = pd.DataFrame({"periodo_id": range(max_periodo + 1)})
        datos_completos = pd.merge(todos_los_periodos, datos_skill, on="periodo_id", how="left")
        datos_completos["porcentaje"] = datos_completos["porcentaje"].fillna(0.0)
        datos_completos["habilidad"] = skill
        
        # Ajustar Regresión Lineal
        X = datos_completos["periodo_id"].values.reshape(-1, 1)
        y = datos_completos["porcentaje"].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        pendiente = model.coef_[0]  # La pendiente de la recta indica si sube o baja la demanda
        intercepto = model.intercept_
        
        # Predecir frecuencia para el próximo periodo (max_periodo + 1)
        proxima_demanda_pred = max(0.0, float(model.predict([[max_periodo + 1]])[0]))
        
        # Clasificar la tendencia
        if pendiente > 0.3:
            estado_tendencia = "Emergente / Crecimiento Rápido"
        elif pendiente > 0.05:
            estado_tendencia = "Crecimiento Estable"
        elif pendiente < -0.3:
            estado_tendencia = "En Declive"
        else:
            estado_tendencia = "Madura / Estable"
            
        tendencias.append({
            "habilidad": skill,
            "pendiente": float(pendiente),
            "intercepto": float(intercepto),
            "porcentaje_actual": float(y[-1]) if len(y) > 0 else 0.0,
            "porcentaje_predicho_futuro": proxima_demanda_pred,
            "tendencia": estado_tendencia
        })
        
    df_tendencias = pd.DataFrame(tendencias).sort_values(by="pendiente", ascending=False)
    
    print("\n[+] Top 5 Habilidades Emergentes Predictivas en Panamá:")
    for idx, row in df_tendencias.head(5).iterrows():
        print(f"    - {row['habilidad']}: Pendiente de Crecimiento = {row['pendiente']:.3f} | Estado: {row['tendencia']}")
        
    return df_tendencias


# =====================================================================
# Función Principal de Machine Learning
# =====================================================================
def ejecutar_modelado():
    """
    Ejecuta el pipeline de ciencia de datos completo:
    Carga de datos, agrupamiento K-Means, análisis y regresión temporal,
    y guardado de datasets de salida procesados para el Dashboard.
    """
    print("======================================================================")
    print("               EJECUTANDO MODELADO DE MACHINE LEARNING                ")
    print("======================================================================")
    
    # 1. Cargar Datos
    df = cargar_datos()
    
    # 2. K-Means Clustering
    df_clustered = ejecutar_clustering(df, n_clusters=4)
    df_clustered.to_csv(MODEL_JOBS_PATH, index=False, encoding="utf-8-sig")
    print(f"[+] Archivo guardado con clusters en: {MODEL_JOBS_PATH}")
    
    # 3. Regresión Temporal de Tendencias
    df_tendencias = predecir_habilidades_emergentes(df_clustered)
    df_tendencias.to_csv(MODEL_TRENDS_PATH, index=False, encoding="utf-8-sig")
    print(f"[+] Archivo de tendencias guardado en: {MODEL_TRENDS_PATH}")
    
    print("\n[+] Fase de Machine Learning completada exitosamente.")
    print("======================================================================\n")


if __name__ == "__main__":
    ejecutar_modelado()
