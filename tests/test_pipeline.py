#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pruebas unitarias del pipeline de ingesta y transformación.

Ejecutar desde la raíz del proyecto:
    pytest -v

Estas pruebas NO dependen de internet ni de la API de Gemini: validan la lógica
pura (parser heurístico, mapeo de la API, persistencia) usando datos en memoria.
Por eso son deterministas y rápidas, ideales para CI.
"""

import os
import sys
import sqlite3
import datetime

import pytest

# Permitir importar el paquete src/ sin instalar nada.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pipeline  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Parser heurístico (fallback sin Gemini)
# ---------------------------------------------------------------------------
def test_extractor_detecta_habilidades_y_salario(monkeypatch):
    """Sin API key, debe caer al heurístico y extraer skills, salario y experiencia."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # forzar el fallback
    desc = ("Buscamos Backend Developer con Python, SQL y Docker. "
            "Salario de $1500 a $2500. Mínimo 3 años de experiencia.")
    res = pipeline.extract_info_with_gemini("Backend Developer", desc)

    assert "Python" in res.habilidades_tecnicas
    assert "SQL" in res.habilidades_tecnicas
    assert "Docker" in res.habilidades_tecnicas
    assert res.salario_min == 1500.0
    assert res.salario_max == 2500.0
    assert res.experiencia_anios == 3
    assert res.categoria_rol in {
        "Frontend", "Backend", "Fullstack", "Data & Analytics",
        "Mobile", "DevOps & Cloud", "Soporte & IT", "Gestión & Agile",
    }


def test_extractor_sin_skills_usa_default_seguro(monkeypatch):
    """Una descripción sin tecnologías no debe romper: devuelve un default no vacío."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    res = pipeline.extract_info_with_gemini("Recepcionista", "Atención al cliente y orden.")
    assert len(res.habilidades_tecnicas) >= 1  # nunca lista vacía


# ---------------------------------------------------------------------------
# 2. Mapeo de la API de Arbeitnow (Fuente 2) sin tocar la red
# ---------------------------------------------------------------------------
class _FakeResp:
    """Respuesta HTTP simulada para inyectar en http_get."""
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_arbeitnow_filtra_solo_tech(monkeypatch):
    """La fuente API debe quedarse con roles IT y descartar los no técnicos."""
    fake_payload = {
        "data": [
            {
                "title": "Senior Python Engineer",
                "company_name": "TechCo",
                "description": "<p>We need <strong>Python</strong> and AWS.</p>",
                "tags": ["Backend"],
                "job_types": ["Full Time"],
                "created_at": int(datetime.datetime(2026, 5, 1).timestamp()),
            },
            {
                "title": "Marketing Manager",   # NO es IT -> debe descartarse
                "company_name": "BrandCo",
                "description": "<p>Social media and ads.</p>",
                "tags": ["Marketing"],
                "job_types": ["Full Time"],
                "created_at": int(datetime.datetime(2026, 5, 1).timestamp()),
            },
        ]
    }

    # Primera página devuelve datos; segunda devuelve vacío para cortar el bucle.
    paginas = iter([_FakeResp(fake_payload), _FakeResp({"data": []})])
    monkeypatch.setattr(pipeline, "http_get", lambda *a, **k: next(paginas))

    res = pipeline.fetch_jobs_arbeitnow_api(max_paginas=2)
    assert len(res) == 1
    vac = res[0]
    assert vac["titulo_original"] == "Senior Python Engineer"
    assert vac["portal"].startswith("Arbeitnow")
    assert "Python" in vac["descripcion"]          # HTML limpiado a texto
    assert "<" not in vac["descripcion"]            # sin etiquetas HTML
    assert vac["fecha_publicacion"] == "2026-05-01"


def test_arbeitnow_sin_red_devuelve_lista_vacia(monkeypatch):
    """Si http_get devuelve None (sin red), la fuente no debe lanzar excepción."""
    monkeypatch.setattr(pipeline, "http_get", lambda *a, **k: None)
    assert pipeline.fetch_jobs_arbeitnow_api(max_paginas=2) == []


# ---------------------------------------------------------------------------
# 3. Persistencia: SQLite + export CSV (Fuente -> Carga)
# ---------------------------------------------------------------------------
def test_guardar_y_exportar(tmp_path, monkeypatch):
    """Guardar en SQLite y exportar a CSV debe producir filas con la habilidad unida."""
    db = tmp_path / "test.db"
    csv = tmp_path / "test.csv"
    monkeypatch.setattr(pipeline, "DB_PATH", str(db))
    monkeypatch.setattr(pipeline, "CSV_PATH", str(csv))

    vacantes = [{
        "titulo_original": "Data Analyst",
        "empresa": "Banco General",
        "portal": "Computrabajo",
        "fecha_publicacion": "2026-06-01",
        "descripcion": "Análisis con Python y SQL.",
        "puesto": "Data Analyst",
        "habilidades_tecnicas": ["Python", "SQL"],
        "salario_min": 1500.0,
        "salario_max": 2500.0,
        "experiencia_anios": 2,
        "categoria_rol": "Data & Analytics",
    }]

    pipeline.guardar_en_db(vacantes)
    pipeline.exportar_a_csv()

    # La relación M:N debe haber creado 2 habilidades enlazadas a la vacante.
    conn = sqlite3.connect(str(db))
    n_vac = conn.execute("SELECT COUNT(*) FROM vacantes").fetchone()[0]
    n_skill = conn.execute("SELECT COUNT(*) FROM habilidades").fetchone()[0]
    conn.close()
    assert n_vac == 1
    assert n_skill == 2

    import pandas as pd
    df = pd.read_csv(str(csv))
    assert len(df) == 1
    assert "Python" in df.loc[0, "habilidades"]
    assert "SQL" in df.loc[0, "habilidades"]


# ---------------------------------------------------------------------------
# 4. Generador de respaldo sintético
# ---------------------------------------------------------------------------
def test_mock_data_estructura_y_volumen():
    datos = pipeline.generate_panama_mock_data(num_records=20)
    assert len(datos) == 20
    requeridos = {"titulo_original", "empresa", "portal", "fecha_publicacion",
                  "habilidades_tecnicas", "salario_min", "salario_max",
                  "experiencia_anios", "categoria_rol"}
    assert requeridos.issubset(datos[0].keys())
    # Salarios coherentes: el máximo siempre >= mínimo.
    assert all(d["salario_max"] >= d["salario_min"] for d in datos)
