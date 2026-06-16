#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runner de ingesta REAL, rápido y aislado.

Usa una base de datos temporal propia para no chocar con otras ejecuciones,
desactiva el enriquecimiento de detalle (que es lento por las pausas de cortesía)
y al final reemplaza atómicamente los archivos finales del proyecto.
Salida sin buffer para ver el progreso en vivo.
"""
import os
import sys
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import pipeline  # noqa: E402

PROC = os.path.join(os.path.dirname(__file__), "data", "processed")
TMP_DB = os.path.join(PROC, "_tmp_real.db")
TMP_CSV = os.path.join(PROC, "_tmp_real.csv")

# Redirigir las rutas de persistencia del pipeline a archivos temporales aislados.
pipeline.DB_PATH = TMP_DB
pipeline.CSV_PATH = TMP_CSV
for p in (TMP_DB, TMP_CSV):
    if os.path.exists(p):
        os.remove(p)

print("== INGESTA REAL (rápida, sin enriquecimiento de detalle) ==", flush=True)
crudas = []
try:
    crudas += pipeline.scrape_portal_computrabajo(query="tecnologia", max_paginas=1,
                                                  enriquecer_detalle=False)
except Exception as e:
    print(f"[!] Computrabajo error: {e}", flush=True)
try:
    crudas += pipeline.fetch_jobs_arbeitnow_api(max_paginas=3)
except Exception as e:
    print(f"[!] Arbeitnow error: {e}", flush=True)

from collections import Counter
print(f"Crudas: {len(crudas)} -> {dict(Counter(v['portal'] for v in crudas))}", flush=True)

if not crudas:
    print("[!] Sin datos reales (sin red/bloqueo). Abortando para no sobrescribir con vacío.", flush=True)
    sys.exit(2)

procesadas = pipeline.estructurar_vacantes(crudas)
pipeline.guardar_en_db(procesadas)
pipeline.exportar_a_csv()

# Reemplazo atómico de los archivos finales del proyecto.
shutil.move(TMP_DB, os.path.join(PROC, "laboral_it.db"))
shutil.move(TMP_CSV, os.path.join(PROC, "vacantes_limpias.csv"))
print(f"[OK] {len(procesadas)} vacantes reales guardadas en laboral_it.db / vacantes_limpias.csv", flush=True)
