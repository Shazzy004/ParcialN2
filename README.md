# Análisis del Mercado Laboral IT en Panamá 🇵🇦

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Gemini](https://img.shields.io/badge/Google%20Gemini%20API-v2.5-4285F4?style=for-the-badge&logo=google-gemini&logoColor=white)](https://aistudio.google.com/)

---

### 🏫 Universidad Tecnológica de Panamá (UTP)
* **Facultad:** Facultad de Ingeniería de Sistemas Computacionales (FISC)
* **Carrera:** Licenciatura en Desarrollo y Gestión de Software
* **Curso:** Gestión de la Información (Semestre I, 2026)
* **Asignación:** Segundo Parcial — Pipeline + Visualización

### 👥 Integrantes (Grupo 4):
* **Bryan Law** — 8-1011-2459
* **Evaristo Alvarez** — 8-1011-177
* **Fernando Jimenez** — 20-24-7669
* **Manuel Campos** — 8-1022-1118
* **Diego Gordon** — 8-1017-349

---

## 📌 1. Introducción del Proyecto

Este proyecto consiste en el desarrollo de un **Pipeline de Datos Inteligente**, un **Modelo de Aprendizaje Automático (Machine Learning)** y un **Dashboard Interactivo** diseñado para analizar el ecosistema de empleo en tecnologías de la información (IT) en la República de Panamá. 

El sistema realiza web scraping automatizado de portales de empleo, estructura los datos usando un modelo de lenguaje masivo para extraer habilidades y salarios, aplica agrupamiento no supervisado (**K-Means**) para categorizar perfiles profesionales, modela el crecimiento de la demanda tecnológica con **Regresión Lineal** y expone todo a través de una interfaz interactiva de **Streamlit** adaptable a modo oscuro y claro.

---

## 🎯 2. Justificación del Problema en Panamá

El sector tecnológico en Panamá se encuentra en constante crecimiento debido a su posición como hub logístico y financiero. Sin embargo, existe una notable brecha entre las habilidades requeridas por las empresas y los contenidos académicos universitarios. 

Este proyecto provee a la UTP una herramienta basada en ciencia de datos que monitoriza las demandas del mercado a partir de **dos fuentes de datos diferentes**: web scraping del portal *Computrabajo Panamá* y la *API REST pública de Arbeitnow* (empleos IT remotos/internacionales). Esto ayuda a identificar habilidades emergentes en tiempo real antes de que queden obsoletas, optimizando el perfil de egreso de los estudiantes de la FISC.

---

## 📁 3. Estructura del Repositorio

El proyecto está organizado siguiendo las mejores prácticas de ingeniería de software para Ciencia de Datos:

```text
parcial_n2_grupo4/
├── data/
│   ├── raw/                        # Documentos crudos extraídos de scraping
│   └── processed/                  # Base de datos SQLite y CSV limpios (Pre-poblados)
├── src/
│   ├── __init__.py
│   ├── pipeline.py                 # Ingesta (2 fuentes) + estructuración con Gemini/heurística
│   ├── modelo.py                   # Modelado K-Means (Clustering) y Regresión Temporal
│   └── app.py                      # Dashboard de Streamlit con visualizaciones Plotly
├── tests/
│   └── test_pipeline.py            # Pruebas unitarias (pytest) deterministas y sin red
├── requirements.txt                # Dependencias necesarias
└── README.md                       # Documentación del proyecto (este archivo)
```

---

## 🚀 4. Guía de Instalación y Ejecución Rápida

*(Nota: Dado que la base de datos `laboral_it.db` y el CSV `vacantes_limpias.csv` ya están pre-poblados y guardados en el repositorio, **no es necesario correr el pipeline de scraping la primera vez**; puedes lanzar el Dashboard de inmediato).*

> ### 📊 Estrategia de datos (importante)
> El pipeline realiza **ingesta REAL de dos fuentes diferentes** (scraping de Computrabajo + API de Arbeitnow); una ejecución real reciente obtuvo **176 vacantes (20 de Computrabajo + 156 de Arbeitnow)**, cuya muestra se conserva como evidencia en `data/processed/vacantes_reales_muestra.csv`.
>
> Sin embargo, una captura en vivo de portales solo contiene ofertas de los últimos días, por lo que **no permite observar tendencias temporales de meses** (el modelo de habilidades emergentes requiere datos longitudinales). Por eso, el dataset **incluido para la demostración** del dashboard se genera con el generador sintético (`generate_panama_mock_data`), que distribuye las vacantes en ~6 meses y permite demostrar el modelo de Regresión Lineal y la predicción de habilidades emergentes. Este dataset es **claramente sintético y está etiquetado como tal**; no se falsean datos reales.
>
> Para regenerar datos **reales** en cualquier momento: `python src/pipeline.py` (modo por defecto). Para acumular un histórico real a lo largo del tiempo, ejecútalo periódicamente (el `INSERT` conserva lo anterior).

### Paso 1: Clonar el Repositorio e Ingresar
Abre la terminal de tu computadora y ejecuta:
```bash
git clone https://github.com/tu-usuario/analisis-mercado-laboral-it-panama.git
cd analisis-mercado-laboral-it-panama
```

### Paso 2: Crear el Entorno Virtual e Instalar Dependencias
* **En Windows (Símbolo del Sistema / CMD):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  pip install -r requirements.txt
  ```
* **En macOS / Linux (Terminal):**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

### Paso 3: Lanzar el Dashboard de Streamlit
Ejecuta el servidor web local:
```bash
streamlit run src/app.py
```
Una ventana de tu navegador predeterminada se abrirá automáticamente en la dirección: `http://localhost:8501`.

---

## ⚙️ 5. Cómo Funcionan los Componentes

### A. Ingesta y Estructuración (`pipeline.py`)

El pipeline implementa un flujo **ETL** clásico (Extract → Transform → Load) y por defecto ejecuta **ingesta REAL** de dos fuentes distintas:

1. **Fuente 1 — Web Scraping (HTML):** `scrape_portal_computrabajo()` usa `requests` + `BeautifulSoup` sobre *Computrabajo Panamá*. El diseño es **defensivo**: múltiples selectores de respaldo por campo (`data-id`, etiquetas semánticas, patrones de URL), paginación controlada, enriquecimiento opcional con la página de detalle y reintentos con *backoff* exponencial (`http_get`). Si el sitio cambia su HTML o bloquea, degrada con elegancia devolviendo lista vacía.
2. **Fuente 2 — API REST (JSON):** `fetch_jobs_arbeitnow_api()` consume la API pública de **Arbeitnow** (sin API key), filtra a roles IT y limpia las descripciones HTML. Aporta dos *paradigmas de ingesta diferentes* (scraping vs. API) y resiliencia: si una fuente falla, la otra suele seguir funcionando.
3. **Transformación con LLM:** cada vacante cruda pasa por `extract_info_with_gemini()`, que usa la API de **Gemini** para extraer un JSON estructurado (habilidades, salarios min/máx, experiencia, categoría). Si no hay `GEMINI_API_KEY`, cae automáticamente a un **parser heurístico local** (regex + diccionario de skills), por lo que el pipeline funciona aun sin clave.
4. **Carga:** persistencia en SQLite (`data/processed/laboral_it.db`) con modelo relacional (vacantes, habilidades y tabla intermedia `vacante_habilidad` de relación *muchos-a-muchos*), más exportación desnormalizada a CSV.
5. **Fallback de seguridad:** si **ninguna** fuente real devuelve datos (sin internet o bloqueo total), `generate_panama_mock_data()` genera un dataset sintético realista del mercado panameño para que la base nunca quede vacía. Esto queda registrado explícitamente en consola para no confundir datos reales con sintéticos.

### B. Machine Learning (`modelo.py`)
1. **K-Means Clustering:** Convierte las habilidades a una matriz numérica usando TF-IDF. Agrupa las ofertas en 4 clusters correspondientes a perfiles profesionales (e.g. *Frontend / UI Web*, *Data & Analytics*, *DevOps*, *Backend / Core Systems*). Aplica **PCA** para reducir las dimensiones a 2D para graficarlas.
2. **Regresión Lineal (Habilidades Emergentes):** Agrupa las vacantes en periodos quincenales, calcula la frecuencia porcentual de cada tecnología y ajusta una regresión lineal. La **pendiente ($m$)** determina la tasa de crecimiento quincenal de la habilidad, prediciendo cuáles tendrán mayor demanda a futuro.

---

## 🛠️ 6. Ejecución del Pipeline Completo (Opcional)

Para forzar una **extracción de datos nueva y real** (scraping de Computrabajo + API de Arbeitnow) y reentrenar los modelos, ejecuta con el entorno virtual activo:
```bash
python src/pipeline.py     # ingesta REAL por defecto; cae a sintético si no hay red
python src/modelo.py       # reentrena K-Means + Regresión sobre los datos nuevos
```
Para forzar explícitamente **datos sintéticos** (demo offline, sin red), abre un intérprete de Python y ejecuta:
```python
from src.pipeline import ejecutar_pipeline
ejecutar_pipeline(modo_simulado=True)
```
*(Nota: El dashboard de Streamlit también cuenta con un botón en la interfaz para ejecutar este paso automáticamente si deseas ver nuevos datos).*

---

## 🧪 7. Pruebas Unitarias

El proyecto incluye una suite de pruebas con **pytest** que valida la lógica crítica sin depender de internet ni de la API de Gemini (usa *mocks* y datos en memoria, por lo que es determinista y apta para CI):

```bash
pip install pytest
pytest -v
```

Cubre: el parser heurístico de habilidades/salarios, el mapeo de la API de Arbeitnow (filtrado IT, limpieza de HTML, manejo de fechas inválidas), la persistencia en SQLite con la relación muchos-a-muchos, y el generador de respaldo sintético.

---

## ❓ 8. Solución de Problemas Comunes (Troubleshooting)

### A. Pantalla "Welcome to Streamlit! Email:" en la terminal
Al ejecutar por primera vez `streamlit run src/app.py`, Streamlit te solicitará un correo electrónico para noticias.
* **Solución**: Simplemente presiona **Enter** en tu teclado (dejando el campo vacío) para continuar. El servidor local se iniciará de inmediato.

### B. Error al activar el entorno virtual en PowerShell (Windows)
Si al ejecutar `.\venv\Scripts\Activate.ps1` recibes un mensaje indicando que *"la ejecución de scripts está deshabilitada en este sistema"*:
* **Solución 1 (CMD)**: Utiliza el **Símbolo del Sistema (CMD)** estándar de Windows en lugar de PowerShell y ejecuta:
  ```cmd
  venv\Scripts\activate
  ```
* **Solución 2 (PowerShell)**: Abre una ventana de PowerShell como Administrador y ejecuta la siguiente política de permisos:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  ```

### C. Error 404 de API en la Generación de Conclusiones con IA
* **Solución**: Asegúrate de haber guardado tu API Key en el archivo `.env` o en los Secrets de Streamlit Cloud como: `GEMINI_API_KEY = "tu_clave"`. El sistema incluye un bucle de compatibilidad automático que buscará los modelos activos de Google AI Studio en 2026 (`gemini-2.5-flash`, `gemini-pro`, etc.) para evitar caídas del servicio.
