# Fairness Model Evals 🔬

Resumen y utilidades para evaluar modelos LLM como *summarizers* en una tubería de evaluación de fairness.

## Descripción

Este repositorio contiene una pipeline para generar resúmenes automáticos de CVs (Trabajador/a Social - Chile) con distintos modelos LLM, evaluar la calidad de los resúmenes con métricas ROUGE y analizar sesgos por sentimiento entre variantes HIGH_SES y LOW_SES.

- Script principal: `src/summarize_resumes.py`
- Scripts auxiliares para generar datos y añadir proxies sensibles: `src/add_sensitive_attrs.py`, `src/generate_resumes.py`

## Estructura del proyecto

```
LICENSE
README.md
requirements.txt
data/
  raw_resumes.jsonl
  resumes_with_names.jsonl
  summaries_gpt4o_dummy.jsonl
  summaries_llama2-7b.jsonl
notebooks/
  run_colab.ipynb
src/
  __init__.py
  add_sensitive_attrs.py
  config.py
  generate_resumes.py
  llm_models.py
  summarize_resumes.py
```

## Requisitos

- Python 3.10+ (se recomienda 3.11)
- Dependencias listadas en `requirements.txt` (instala con `pip install -r requirements.txt`)
- (Opcional) `pysentimiento` si quieres análisis de sentimiento y análisis de sesgo por grupo SES: `pip install pysentimiento`

## Uso

### 1) Preparar datos

Asegúrate de tener el archivo de entrada con CVs procesados por `add_sensitive_attrs` (por defecto `data/resumes_with_names.jsonl`). Si no existe, ejecútalo primero:

```bash
python -m src.add_sensitive_attrs --input data/raw_resumes.jsonl --output data/resumes_with_names.jsonl
```

### 2) Generar resúmenes para un modelo

Genera resúmenes con un modelo (modo real si está configurado, o **dummy** como fallback):

```bash
python -m src.summarize_resumes --model-name llama2-7b --output-dir data
```

Salida esperada:
- `data/summaries_llama2-7b.jsonl` — archivo JSONL con un registro por CV, incluyendo `summary` y `metadata`.
- `data/rouge_analysis_llama2-7b.json` — análisis por-resumen y estadísticas agregadas de ROUGE (ver formato abajo).
- `data/bias_analysis_llama2-7b.json` — (si `pysentimiento` está instalado) análisis de sesgo por sentimiento entre HIGH_SES y LOW_SES.

El mismo comando funciona para Qwen:

```bash
python -m src.summarize_resumes --model-name qwen2.5-7b --output-dir data
```

### 3) Comparar dos modelos

Una vez generados los archivos `summaries_{model}.jsonl` para ambos modelos, ejecuta la comparación:

```bash
python -m src.summarize_resumes --compare-models llama2-7b qwen2.5-7b --output-dir data
```

Salida esperada:
- `data/model_comparison_llama2-7b_vs_qwen2.5-7b.json` — resultados numéricos de la comparación (ROUGE, sentimiento, bias si disponible).
- `data/model_comparison_llama2-7b_vs_qwen2.5-7b.md` — informe legible en Markdown.

## Formato de outputs

### `summaries_{model}.jsonl` (por línea)

Cada línea es un JSON con campos principales:
- `id`, `base_id`, `group` (HIGH_SES/LOW_SES), `model`, `summary`
- `metadata`: contiene `prompt_length`, `resume_length`, `summary_length`, `rouge_scores`, `sentiment` (si disponible), `proxies_present`

Ejemplo (esquema):

```
{
  "id": "123",
  "base_id": "base-45",
  "group": "high_ses",
  "model": "llama2-7b",
  "summary": "...",
  "metadata": {
    "prompt_length": 1234,
    "resume_length": 2456,
    "summary_length": 180,
    "rouge_scores": {
      "rouge1": {"precision": 0.45, "recall": 0.6, "fmeasure": 0.51},
      "rouge2": {...},
      "rougeL": {...}
    },
    "sentiment": {"POS": 0.2, "NEG": 0.1, "NEU": 0.7, "label": "NEU"},
    "proxies_present": {"name": true, "comuna": true, ...}
  }
}
```

### `rouge_analysis_{model}.json`

- `model`: nombre del modelo
- `total_summaries`: número de resúmenes evaluados
- `per_summary`: lista con objetos por resumen:
  - `id`, `base_id`, `group`, `model`, `summary_length`, `resume_length`, `rouge_scores`
- `statistics`: agregados para las F-measures (`avg`, `median`, `min`, `max`) en `rouge1_f`, `rouge2_f`, `rougeL_f`

Ejemplo (esquema):

```
{
  "model": "llama2-7b",
  "total_summaries": 200,
  "per_summary": [ {"id": "123", "rouge_scores": {...}}, ... ],
  "statistics": {
    "rouge1_f": {"avg": 0.45, "median": 0.47, "min": 0.0, "max": 0.9},
    "rouge2_f": {...},
    "rougeL_f": {...}
  }
}
```

### `bias_analysis_{model}.json` (si disponible)

Resultado del análisis de sesgo por sentimiento entre pares HIGH_SES vs LOW_SES. Incluye comparaciones por `base_id` y estadísticas agregadas.

### `model_comparison_{model1}_vs_{model2}.json` / `.md`

Contiene:
- Resumen ejecutivo
- ROUGE promedio por modelo
- Comparación de sentimiento promedio
- Comparación de sesgo (si `bias_analysis` está disponible para ambos modelos)

## Consejos y notas 💡

- Si no puedes usar los modelos locales (Llama/Qwen), el script cae automáticamente a **modo dummy** y genera resúmenes sintéticos útiles para pruebas.
- Para reproducibilidad, usa siempre `--output-dir` y controla el `--input` si trabajas con subconjuntos.
- Instala `pysentimiento` si quieres análisis de sentimiento y el archivo `bias_analysis_{model}.json`.

## Preguntas frecuentes

- ¿Puedo evaluar otros modelos? ✅ Sí: pasa `--model-name <tu_modelo>` y adapta `src/llm_models.py` para integrar la llamada al modelo.
- ¿Qué significa una buena ROUGE? Depende del dominio; para resúmenes extractivos de CVs, ROUGE-1/ROUGE-L más altos indican mayor superposición con el texto original, pero no siempre reflejan calidad humana. Usa ROUGE junto con inspección manual.

---

Si quieres, puedo ejecutar un ejemplo en **modo dummy** para verificar la creación de `rouge_analysis_{model}.json` y mostrarte el archivo resultante. ¿Lo corro por ti? ✅
