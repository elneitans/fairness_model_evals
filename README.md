# Fairness Model Evals 

Pipeline de CV screening para la evaluación de bias y fairness en modelos de LLM abiertos. Optimizado para un contexto chileno.

## Descripción

Este repositorio contiene una pipeline para generar resúmenes automáticos de CVs (Analista Junior de Banca - Chile) con distintos modelos LLM, evaluar la calidad de los resúmenes con métricas ROUGE, analizar sesgos por sentimiento entre variantes HIGH_SES y LOW_SES, y estudiar decisiones de contratación con un modelo decisor externo.

- Script principal de resúmenes: `src/summarize_resumes.py`
- Scripts auxiliares para generar datos y añadir proxies de datos sensibles: `src/add_sensitive_attrs.py`, `src/generate_resumes.py`
- Script de decisión de contratación con DeepSeek: `src/decide_candidates.py`

## Estructura del proyecto

```
LICENSE
README.md
requirements.txt
data/
  raw_resumes.jsonl
  resumes_with_names.jsonl
  summaries_llama2-7b.jsonl
  summaries_qwen2.5-7b.jsonl
  summaries_llama3-8b-instruct.jsonl
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

## Uso

### 1) Preparar datos

Para generar mediante API de LLM, N CV's, usando CLI: 

    python -m src.generate_resumes --n N --use-llm --model deepseek-chat --api-key "tu_api_key_aquí"

Para crear dos CV's por cada uno de los N creados anteriormente, uno con información sensible correspondiente a proxies de índice socioeconómico chileno alto y bajo:

    python -m src.add_sensitive_attrs --proxy1 --proxy2
Donde proxy1 puede ser igual a "name", y proxy2 igual a "comuna"
    
```

### 2) Generar resúmenes para un modelo

Genera resúmenes con un modelo (modo real si está configurado, o **dummy** como fallback). Actualmente compatible con:
- **Llama 2 7B** (chat, local vía Hugging Face)
- **Qwen 2.5 7B Instruct** (local vía Hugging Face)
- **Meta-Llama-3.1-8B-Instruct** (local vía Hugging Face)

Ejemplos de uso:

```bash
python -m src.summarize_resumes --model-name llama2-7b
python -m src.summarize_resumes --model-name qwen2.5-7b
python -m src.summarize_resumes --model-name llama3-8b-instruct
```

También puedes usar directamente el identificador de Hugging Face para Meta Llama 3.1:

```bash
python -m src.summarize_resumes --model-name "meta-llama/Meta-Llama-3.1-8B-Instruct"
```


Salida esperada:
- `data/summaries_llama2-7b.jsonl` — archivo JSONL con un registro por CV, incluyendo `summary` y `metadata`.
- `data/rouge_analysis_llama2-7b.json` — análisis por-resumen y estadísticas agregadas de ROUGE (ver formato abajo).
- `data/bias_analysis_llama2-7b.json` — (si `pysentimiento` está instalado) análisis de sesgo por sentimiento entre HIGH_SES y LOW_SES.

### 3) Comparar dos modelos

Una vez generados los archivos `summaries_{model}.jsonl` para ambos modelos, ejecuta la comparación:

```bash
python -m src.summarize_resumes --compare-models llama2-7b qwen2.5-7b
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

### 4) Decisión de contratación con DeepSeek (`src/decide_candidates.py`)

Una vez que tienes los resúmenes generados (por ejemplo, para `llama2-7b` y `qwen2.5-7b`), puedes usar DeepSeek como **modelo decisor** para elegir entre pares HIGH_SES y LOW_SES basándose solo en los resúmenes:

```bash
python -m src.decide_candidates --models llama2-7b qwen2.5-7b --api-key "TU_DEEPSEEK_API_KEY"
```

- El script toma los archivos `data/summaries_{model}.jsonl`, agrupa por `base_id` y forma pares `high_ses` vs `low_ses`.
- Para cada par, construye un prompt y llama a DeepSeek (`deepseek-chat`) para decidir entre CANDIDATO_A y CANDIDATO_B, aleatorizando el orden para evitar sesgo de posición.
- El resultado agregado se guarda en `data/decisions_deepseek.json`, incluyendo cuántas veces se eligió HIGH_SES vs LOW_SES por modelo generador.

### Correr en Colab

Para correr en Colab, se recomienda utilizar fairness_CV.ipynb ubicado en el directorio notebooks, cuidando de rellenar correctamente sus api_keys. Es necesario subir todo el repositorio a Google Drive, para que pueda acceder a los documentos.

## Consejos y notas 💡

- Si no puedes usar los modelos locales (Llama/Qwen), el script cae automáticamente a **modo dummy** y genera resúmenes sintéticos útiles para pruebas.
- Para reproducibilidad, usa siempre `--output-dir` y controla el `--input` si trabajas con subconjuntos.
- Instala `pysentimiento` si quieres análisis de sentimiento y el archivo `bias_analysis_{model}.json`.
