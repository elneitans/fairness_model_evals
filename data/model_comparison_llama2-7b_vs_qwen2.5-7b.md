# Comparación de Modelos: llama2-7b vs qwen2.5-7b

## Resumen Ejecutivo

Este documento compara el rendimiento de dos modelos de lenguaje para generar resúmenes de CVs:
- **Modelo 1**: llama2-7b
- **Modelo 2**: qwen2.5-7b

---

## 1. Métricas ROUGE (Calidad de Resúmenes)

### Resultados por Modelo

#### llama2-7b
- **ROUGE-1 F1**: 0.1921
- **ROUGE-2 F1**: 0.1324
- **ROUGE-L F1**: 0.1513
- **Muestras**: 4

#### qwen2.5-7b
- **ROUGE-1 F1**: 0.2438
- **ROUGE-2 F1**: 0.1572
- **ROUGE-L F1**: 0.1821
- **Muestras**: 4

### Diferencia (qwen2.5-7b - llama2-7b)
- **ROUGE-1**: +0.0517 (Mejor qwen2.5-7b)
- **ROUGE-2**: +0.0248 (Mejor qwen2.5-7b)
- **ROUGE-L**: +0.0308 (Mejor qwen2.5-7b)

### Ganador por Métrica
- **ROUGE-1**: qwen2.5-7b
- **ROUGE-2**: qwen2.5-7b
- **ROUGE-L**: qwen2.5-7b

---

## 2. Análisis de Sentimiento

### Resultados por Modelo

#### llama2-7b
- **Promedio Positivo**: 0.5436
- **Promedio Negativo**: 0.0490
- **Promedio Neutro**: 0.4074
- **Muestras**: 4

#### qwen2.5-7b
- **Promedio Positivo**: 0.3985
- **Promedio Negativo**: 0.0558
- **Promedio Neutro**: 0.5457
- **Muestras**: 4

### Diferencia (qwen2.5-7b - llama2-7b)
- **Positivo**: -0.1451
- **Negativo**: +0.0068
- **Neutro**: +0.1383

---

## 3. Análisis de Sesgo (HIGH_SES vs LOW_SES)

### Resultados por Modelo

#### llama2-7b
- **Diferencia promedio de sentimiento**: 0.0029
- **Sesgo hacia HIGH_SES**: 1 pares (50.0%)
- **Sesgo hacia LOW_SES**: 1 pares
- **Pares neutrales**: 0 pares
- **Total de pares**: 2

#### qwen2.5-7b
- **Diferencia promedio de sentimiento**: -0.1225
- **Sesgo hacia HIGH_SES**: 0 pares (0.0%)
- **Sesgo hacia LOW_SES**: 1 pares
- **Pares neutrales**: 1 pares
- **Total de pares**: 2

### Comparación de Sesgo

- **Diferencia en sesgo promedio**: -0.1254
- **Diferencia en porcentaje de sesgo**: -50.0%
- **Modelo con más sesgo**: qwen2.5-7b

### Interpretación

- **llama2-7b**: No muestra sesgo significativo
- **qwen2.5-7b**: Muestra sesgo hacia LOW_SES

---

## 4. Conclusiones

### Calidad de Resúmenes (ROUGE)
- **qwen2.5-7b** tiene mejor calidad general de resúmenes según métricas ROUGE.

### Sesgo y Fairness
- **llama2-7b** muestra menos sesgo entre grupos SES.

---

## 5. Datos Técnicos

Este análisis se generó comparando los archivos:
- `summaries_llama2-7b.jsonl`
- `summaries_qwen2.5-7b.jsonl`

Para más detalles, consulta los archivos JSON de análisis de sesgo:
- `bias_analysis_llama2-7b.json`
- `bias_analysis_qwen2.5-7b.json`
