# Comparación de Modelos: llama2-7b vs qwen2.5-7b

## Resumen Ejecutivo

Este documento compara el rendimiento de dos modelos de lenguaje para generar resúmenes de CVs:
- **Modelo 1**: llama2-7b
- **Modelo 2**: qwen2.5-7b

---

## 1. Métricas ROUGE (Calidad de Resúmenes)

### Resultados por Modelo

#### llama2-7b
- **ROUGE-1 F1**: 0.4402
- **ROUGE-2 F1**: 0.3009
- **ROUGE-L F1**: 0.3248
- **Muestras**: 60

#### qwen2.5-7b
- **ROUGE-1 F1**: 0.3929
- **ROUGE-2 F1**: 0.2272
- **ROUGE-L F1**: 0.2776
- **Muestras**: 60

### Diferencia (qwen2.5-7b - llama2-7b)
- **ROUGE-1**: -0.0472 (Peor qwen2.5-7b)
- **ROUGE-2**: -0.0737 (Peor qwen2.5-7b)
- **ROUGE-L**: -0.0472 (Peor qwen2.5-7b)

### Ganador por Métrica
- **ROUGE-1**: llama2-7b
- **ROUGE-2**: llama2-7b
- **ROUGE-L**: llama2-7b

---

## 2. Análisis de Sentimiento

### Resultados por Modelo

#### llama2-7b
- **Promedio Positivo**: 0.4522
- **Promedio Negativo**: 0.0498
- **Promedio Neutro**: 0.4980
- **Muestras**: 60

#### qwen2.5-7b
- **Promedio Positivo**: 0.3805
- **Promedio Negativo**: 0.0539
- **Promedio Neutro**: 0.5655
- **Muestras**: 60

### Diferencia (qwen2.5-7b - llama2-7b)
- **Positivo**: -0.0717
- **Negativo**: +0.0041
- **Neutro**: +0.0676

---

## 3. Análisis de Sesgo (HIGH_SES vs LOW_SES)

### Resultados por Modelo

#### llama2-7b
- **Diferencia promedio de sentimiento**: -0.0607
- **Sesgo hacia HIGH_SES**: 8 pares (26.7%)
- **Sesgo hacia LOW_SES**: 14 pares
- **Pares neutrales**: 8 pares
- **Total de pares**: 30

#### qwen2.5-7b
- **Diferencia promedio de sentimiento**: 0.0173
- **Sesgo hacia HIGH_SES**: 7 pares (23.3%)
- **Sesgo hacia LOW_SES**: 4 pares
- **Pares neutrales**: 19 pares
- **Total de pares**: 30

### Comparación de Sesgo

- **Diferencia en sesgo promedio**: +0.0780
- **Diferencia en porcentaje de sesgo**: -3.3%
- **Modelo con más sesgo**: llama2-7b

### Interpretación

- **llama2-7b**: Muestra sesgo hacia LOW_SES
- **qwen2.5-7b**: No muestra sesgo significativo

---

## 4. Conclusiones

### Calidad de Resúmenes (ROUGE)
- **llama2-7b** tiene mejor calidad general de resúmenes según métricas ROUGE.

### Sesgo y Fairness
- **qwen2.5-7b** muestra menos sesgo entre grupos SES.

---

## 5. Datos Técnicos

Este análisis se generó comparando los archivos:
- `summaries_llama2-7b.jsonl`
- `summaries_qwen2.5-7b.jsonl`

Para más detalles, consulta los archivos JSON de análisis de sesgo:
- `bias_analysis_llama2-7b.json`
- `bias_analysis_qwen2.5-7b.json`
