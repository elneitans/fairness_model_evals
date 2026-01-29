"""Visualization utilities for fairness_model_evals

Creates four figures:
1) Average ROUGE-1 recall by group (high_ses vs low_ses) per model (two bars per model)
2) Average POS-NEG (per-case composite) by group (high_ses vs low_ses) per model (two bars per model)
3) Bias counts by model: counts of bias towards HIGH_SES, LOW_SES, and no bias (three bars per model)
4) DeepSeek decisions by model: counts of HIGH_SES and LOW_SES candidates selected by DeepSeek (two bars per generator model)

Usage:
    python -m src.viz

Saves PNGs to `outputs/` by default.
"""
from pathlib import Path
import json
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

sns.set(style="whitegrid")


def _load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _load_jsonl(path: Path):
    """Load a JSONL file into a list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def avg_rouge1_by_group(models=None, data_dir=DATA_DIR):
    """Return dict[model] -> dict[group] -> mean rouge1 recall"""
    if models is None:
        models = ["llama2-7b", "qwen2.5-7b"]

    results = {}
    for model in models:
        path = data_dir / f"rouge_analysis_{model}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        data = _load_json(path)
        groups = defaultdict(list)
        for item in data.get("per_summary", []):
            group = item.get("group")
            rouge1 = item.get("rouge_scores", {}).get("rouge1", {}).get("recall")
            if rouge1 is not None and group is not None:
                groups[group].append(rouge1)
        results[model] = {g: float(np.mean(v)) if v else float('nan') for g, v in groups.items()}
    return results


def avg_composite_pos_minus_neg_by_group(models=None, data_dir=DATA_DIR):
    """Compute per-case composite (POS - NEG) for each group, then average per model.

    Returns dict[model] -> dict['high_ses'|'low_ses'] -> mean(composite_per_case)
    """
    if models is None:
        models = ["llama2-7b", "qwen2.5-7b"]

    results = {}
    for model in models:
        path = data_dir / f"bias_analysis_{model}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        data = _load_json(path)
        high_vals = []
        low_vals = []
        for comp in data.get("comparisons", []):
            high = comp.get("high_ses", {}).get("sentiment", {})
            low = comp.get("low_ses", {}).get("sentiment", {})
            if "POS" in high and "NEG" in high:
                high_vals.append(high["POS"] - high["NEG"])
            if "POS" in low and "NEG" in low:
                low_vals.append(low["POS"] - low["NEG"])
        results[model] = {
            "high_ses": float(np.mean(high_vals)) if high_vals else float('nan'),
            "low_ses": float(np.mean(low_vals)) if low_vals else float('nan'),
        }
    return results


def _grouped_bar_chart(data, models, groups, title, ylabel, out_path: Path, group_names=None):
    """Create a grouped bar chart.

    data: dict[model] -> dict[group] -> value
    models: list of model names (x-axis)
    groups: list of group keys in desired order (e.g., ['high_ses','low_ses'])
    group_names: optional list of display names for legend (same order as groups)
    """
    n_models = len(models)
    n_groups = len(groups)

    values = np.array([[data[m].get(g, np.nan) for g in groups] for m in models])

    x = np.arange(n_models)
    # Ajustar ancho dinámicamente según el número de grupos
    # Para 2 grupos: width = 0.35, para 3 grupos: width = 0.25
    width = max(0.2, min(0.4, 0.7 / n_groups))

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, g in enumerate(groups):
        label = group_names[i] if group_names is not None else g
        offset = (i - (n_groups - 1) / 2) * width
        ax.bar(x + offset, values[:, i], width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Group")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_rouge1_by_group(models=None, out_dir=OUTPUT_DIR):
    models = models or ["llama2-7b", "qwen2.5-7b"]
    data = avg_rouge1_by_group(models=models)
    out_path = Path(out_dir) / "rouge1_by_group.png"
    _grouped_bar_chart(
        data,
        models,
        groups=["high_ses", "low_ses"],
        title="Average ROUGE-1 Recall by SES Group",
        ylabel="ROUGE-1 Recall",
        out_path=out_path,
    )
    return out_path


def plot_composite_by_group(models=None, out_dir=OUTPUT_DIR):
    models = models or ["llama2-7b", "qwen2.5-7b"]
    data = avg_composite_pos_minus_neg_by_group(models=models)
    out_path = Path(out_dir) / "composite_pos_minus_neg_by_group.png"
    _grouped_bar_chart(
        data,
        models,
        groups=["high_ses", "low_ses"],
        title="Average (POS - NEG) by SES Group",
        ylabel="Average (POS - NEG)",
        out_path=out_path,
    )
    return out_path


def bias_counts_by_model(models=None, data_dir=DATA_DIR, threshold=0.1):
    """Return counts of bias outcomes per model.

    For each comparison case:
      - "bias_high": abs(composite_score) > threshold and composite_score > 0
      - "bias_low": abs(composite_score) > threshold and composite_score < 0
      - "no_bias": abs(composite_score) <= threshold OR composite_score is None

    Note: This function uses abs(composite_score) > threshold instead of relying
    on bias_detected, because bias_detected only detects bias when composite_score > 0.1,
    missing bias towards LOW_SES.

    Returns dict[model] -> dict['bias_high','bias_low','no_bias'] -> int
    """
    if models is None:
        models = ["llama2-7b", "qwen2.5-7b"]

    results = {}
    for model in models:
        path = data_dir / f"bias_analysis_{model}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        data = _load_json(path)
        high = 0
        low = 0
        none = 0
        for comp in data.get("comparisons", []):
            composite = comp.get("sentiment_difference", {}).get("composite_score")
            if composite is None:
                none += 1
            elif abs(composite) > threshold:
                # Detecta bias cuando el valor absoluto supera el umbral
                if composite > 0:
                    high += 1  # Bias hacia HIGH_SES
                else:
                    low += 1   # Bias hacia LOW_SES
            else:
                none += 1  # No hay bias significativo
        results[model] = {"bias_high": high, "bias_low": low, "no_bias": none}
    return results


def plot_bias_counts_by_model(models=None, out_dir=OUTPUT_DIR):
    models = models or ["llama2-7b", "qwen2.5-7b"]
    data = bias_counts_by_model(models=models)
    out_path = Path(out_dir) / "bias_counts_by_model.png"
    _grouped_bar_chart(
        data,
        models,
        groups=["bias_high", "bias_low", "no_bias"],
        group_names=["Bias (high SES)", "Bias (low SES)", "No bias"],
        title="Bias counts by model and SES",
        ylabel="Count",
        out_path=out_path,
    )
    return out_path


def deepseek_decisions_by_model(models=None, data_dir=DATA_DIR):
    """Return DeepSeek decision counts per generator model and SES group.
    
    Returns dict[generator_model] -> dict['high_ses_selected'|'low_ses_selected'] -> int
    """
    if models is None:
        models = ["llama2-7b", "qwen2.5-7b"]
    
    decisions_path = data_dir / "decisions_deepseek.json"
    if not decisions_path.exists():
        raise FileNotFoundError(
            f"Missing file: {decisions_path}. "
            f"Execute first: python -m src.decide_candidates"
        )
    
    data = _load_json(decisions_path)
    results = {}
    
    for model in models:
        model_result = data.get("results_by_model", {}).get(model, {})
        if "error" in model_result:
            results[model] = {"high_ses_selected": 0, "low_ses_selected": 0}
        else:
            results[model] = {
                "high_ses_selected": model_result.get("high_ses_selected", 0),
                "low_ses_selected": model_result.get("low_ses_selected", 0)
            }
    
    return results


def plot_deepseek_decisions_by_model(models=None, out_dir=OUTPUT_DIR):
    models = models or ["llama2-7b", "qwen2.5-7b"]
    data = deepseek_decisions_by_model(models=models)
    out_path = Path(out_dir) / "deepseek_decisions_by_model.png"
    _grouped_bar_chart(
        data,
        models,
        groups=["high_ses_selected", "low_ses_selected"],
        group_names=["High SES Selected", "Low SES Selected"],
        title="DeepSeek Decisions by Generator Model and SES Group",
        ylabel="Number of Candidates Selected",
        out_path=out_path,
    )
    return out_path


def calificacion_diff_stats(models=None, data_dir=DATA_DIR):
    """
    Para cada modelo (LLaMA / Qwen), calcula por cada CV tipo (base_id) la diferencia:
        calificacion_high_ses - calificacion_low_ses

    Luego imprime media y mediana por modelo.
    """
    if models is None:
        models = ["llama2-7b", "qwen2.5-7b"]

    print("\n📌 Calificación (High - Low) por modelo")

    for model in models:
        path = data_dir / f"summaries_{model}.jsonl"
        if not path.exists():
            print(f" - {model}: archivo no encontrado ({path})")
            continue

        rows = _load_jsonl(path)

        # base_id -> {group: calificacion}
        grouped = defaultdict(dict)
        for r in rows:
            base_id = r.get("base_id")
            group = r.get("group")
            cal = r.get("calificacion")
            if base_id and group and isinstance(cal, int):
                grouped[base_id][group] = cal

        diffs = []
        skipped = 0
        for base_id, groups in grouped.items():
            if "high_ses" in groups and "low_ses" in groups:
                diffs.append(groups["high_ses"] - groups["low_ses"])
            else:
                skipped += 1

        if not diffs:
            print(f" - {model}: no hay pares con calificación completa (high+low). Skipped base_ids={skipped}")
            continue

        mean_val = float(np.mean(diffs))
        median_val = float(np.median(diffs))
        print(f" - {model}: N_pares={len(diffs)} | media={mean_val:.3f} | mediana={median_val:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR), help="Output directory for figures")
    parser.add_argument("--models", nargs="*", default=None, help="List of models to include")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = args.models or ["llama2-7b", "qwen2.5-7b"]

    p1 = plot_rouge1_by_group(models=models, out_dir=out_dir)
    p2 = plot_composite_by_group(models=models, out_dir=out_dir)
    p3 = plot_bias_counts_by_model(models=models, out_dir=out_dir)
    
    # Intentar generar la visualización de DeepSeek (puede fallar si no existe el archivo)
    try:
        p4 = plot_deepseek_decisions_by_model(models=models, out_dir=out_dir)
        print(f"Saved figures:\n - {p1}\n - {p2}\n - {p3}\n - {p4}")
    except FileNotFoundError:
        print(f"Saved figures:\n - {p1}\n - {p2}\n - {p3}")
        print("Note: DeepSeek decisions plot skipped (run 'python -m src.decide_candidates' first)")

    # Imprimir estadísticas de calificación (High - Low) por modelo
    calificacion_diff_stats(models=models, data_dir=DATA_DIR)


if __name__ == "__main__":
    main()
