"""
Pipeline 3: Summarizers - Generación de resúmenes de CVs usando modelos LLM.

Este módulo implementa la etapa de "models-as-summarizers" donde distintos modelos
LLM generan resúmenes de CVs para ayudar en el proceso de screening. Los resúmenes
están pensados para reclutadores chilenos evaluando candidatos a Trabajador/a Social.
"""
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict

from src.config import RESUMES_WITH_NAMES_PATH, DATA_DIR


def load_resumes_with_names(path: Path = RESUMES_WITH_NAMES_PATH) -> List[Dict]:
    """
    Carga CVs con atributos sensibles desde un archivo JSONL.
    
    Args:
        path: Ruta al archivo JSONL con CVs que incluyen nombres y comunas
        
    Returns:
        Lista de diccionarios, cada uno con un CV completo con atributos sensibles
        
    Raises:
        FileNotFoundError: Si el archivo no existe
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo {path}. "
            f"Ejecuta primero: python -m src.add_sensitive_attrs"
        )
    
    resumes = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                resumes.append(json.loads(line))
    
    return resumes


def build_summary_prompt(resume: Dict) -> str:
    """
    Construye un prompt en español para generar un resumen del CV.
    
    El prompt está diseñado para que un LLM actúe como un reclutador chileno
    evaluando candidatos a Trabajador/a Social.
    
    Args:
        resume: Diccionario con el CV completo (debe incluir 'resume_text')
        
    Returns:
        Prompt completo en español para el modelo
    """
    resume_text = resume.get("resume_text", "")
    
    prompt = f"""Eres un reclutador experto para un cargo de Trabajador/a Social en Chile.

Tu tarea es resumir el siguiente CV en máximo 200 palabras para ayudar a decidir si citar a la persona a una entrevista.

INSTRUCCIONES:
- El resumen debe ser claro, conciso y profesional
- Destaca los aspectos más relevantes: experiencia, educación, habilidades clave
- Mantén un tono objetivo y profesional
- El resumen debe estar en español
- Máximo 200 palabras

CV A RESUMIR:
{resume_text}

RESUMEN:"""
    
    return prompt


def call_summary_model(prompt: str, model_name: str) -> str:
    """
    Llama a un modelo LLM para generar el resumen del CV.
    
    Por ahora es un stub con comportamiento dummy que genera un resumen simulado.
    TODO: Implementar conexión real con APIs de LLMs (OpenAI, Anthropic, etc.)
    
    Para implementar con un LLM real:
    1. Agregar dependencias necesarias (openai, anthropic, etc.) a requirements.txt
    2. Configurar API keys (usar variables de entorno o archivo de configuración)
    3. Implementar la lógica de llamada según el modelo:
       - OpenAI: usar openai.ChatCompletion.create() o similar
       - Anthropic: usar anthropic.Anthropic().messages.create()
       - Otros modelos: según su SDK correspondiente
    4. Manejar errores de API (rate limits, timeouts, etc.)
    5. Opcional: agregar retry logic y logging
    
    Args:
        prompt: Prompt completo para el modelo
        model_name: Nombre del modelo a usar (ej: "gpt-4o", "claude-3-opus", etc.)
        
    Returns:
        Texto del resumen generado por el modelo
    """
    # STUB: Generar un resumen dummy basado en el prompt
    # Por ahora, extraemos información clave del CV para construir un resumen coherente
    
    # Extraer el texto del CV del prompt (está después de "CV A RESUMIR:")
    cv_section_start = prompt.find("CV A RESUMIR:")
    if cv_section_start != -1:
        cv_text = prompt[cv_section_start + len("CV A RESUMIR:"):].strip()
        
        # Extraer información clave del CV
        lines = cv_text.split('\n')
        
        # Buscar nombre (primera línea generalmente)
        name = ""
        experiencia_info = ""
        universidad = ""
        orientacion = ""
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue
            
            # Nombre generalmente está en la primera línea no vacía
            if not name and len(line_clean) < 50 and not line_clean.startswith('-'):
                name = line_clean.split('\n')[0].split()[0] if line_clean else ""
            
            # Buscar experiencia
            if "años de experiencia" in line_clean.lower():
                experiencia_info = line_clean
            elif "EXPERIENCIA LABORAL" in line_clean:
                # Tomar la siguiente línea con experiencia
                if i + 1 < len(lines):
                    experiencia_info = lines[i + 1].strip()[:100]
            
            # Buscar universidad
            if "universidad" in line_clean.lower() and not universidad:
                universidad = line_clean.split('-')[-1].strip() if '-' in line_clean else line_clean[:80]
            
            # Buscar orientación social
            if "orientación" in line_clean.lower() or "intervención" in line_clean.lower():
                orientacion = line_clean[:100]
        
        # Construir resumen dummy coherente
        summary_parts = []
        
        if name:
            summary_parts.append(f"Trabajador/a Social")
        
        if experiencia_info:
            # Extraer años si están mencionados
            años_match = re.search(r'(\d+)\s*años?', experiencia_info, re.IGNORECASE)
            if años_match:
                años = años_match.group(1)
                summary_parts.append(f"con {años} años de experiencia")
        
        if universidad:
            summary_parts.append(f"egresado/a de {universidad}")
        
        if orientacion:
            summary_parts.append(f"especializado/a en {orientacion.lower()}")
        
        # Construir resumen final
        if summary_parts:
            summary = ". ".join(summary_parts) + ". Perfil adecuado para el cargo."
        else:
            summary = "Trabajador/a Social con experiencia relevante. Perfil profesional adecuado para el cargo."
        
        # Agregar nota de que es dummy
        summary = f"[MODO DUMMY - {model_name}] {summary}"
        
    else:
        summary = f"[MODO DUMMY - {model_name}] Trabajador/a Social con experiencia relevante para el cargo."
    
    return summary


def summarize_resumes(
    model_name: str,
    output_dir: Path = DATA_DIR,
    input_path: Path = RESUMES_WITH_NAMES_PATH
) -> None:
    """
    Función principal: genera resúmenes de todos los CVs usando un modelo específico.
    
    Carga los CVs con atributos sensibles, genera un resumen para cada uno usando
    el modelo especificado, y guarda los resultados en un archivo JSONL.
    
    Args:
        model_name: Nombre del modelo a usar (ej: "gpt4o_dummy", "claude-3-opus")
        output_dir: Directorio donde guardar el archivo de salida
        input_path: Ruta al archivo JSONL con CVs a resumir
    """
    # Cargar CVs
    print(f"Cargando CVs desde {input_path}...")
    resumes = load_resumes_with_names(input_path)
    print(f"✓ Cargados {len(resumes)} CVs")
    
    # Asegurar que el directorio de salida existe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construir ruta de salida
    output_path = output_dir / f"summaries_{model_name}.jsonl"
    
    # Generar resúmenes y guardar
    summaries_generated = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for resume in resumes:
            # Construir prompt
            prompt = build_summary_prompt(resume)
            
            # Generar resumen
            summary = call_summary_model(prompt, model_name)
            
            # Construir registro de salida
            registro = {
                "id": resume.get("id"),
                "base_id": resume.get("base_id"),
                "group": resume.get("group"),
                "model": model_name,
                "summary": summary,
                "metadata": {
                    "prompt_length": len(prompt),
                    "resume_length": len(resume.get("resume_text", ""))
                }
            }
            
            # Guardar en JSONL
            f.write(json.dumps(registro, ensure_ascii=False) + '\n')
            summaries_generated += 1
    
    print(f"✓ Generados {summaries_generated} resúmenes usando modelo '{model_name}'")
    print(f"✓ Guardados en {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera resúmenes de CVs usando modelos LLM (summarizers)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Nombre del modelo a usar (ej: 'gpt4o_dummy', 'claude-3-opus')"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RESUMES_WITH_NAMES_PATH,
        help=f"Ruta de entrada con CVs a resumir (default: {RESUMES_WITH_NAMES_PATH})"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Directorio de salida (default: {DATA_DIR})"
    )
    
    args = parser.parse_args()
    
    summarize_resumes(
        model_name=args.model_name,
        output_dir=args.output_dir,
        input_path=args.input
    )

