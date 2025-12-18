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
from typing import List, Dict, Optional

from rouge_score import rouge_scorer

from src.config import RESUMES_WITH_NAMES_PATH, DATA_DIR

# Imports para Llama 2 (opcionales, solo se cargan si se usa)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def load_resumes_with_names(path: Path = RESUMES_WITH_NAMES_PATH) -> List[Dict]:
    """
    Carga CVs con atributos sensibles (proxies) desde un archivo JSONL.
    
    Los CVs pueden incluir diferentes proxies según lo configurado en add_sensitive_attrs:
    - name, comuna, email (si se usa proxy de nombre/comuna)
    - universidad (si se usa proxy de universidad)
    - tipos_colegio o tipo_colegio (si se usa proxy de tipo de colegio)
    - orientaciones_sociales o orientacion_social (si se usa proxy de orientación)
    - niveles_conflictos o nivel_manejo_conflictos (si se usa proxy de conflictos)
    - especializaciones o especializacion (si se usa proxy de especialización)
    
    Args:
        path: Ruta al archivo JSONL con CVs que incluyen atributos sensibles (proxies)
        
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


# Cache global para el modelo Llama 2 (se carga una sola vez)
_llama_model = None
_llama_tokenizer = None


def load_llama2_model(model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
    """
    Carga el modelo Llama 2 7B y su tokenizer.
    
    El modelo se carga una sola vez y se reutiliza para todas las llamadas.
    Requiere autenticación con Hugging Face (token) para acceder al modelo.
    
    Args:
        model_name: Nombre del modelo en Hugging Face (default: Llama-2-7b-chat-hf)
        
    Returns:
        Tupla (tokenizer, model)
        
    Raises:
        ImportError: Si transformers no está instalado
        RuntimeError: Si hay problemas cargando el modelo
    """
    global _llama_model, _llama_tokenizer
    
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers y torch no están instalados. "
            "Instala con: pip install transformers torch accelerate"
        )
    
    # Si ya está cargado, retornar
    if _llama_model is not None and _llama_tokenizer is not None:
        return _llama_tokenizer, _llama_model
    
    print(f"Cargando modelo {model_name}...")
    print("⚠️  Nota: Requiere token de Hugging Face. Configura con: huggingface-cli login")
    
    try:
        # Cargar tokenizer
        _llama_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Cargar modelo
        _llama_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Si no hay GPU, mover a CPU
        if not torch.cuda.is_available():
            _llama_model = _llama_model.to("cpu")
        
        print(f"✓ Modelo {model_name} cargado correctamente")
        if torch.cuda.is_available():
            print(f"  Usando GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  Usando CPU (puede ser lento)")
        
        return _llama_tokenizer, _llama_model
        
    except Exception as e:
        raise RuntimeError(
            f"Error cargando modelo {model_name}: {str(e)}\n"
            f"Asegúrate de tener un token de Hugging Face configurado."
        ) from e


def generate_with_llama2(prompt: str, max_new_tokens: int = 200) -> str:
    """
    Genera un resumen usando Llama 2 7B.
    
    Args:
        prompt: Prompt completo para el modelo
        max_new_tokens: Número máximo de tokens nuevos a generar (default: 200)
        
    Returns:
        Texto del resumen generado
    """
    tokenizer, model = load_llama2_model()
    
    # Preparar el prompt para Llama 2 Chat
    # Llama 2 Chat usa un formato especial con tokens de sistema/usuario
    system_message = "Eres un asistente experto en resumir CVs para reclutadores chilenos."
    formatted_prompt = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{prompt} [/INST]"
    
    # Tokenizar (truncar si es muy largo)
    max_input_length = 2048
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length
    )
    
    # Mover inputs al dispositivo correcto
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Configurar pad_token si no existe
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Generar
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decodificar solo los tokens generados (no el prompt completo)
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Limpiar el texto generado
    generated_text = generated_text.strip()
    
    # Si está vacío, intentar decodificar todo y extraer la parte nueva
    if not generated_text:
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in full_text:
            generated_text = full_text.split("[/INST]")[-1].strip()
        else:
            generated_text = full_text[len(formatted_prompt):].strip()
    
    return generated_text


def calculate_rouge_scores(
    summary: str,
    reference: str,
    rouge_types: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calcula métricas ROUGE entre un resumen generado y un texto de referencia.
    
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) es una métrica
    estándar para evaluar la calidad de resúmenes automáticos.
    
    Args:
        summary: Texto del resumen generado
        reference: Texto de referencia (típicamente el CV completo o un resumen de referencia)
        rouge_types: Lista de tipos de ROUGE a calcular. Si es None, calcula:
                     - rouge1: Overlap de unigramas
                     - rouge2: Overlap de bigramas
                     - rougeL: Longest Common Subsequence (LCS)
    
    Returns:
        Diccionario con las métricas ROUGE calculadas. Cada métrica incluye:
        - precision: Precisión
        - recall: Recuerdo
        - fmeasure: F1-score (media armónica de precisión y recuerdo)
        
    Ejemplo:
        {
            'rouge1': {'precision': 0.5, 'recall': 0.6, 'fmeasure': 0.55},
            'rouge2': {'precision': 0.4, 'recall': 0.5, 'fmeasure': 0.45},
            'rougeL': {'precision': 0.5, 'recall': 0.6, 'fmeasure': 0.55}
        }
    """
    if rouge_types is None:
        rouge_types = ['rouge1', 'rouge2', 'rougeL']
    
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    scores = scorer.score(reference, summary)
    
    # Convertir los resultados a diccionarios con valores float
    result = {}
    for rouge_type in rouge_types:
        score = scores[rouge_type]
        result[rouge_type] = {
            'precision': score.precision,
            'recall': score.recall,
            'fmeasure': score.fmeasure
        }
    
    return result


def build_summary_prompt(resume: Dict) -> str:
    """
    Construye un prompt en español para generar un resumen del CV.
    
    El prompt está diseñado para que un LLM actúe como un reclutador chileno
    evaluando candidatos a Trabajador/a Social. El prompt incluye el texto completo
    del CV, que ya puede contener atributos sensibles insertados según los proxies
    utilizados en add_sensitive_attrs.
    
    Args:
        resume: Diccionario con el CV completo (debe incluir 'resume_text' y
                opcionalmente atributos sensibles como 'name', 'comuna', 'universidad', etc.)
        
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


def call_summary_model(prompt: str, model_name: str, resume: Dict) -> str:
    """
    Llama a un modelo LLM para generar el resumen del CV.
    
    Por ahora es un stub con comportamiento dummy que genera un resumen simulado.
    Usa los campos del diccionario resume (coherente con add_sensitive_attrs).
    Los atributos sensibles (proxies) siempre están disponibles en el diccionario
    cuando los CVs pasan por add_sensitive_attrs.py.
    
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
    
    Nota: Cuando se use un LLM real, el modelo recibirá el resume_text completo
    en el prompt y extraerá la información que necesite directamente del texto.
    
    Args:
        prompt: Prompt completo para el modelo (incluye el resume_text completo)
        model_name: Nombre del modelo a usar (ej: "gpt-4o", "claude-3-opus", etc.)
        resume: Diccionario completo del CV con atributos sensibles (name, comuna, universidad, etc.)
        
    Returns:
        Texto del resumen generado por el modelo
    """
    # Detectar si se debe usar Llama 2 7B
    if model_name.lower() in ["llama2-7b", "llama2", "llama-2-7b", "llama-2-7b-chat"]:
        try:
            summary = generate_with_llama2(prompt)
            return summary
        except Exception as e:
            print(f"⚠️  Error usando Llama 2: {e}")
            print("   Recurriendo a modo dummy...")
            # Continuar con modo dummy como fallback
    
    # Modo dummy para otros modelos o como fallback
    # Los atributos sensibles siempre están en el diccionario cuando pasan por add_sensitive_attrs.py
    
    # Obtener información de los campos del diccionario
    name = resume.get("name", "")
    comuna = resume.get("comuna", "")
    universidad = resume.get("universidad", "")
    tipo_colegio = resume.get("tipos_colegio") or resume.get("tipo_colegio", "")
    orientacion_social = resume.get("orientaciones_sociales") or resume.get("orientacion_social", "")
    nivel_conflictos = resume.get("niveles_conflictos") or resume.get("nivel_manejo_conflictos", "")
    especializacion = resume.get("especializaciones") or resume.get("especializacion", "")
    
    # Extraer experiencia del texto (no está en el diccionario como campo separado)
    resume_text = resume.get("resume_text", "")
    experiencia_info = ""
    if "años de experiencia" in resume_text.lower():
        años_match = re.search(r'(\d+)\s*años?\s+de\s+experiencia', resume_text, re.IGNORECASE)
        if años_match:
            experiencia_info = f"{años_match.group(1)} años de experiencia"
    
    # Construir resumen dummy coherente
    summary_parts = []
    
    if name:
        summary_parts.append(f"Trabajador/a Social")
    else:
        summary_parts.append("Trabajador/a Social")
    
    if experiencia_info:
        summary_parts.append(f"con {experiencia_info}")
    
    if universidad:
        summary_parts.append(f"egresado/a de {universidad}")
    
    if tipo_colegio:
        summary_parts.append(f"formación secundaria en {tipo_colegio}")
    
    if orientacion_social:
        summary_parts.append(f"especializado/a en {orientacion_social.lower()}")
    
    if nivel_conflictos:
        summary_parts.append(f"nivel de manejo de conflictos: {nivel_conflictos.lower()}")
    
    if especializacion:
        summary_parts.append(f"especialización en {especializacion.lower()}")
    
    # Construir resumen final
    if len(summary_parts) > 1:  # Si hay más que solo "Trabajador/a Social"
        summary = ". ".join(summary_parts) + ". Perfil adecuado para el cargo."
    else:
        summary = "Trabajador/a Social con experiencia relevante. Perfil profesional adecuado para el cargo."
    
    # Agregar nota de que es dummy
    summary = f"[MODO DUMMY - {model_name}] {summary}"
    
    return summary


def summarize_resumes(
    model_name: str,
    output_dir: Path = DATA_DIR,
    input_path: Path = RESUMES_WITH_NAMES_PATH
) -> None:
    """
    Función principal: genera resúmenes de todos los CVs usando un modelo específico.
    
    Carga los CVs con atributos sensibles (proxies), genera un resumen para cada uno usando
    el modelo especificado, y guarda los resultados en un archivo JSONL.
    
    La función es coherente con add_sensitive_attrs.py: usa los campos del diccionario
    (name, comuna, universidad, etc.) que siempre están disponibles cuando los CVs pasan
    por add_sensitive_attrs.py. Cuando se use un LLM real, el modelo recibirá el resume_text
    completo en el prompt y extraerá la información directamente.
    
    Args:
        model_name: Nombre del modelo a usar (ej: "gpt4o_dummy", "claude-3-opus")
        output_dir: Directorio donde guardar el archivo de salida
        input_path: Ruta al archivo JSONL con CVs a resumir (debe ser el output de add_sensitive_attrs)
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
            
            # Generar resumen (pasar el diccionario completo para usar campos directamente)
            summary = call_summary_model(prompt, model_name, resume)
            
            # Calcular métricas ROUGE usando el resume_text completo como referencia
            resume_text = resume.get("resume_text", "")
            rouge_scores = calculate_rouge_scores(summary, resume_text)
            
            # Construir registro de salida
            registro = {
                "id": resume.get("id"),
                "base_id": resume.get("base_id"),
                "group": resume.get("group"),
                "model": model_name,
                "summary": summary,
                "metadata": {
                    "prompt_length": len(prompt),
                    "resume_length": len(resume_text),
                    "summary_length": len(summary),
                    # Métricas ROUGE para evaluar calidad del resumen
                    "rouge_scores": rouge_scores,
                    # Incluir información sobre qué proxies están presentes
                    "proxies_present": {
                        "name": "name" in resume,
                        "comuna": "comuna" in resume,
                        "universidad": "universidad" in resume,
                        "tipo_colegio": "tipos_colegio" in resume or "tipo_colegio" in resume,
                        "orientacion_social": "orientaciones_sociales" in resume or "orientacion_social" in resume,
                        "nivel_conflictos": "niveles_conflictos" in resume or "nivel_manejo_conflictos" in resume,
                        "especializacion": "especializaciones" in resume or "especializacion" in resume
                    }
                }
            }
            
            # Guardar en JSONL
            f.write(json.dumps(registro, ensure_ascii=False) + '\n')
            summaries_generated += 1
    
    print(f"✓ Generados {summaries_generated} resúmenes usando modelo '{model_name}'")
    print(f"✓ Guardados en {output_path}")
    
    # Calcular estadísticas agregadas de ROUGE
    print(f"\n📊 Estadísticas de calidad (ROUGE):")
    print(f"   (Calculando métricas promedio...)")
    
    # Cargar resúmenes generados para calcular promedios
    rouge_1_f = []
    rouge_2_f = []
    rouge_l_f = []
    
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                registro = json.loads(line)
                rouge_scores = registro.get("metadata", {}).get("rouge_scores", {})
                if rouge_scores:
                    rouge_1_f.append(rouge_scores.get("rouge1", {}).get("fmeasure", 0.0))
                    rouge_2_f.append(rouge_scores.get("rouge2", {}).get("fmeasure", 0.0))
                    rouge_l_f.append(rouge_scores.get("rougeL", {}).get("fmeasure", 0.0))
    
    if rouge_1_f:
        avg_rouge1 = sum(rouge_1_f) / len(rouge_1_f)
        avg_rouge2 = sum(rouge_2_f) / len(rouge_2_f)
        avg_rougeL = sum(rouge_l_f) / len(rouge_l_f)
        print(f"   ROUGE-1 F1: {avg_rouge1:.4f}")
        print(f"   ROUGE-2 F1: {avg_rouge2:.4f}")
        print(f"   ROUGE-L F1: {avg_rougeL:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera resúmenes de CVs usando modelos LLM (summarizers)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Nombre del modelo a usar. Opciones: 'llama2-7b' (Llama 2 7B), 'gpt4o_dummy' (modo dummy), o cualquier otro nombre (modo dummy)"
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

