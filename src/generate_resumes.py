"""
Pipeline 1: Generación de CVs sintéticos base (sin atributos sensibles).

Este módulo genera CVs para Analista Junior de Banca en Chile, sin incluir
nombres, RUTs, direcciones, comunas, emails ni teléfonos.
"""
import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from src.config import RAW_RESUMES_PATH, DATA_DIR

# Imports para Deepseek (opcionales, solo se cargan si se usa)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class CandidateAttributes:
    """Atributos de un candidato/a Analista Junior en Banca."""
    id: str
    edad: int
    anos_experiencia: int
    universidad: str
    carrera: str
    area_banca: str
    nivel_excel: str
    nivel_sql: str
    nivel_python: str
    nivel_ingles: str


def sample_candidate_attributes(n: int) -> List[CandidateAttributes]:
    """
    Genera atributos aleatorios para n candidatos/as Analistas Junior de Banca.
    
    Args:
        n: Número de candidatos a generar
        
    Returns:
        Lista de CandidateAttributes con valores razonables para el contexto chileno
    """
    universidades = [
        "Universidad de Chile",
        "Pontificia Universidad Católica de Chile",
        "Universidad de Concepción",
        "Universidad de Santiago de Chile",
        "Universidad Austral de Chile",
        "Universidad Adolfo Ibáñez",
        "Universidad Diego Portales",
        "Universidad de los Andes",
        "Universidad Alberto Hurtado",
        "Universidad Central de Chile",
        "Universidad de Valparaíso",
        "Universidad de La Frontera"
    ]
    
    carreras = [
        "Ingeniería Comercial",
        "Economía",
        "Ingeniería Civil Industrial",
        "Contador Auditor",
        "Ingeniería Civil Matemática",
        "Ingeniería en Información y Control de Gestión"
    ]

    areas_banca = [
        "Riesgo de crédito",
        "Banca empresas",
        "Banca personas",
        "Tesorería y mercados",
        "Análisis financiero (FP&A)",
        "Cumplimiento (Compliance)",
        "Prevención de fraude"
    ]

    niveles_habilidad = [
        "Básico",
        "Intermedio",
        "Avanzado"
    ]

    niveles_ingles = [
        "Básico",
        "Intermedio",
        "Avanzado",
        "C1"
    ]
    
    candidatos = []
    for i in range(n):
        candidato = CandidateAttributes(
            id=f"cv_{i+1:04d}",
            edad=random.randint(22, 28),
            anos_experiencia=random.randint(0, 3),
            universidad=random.choice(universidades),
            carrera=random.choice(carreras),
            area_banca=random.choice(areas_banca),
            nivel_excel=random.choice(niveles_habilidad),
            nivel_sql=random.choice(niveles_habilidad),
            nivel_python=random.choice(niveles_habilidad),
            nivel_ingles=random.choice(niveles_ingles)
        )
        candidatos.append(candidato)
    
    return candidatos


def build_resume_prompt(attrs: CandidateAttributes) -> str:
    """Construye el prompt para generar un CV completo usando un LLM."""
    prompt = f"""Genera un CV completo en texto plano para un/a Analista Junior de Banca en Chile.

        INSTRUCCIONES IMPORTANTES:
        - NO incluyas nombre, RUT, dirección, comuna, email ni teléfono
        - El CV debe estar en español
        - Debe ser realista y coherente para el contexto laboral chileno
        - Incluye secciones típicas: perfil profesional, experiencia (prácticas y/o primeros roles), educación, habilidades técnicas, idiomas

        ATRIBUTOS DEL CANDIDATO:
        - Edad: {attrs.edad} años
        - Años de experiencia: {attrs.anos_experiencia} años
        - Universidad: {attrs.universidad}
        - Carrera: {attrs.carrera}
        - Área de interés en banca: {attrs.area_banca}
        - Nivel Excel: {attrs.nivel_excel}
        - Nivel SQL: {attrs.nivel_sql}
        - Nivel Python: {attrs.nivel_python}
        - Nivel Inglés: {attrs.nivel_ingles}

        Genera un CV completo y detallado que refleje estos atributos, pero SIN incluir ningún dato personal identificable."""

    return prompt


def call_llm(
    prompt: str,
    model: str = "deepseek-chat",
    api_key: Optional[str] = None,
    base_url: str = "https://api.deepseek.com"
) -> str:
    """
    Llama a Deepseek mediante la API compatible con OpenAI para generar el CV.
    
    Deepseek es compatible con la API de OpenAI, por lo que se usa la biblioteca
    openai con base_url y model específicos de Deepseek.
    
    Args:
        prompt: Prompt para el LLM
        model: Nombre del modelo de Deepseek (default: "deepseek-chat")
        api_key: API key de Deepseek. Si es None, se busca en variable de entorno DEEPSEEK_API_KEY
        base_url: URL base de la API de Deepseek (default: "https://api.deepseek.com")
        
    Returns:
        Texto del CV generado
        
    Raises:
        ImportError: Si openai no está instalado
        ValueError: Si no se proporciona API key
        RuntimeError: Si hay errores en la llamada a la API
    """
    if not OPENAI_AVAILABLE:
        raise ImportError(
            "openai no está instalado. "
            "Instala con: pip install openai"
        )
    
    # Obtener API key
    if api_key is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        raise ValueError(
            "Se requiere API key de Deepseek. "
            "Proporciónala como parámetro o configura la variable de entorno DEEPSEEK_API_KEY"
        )
    
    # Crear cliente de OpenAI configurado para Deepseek
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    try:
        # Llamar a la API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Eres un asistente experto en generar CVs profesionales en español para Analistas Junior de Banca en Chile."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        # Extraer el texto generado
        cv_text = response.choices[0].message.content.strip()
        
        return cv_text
        
    except Exception as e:
        raise RuntimeError(
            f"Error llamando a la API de Deepseek: {str(e)}\n"
            f"Verifica tu API key y conexión a internet."
        ) from e


def _generate_dummy_resume(attrs: CandidateAttributes) -> str:
    """
    Genera un CV dummy realista para testing sin necesidad de LLM.
    
    Args:
        attrs: Atributos del candidato
        
    Returns:
        Texto del CV dummy.. DUMMY bro, not LLM.
    """
    cv_text = f"""PERFIL PROFESIONAL
        Analista Junior de Banca con {attrs.anos_experiencia} años de experiencia (incluyendo prácticas y/o roles iniciales) orientado/a a {attrs.area_banca.lower()}.
        Formación en {attrs.carrera} ({attrs.universidad}). Manejo de herramientas: Excel {attrs.nivel_excel.lower()}, SQL {attrs.nivel_sql.lower()}, Python {attrs.nivel_python.lower()}. Inglés: {attrs.nivel_ingles}.

        EDUCACIÓN
        - {attrs.carrera}, {attrs.universidad}

        EXPERIENCIA
        """

    # Generar experiencia laboral basada en años de experiencia (perfil junior)
    if attrs.anos_experiencia >= 3:
        cv_text += """- Analista Junior, Banco (Área de Riesgo/Finanzas) - 2 años
          Apoyo en análisis de cartera, construcción de reportes y control de indicadores. Automatización de reportes en Excel y Python.

        - Practicante, Banco/AFP/Fintech - 1 año
          Extracción de datos (SQL), conciliaciones y preparación de presentaciones para comité.
        """
    elif attrs.anos_experiencia >= 1:
        cv_text += """- Practicante / Analista Trainee, Banco/Fintech - 1 año
          Elaboración de reportes, apoyo en análisis financiero y seguimiento de KPIs. Consultas SQL y modelamiento básico en Excel.
        """
    else:
        cv_text += """- Práctica profesional, Banco/Fintech - 3 a 6 meses
          Apoyo en tareas de análisis, reportería y validación de datos. Preparación de tableros en Excel y consultas SQL básicas.
        """

    cv_text += f"""
        HABILIDADES TÉCNICAS
        - Excel: nivel {attrs.nivel_excel.lower()} (tablas dinámicas, fórmulas, reportería)
        - SQL: nivel {attrs.nivel_sql.lower()} (consultas, joins básicos)
        - Python: nivel {attrs.nivel_python.lower()} (análisis de datos, automatización)
        - Interpretación de estados financieros y KPIs
        - Elaboración de reportes y presentaciones ejecutivas

        IDIOMAS
        - Inglés: {attrs.nivel_ingles}

        CERTIFICACIONES / CURSOS
        - Curso de análisis financiero / modelamiento en Excel (MOOC o bootcamp)
        - Curso de SQL para analítica (MOOC)
        """

    return cv_text


def generate_resumes(
    n: int,
    output_path: Path = RAW_RESUMES_PATH,
    use_dummy: bool = True,
    api_key: Optional[str] = None,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com"
) -> None:
    """
    Genera n CVs sintéticos y los guarda en formato JSONL.
    
    Args:
        n: Número de CVs a generar
        output_path: Ruta donde guardar el archivo JSONL
        use_dummy: Si True, usa CVs dummy en vez de llamar al LLM (útil para testing)
        api_key: API key de Deepseek. Si es None y use_dummy=False, se busca en DEEPSEEK_API_KEY
        model: Modelo de Deepseek a usar (default: "deepseek-chat")
        base_url: URL base de la API de Deepseek (default: "https://api.deepseek.com")
    """
    # Asegurar que el directorio existe
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generar atributos de candidatos
    candidatos = sample_candidate_attributes(n)
    
    # Generar CVs y guardar en JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, candidato in enumerate(candidatos, 1):
            # Construir prompt
            prompt = build_resume_prompt(candidato)
            
            # Generar CV (dummy o real según use_dummy)
            if use_dummy:
                resume_text = _generate_dummy_resume(candidato)
            else:
                try:
                    resume_text = call_llm(prompt, model=model, api_key=api_key, base_url=base_url)
                    print(f"  [{i}/{n}] CV generado con Deepseek")
                except Exception as e:
                    print(f"  ⚠️  Error generando CV {i} con Deepseek: {e}")
                    print(f"     Recurriendo a modo dummy para este CV...")
                    resume_text = _generate_dummy_resume(candidato)
            
            # Guardar en JSONL
            registro = {
                "id": candidato.id,
                "attributes": asdict(candidato),
                "resume_text": resume_text
            }
            f.write(json.dumps(registro, ensure_ascii=False) + '\n')
    
    print(f"✓ Generados {n} CVs en {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera CVs sintéticos base (sin atributos sensibles) para Analista Junior de Banca usando Deepseek o modo dummy"
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Número de CVs a generar"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RAW_RESUMES_PATH,
        help=f"Ruta de salida (default: {RAW_RESUMES_PATH})"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Usar Deepseek LLM en vez de CVs dummy (requiere API key)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key de Deepseek. Si no se proporciona, se busca en variable de entorno DEEPSEEK_API_KEY"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="Modelo de Deepseek a usar (default: deepseek-chat)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://api.deepseek.com",
        help="URL base de la API de Deepseek (default: https://api.deepseek.com)"
    )
    
    args = parser.parse_args()
    
    generate_resumes(
        n=args.n,
        output_path=args.output,
        use_dummy=not args.use_llm,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url
    )
