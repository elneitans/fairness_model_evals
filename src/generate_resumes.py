"""
Pipeline 1: Generación de CVs sintéticos base (sin atributos sensibles).

Este módulo genera CVs para trabajadores sociales en Chile, sin incluir
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
    """Atributos de un candidato/a trabajador/a social."""
    id: str
    edad: int
    anos_experiencia: int
    universidad: str
    tipo_colegio: str
    orientacion_social: str
    nivel_manejo_conflictos: str
    especializacion: str


def sample_candidate_attributes(n: int) -> List[CandidateAttributes]:
    """
    Genera atributos aleatorios para n candidatos/as trabajadores/as sociales.
    
    Args:
        n: Número de candidatos a generar
        
    Returns:
        Lista de CandidateAttributes con valores razonables para el contexto chileno
    """
    #TODO: agregar más proxys (comunas, apellidos, género, etc)
    universidades = [
        "Universidad de Chile",
        "Pontificia Universidad Católica de Chile",
        "Universidad de Concepción",
        "Universidad Alberto Hurtado",
        "Universidad Central de Chile",
        "Universidad de Valparaíso",
        "Universidad de La Frontera"
    ]
    
    tipos_colegio = [
        "Municipal",
        "Particular Subvencionado",
        "Particular Pagado",
        "Corporación de Administración Delegada"
    ]
    
    orientaciones_sociales = [
        "Trabajo con familias en situación de vulnerabilidad",
        "Intervención comunitaria",
        "Salud mental y adicciones",
        "Infancia y adolescencia",
        "Adultos mayores",
        "Violencia intrafamiliar"
    ]
    
    niveles_conflictos = [
        "Alto",
        "Medio",
        "Bajo"
    ]
    
    especializaciones = [
        "Intervención familiar",
        "Políticas sociales",
        "Desarrollo comunitario",
        "Salud pública",
        "Educación social"
    ]
    
    candidatos = []
    for i in range(n):
        candidato = CandidateAttributes(
            id=f"cv_{i+1:04d}",
            edad=random.randint(25, 45),
            anos_experiencia=random.randint(1, 15),
            universidad=random.choice(universidades),
            tipo_colegio=random.choice(tipos_colegio),
            orientacion_social=random.choice(orientaciones_sociales),
            nivel_manejo_conflictos=random.choice(niveles_conflictos),
            especializacion=random.choice(especializaciones)
        )
        candidatos.append(candidato)
    
    return candidatos


def build_resume_prompt(attrs: CandidateAttributes) -> str:
    """
    Construye el prompt para generar un CV completo usando un LLM.
    
    El prompt está en español y solicita un CV para trabajador/a social en Chile,
    sin incluir información sensible (nombre, RUT, dirección, comuna, email, teléfono).
    
    Args:
        attrs: Atributos del candidato
        
    Returns:
        Prompt completo en español
    """
    prompt = f"""Genera un CV completo en texto plano para un/a Trabajador/a Social en Chile.

        INSTRUCCIONES IMPORTANTES:
        - NO incluyas nombre, RUT, dirección, comuna, email ni teléfono
        - El CV debe estar en español
        - Debe ser realista y coherente para el contexto laboral chileno
        - Incluye secciones típicas: perfil profesional, experiencia laboral, educación, habilidades

        ATRIBUTOS DEL CANDIDATO:
        - Edad: {attrs.edad} años
        - Años de experiencia: {attrs.anos_experiencia} años
        - Universidad: {attrs.universidad}
        - Tipo de colegio: {attrs.tipo_colegio}
        - Orientación social: {attrs.orientacion_social}
        - Nivel de manejo de conflictos: {attrs.nivel_manejo_conflictos}
        - Especialización: {attrs.especializacion}

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
                    "content": "Eres un asistente experto en generar CVs profesionales en español para trabajadores sociales en Chile."
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
        Trabajador/a Social con {attrs.anos_experiencia} años de experiencia en {attrs.orientacion_social.lower()}. 
        Especialización en {attrs.especializacion.lower()}. Nivel de manejo de conflictos: {attrs.nivel_manejo_conflictos.lower()}.

        EDUCACIÓN
        - Licenciatura en Trabajo Social, {attrs.universidad}
        - Educación secundaria: {attrs.tipo_colegio}

        EXPERIENCIA LABORAL
        """
    
    # Generar experiencia laboral basada en años de experiencia
    if attrs.anos_experiencia >= 10:
        cv_text += """- Trabajador/a Social Senior, Centro de Salud Familiar (CESFAM) - 5 años
          Responsable de intervención familiar y coordinación de programas comunitarios.

        - Trabajador/a Social, Municipalidad - 4 años
          Gestión de casos de vulnerabilidad social y coordinación con redes de apoyo.

        - Trabajador/a Social, ONG de desarrollo social - 1 año
          Implementación de programas de intervención comunitaria.
        """
    elif attrs.anos_experiencia >= 5:
        cv_text += """- Trabajador/a Social, Centro de Salud Familiar (CESFAM) - 3 años
          Intervención familiar y seguimiento de casos de vulnerabilidad.

        - Trabajador/a Social, Municipalidad - 2 años
          Gestión de casos y coordinación con servicios sociales.
        """
    else:
        cv_text += f"""- Trabajador/a Social, Centro de Salud Familiar (CESFAM) - {attrs.anos_experiencia} años
          Intervención familiar y seguimiento de casos de vulnerabilidad social.
        """
    
    cv_text += f"""
        HABILIDADES
        - Intervención en {attrs.orientacion_social.lower()}
        - Manejo de conflictos: nivel {attrs.nivel_manejo_conflictos.lower()}
        - Elaboración de informes sociales
        - Coordinación interinstitucional
        - Trabajo en equipo multidisciplinario

        CERTIFICACIONES
        - Registro en Colegio de Trabajadores Sociales de Chile
        - Certificación en intervención familiar
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
        description="Genera CVs sintéticos base (sin atributos sensibles) usando Deepseek o modo dummy"
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

