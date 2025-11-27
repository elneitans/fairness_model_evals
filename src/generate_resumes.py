"""
Pipeline 1: Generación de CVs sintéticos base (sin atributos sensibles).

Este módulo genera CVs para trabajadores sociales en Chile, sin incluir
nombres, RUTs, direcciones, comunas, emails ni teléfonos.
"""
import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from src.config import RAW_RESUMES_PATH, DATA_DIR


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


def call_llm(prompt: str) -> str:
    """
    Llama a un LLM (OpenAI/Anthropic) para generar el CV.
    
    Por ahora es un stub que devuelve un texto dummy para que el script sea ejecutable.
    TODO: Conectar con API de OpenAI o Anthropic.
    
    Args:
        prompt: Prompt para el LLM
        
    Returns:
        Texto del CV generado
    """
    # STUB: Por ahora devuelve un CV dummy realista
    # TODO: Implementar llamada real a LLM (OpenAI/Anthropic)
    raise NotImplementedError(
        "Esta función debe ser implementada para conectar con un LLM. "
        "Por ahora, usa generate_resumes con el modo de simulación."
    )


def _generate_dummy_resume(attrs: CandidateAttributes) -> str:
    """
    Genera un CV dummy realista para testing sin necesidad de LLM.
    
    Args:
        attrs: Atributos del candidato
        
    Returns:
        Texto del CV dummy
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
    use_dummy: bool = True
) -> None:
    """
    Genera n CVs sintéticos y los guarda en formato JSONL.
    
    Args:
        n: Número de CVs a generar
        output_path: Ruta donde guardar el archivo JSONL
        use_dummy: Si True, usa CVs dummy en vez de llamar al LLM (útil para testing)
    """
    # Asegurar que el directorio existe
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generar atributos de candidatos
    candidatos = sample_candidate_attributes(n)
    
    # Generar CVs y guardar en JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for candidato in candidatos:
            # Construir prompt
            prompt = build_resume_prompt(candidato)
            
            # Generar CV (dummy o real según use_dummy)
            if use_dummy:
                resume_text = _generate_dummy_resume(candidato)
            else:
                resume_text = call_llm(prompt)
            
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
        description="Genera CVs sintéticos base (sin atributos sensibles)"
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
        help="Usar LLM real en vez de CVs dummy (requiere implementar call_llm)"
    )
    
    args = parser.parse_args()
    
    generate_resumes(
        n=args.n,
        output_path=args.output,
        use_dummy=not args.use_llm
    )

