"""
Pipeline 2: Agregar atributos sensibles (proxies) a CVs base.

Este módulo toma CVs sin atributos sensibles y crea variantes con diferentes
proxies (nombre, comuna, universidad, tipo de colegio, etc.) asociados a grupos
socioeconómicos altos (HIGH_SES) y bajos (LOW_SES). Los proxies a incluir se
especifican mediante parámetros booleanos.
"""
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

from src.config import (
    RAW_RESUMES_PATH,
    RESUMES_WITH_NAMES_PATH,
    SESGroup,
    HIGH_SES_NAMES,
    HIGH_SES_COMUNAS,
    LOW_SES_NAMES,
    LOW_SES_COMUNAS,
    HIGH_SES_UNIVERSIDADES,
    LOW_SES_UNIVERSIDADES,
    HIGH_SES_TIPOS_COLEGIO,
    LOW_SES_TIPOS_COLEGIO,
    ORIENTACIONES_SOCIALES,
    NIVELES_CONFLICTOS,
    ESPECIALIZACIONES
)


def load_raw_resumes(path: Path = RAW_RESUMES_PATH) -> List[Dict]:
    """
    Carga CVs base desde un archivo JSONL.
    
    Args:
        path: Ruta al archivo JSONL con CVs base
        
    Returns:
        Lista de diccionarios, cada uno con un CV base
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo {path}. "
            f"Ejecuta primero: python -m src.generate_resumes --n <número>"
        )
    
    resumes = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                resumes.append(json.loads(line))
    
    return resumes

def assign_sensitive_attrs(base_id: str, group: SESGroup, name:bool, comuna:bool, universidad:bool, tipos_colegio:bool, orientaciones_sociales:bool, niveles_conflictos:bool, especializaciones:bool) -> Dict[str, str]:
    """
    Asigna atributos sensibles según el grupo socioeconómico.
    Solo incluye los atributos cuyo parámetro correspondiente es True.
    
    Args:
        base_id: ID del CV base
        group: Grupo socioeconómico (HIGH_SES o LOW_SES)
        name: Si True, incluye nombre
        comuna: Si True, incluye comuna
        universidad: Si True, incluye universidad
        tipos_colegio: Si True, incluye tipo de colegio
        orientaciones_sociales: Si True, incluye orientación social
        niveles_conflictos: Si True, incluye nivel de manejo de conflictos
        especializaciones: Si True, incluye especialización
        
    Returns:
        Diccionario solo con los atributos solicitados (cuyo parámetro es True)
    """
    attrs = {}
    
    if name:
        if group == SESGroup.HIGH_SES:
            name_value = random.choice(HIGH_SES_NAMES)
        elif group == SESGroup.LOW_SES:
            name_value = random.choice(LOW_SES_NAMES)
        else:
            raise ValueError(f"Grupo no válido: {group}")
        
        attrs["name"] = name_value
        
        # Generar email derivado del nombre (formato simple)
        nombre_parts = name_value.lower().split()
        apellido = nombre_parts[-1] if len(nombre_parts) > 1 else nombre_parts[0]
        # Remover acentos y caracteres especiales para email
        apellido_clean = apellido.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
        attrs["email"] = f"{nombre_parts[0].lower()}.{apellido_clean}@gmail.com"
    
    if comuna:
        if group == SESGroup.HIGH_SES:
            attrs["comuna"] = random.choice(HIGH_SES_COMUNAS)
        elif group == SESGroup.LOW_SES:
            attrs["comuna"] = random.choice(LOW_SES_COMUNAS)
        else:
            raise ValueError(f"Grupo no válido: {group}")
    
    if universidad:
        if group == SESGroup.HIGH_SES:
            attrs["universidad"] = random.choice(HIGH_SES_UNIVERSIDADES)
        elif group == SESGroup.LOW_SES:
            attrs["universidad"] = random.choice(LOW_SES_UNIVERSIDADES)
        else:
            raise ValueError(f"Grupo no válido: {group}")
    
    if tipos_colegio:
        if group == SESGroup.HIGH_SES:
            attrs["tipos_colegio"] = random.choice(HIGH_SES_TIPOS_COLEGIO)
        elif group == SESGroup.LOW_SES:
            attrs["tipos_colegio"] = random.choice(LOW_SES_TIPOS_COLEGIO)
        else:
            raise ValueError(f"Grupo no válido: {group}")
    
    if orientaciones_sociales:
        # Las orientaciones sociales no se diferencian por SES, se asignan aleatoriamente
        attrs["orientaciones_sociales"] = random.choice(ORIENTACIONES_SOCIALES)
    
    if niveles_conflictos:
        # Los niveles de conflictos no se diferencian por SES, se asignan aleatoriamente
        attrs["niveles_conflictos"] = random.choice(NIVELES_CONFLICTOS)
    
    if especializaciones:
        # Las especializaciones no se diferencian por SES, se asignan aleatoriamente
        attrs["especializaciones"] = random.choice(ESPECIALIZACIONES)
    
    return attrs

def insert_attrs_in_resume(resume_text: str, attrs: Dict[str, str]) -> str:
    """
    Inserta atributos sensibles en el texto del CV de forma coherente.
    
    Inserta solo los atributos que están presentes en el diccionario attrs.
    - Si hay nombre y/o comuna, crea un encabezado profesional
    - Otros atributos (universidad, tipos_colegio, etc.) se insertan en secciones
      relevantes del CV o en un encabezado de información adicional.
    
    Args:
        resume_text: Texto original del CV
        attrs: Diccionario con atributos sensibles a insertar (name, comuna, universidad, etc.)
        
    Returns:
        Texto del CV con atributos insertados
    """
    header_parts = []
    
    # Construir encabezado con nombre si está presente
    if "name" in attrs:
        header_parts.append(attrs['name'].upper())
        header_parts.append("Trabajador/a Social")
    
    # Agregar comuna si está presente
    if "comuna" in attrs:
        if header_parts:
            header_parts.append(f"{attrs['comuna']}, Región Metropolitana")
        else:
            # Si no hay nombre, crear encabezado solo con comuna
            header_parts.append("Trabajador/a Social")
            header_parts.append(f"{attrs['comuna']}, Región Metropolitana")
    
    # Construir sección de información adicional para otros atributos
    additional_info = []
    
    if "universidad" in attrs:
        additional_info.append(f"Universidad: {attrs['universidad']}")
    
    if "tipos_colegio" in attrs or "tipo_colegio" in attrs:
        tipo_colegio = attrs.get("tipos_colegio") or attrs.get("tipo_colegio")
        additional_info.append(f"Tipo de colegio: {tipo_colegio}")
    
    if "orientaciones_sociales" in attrs or "orientacion_social" in attrs:
        orientacion = attrs.get("orientaciones_sociales") or attrs.get("orientacion_social")
        additional_info.append(f"Orientación social: {orientacion}")
    
    if "niveles_conflictos" in attrs or "nivel_manejo_conflictos" in attrs:
        nivel = attrs.get("niveles_conflictos") or attrs.get("nivel_manejo_conflictos")
        additional_info.append(f"Nivel de manejo de conflictos: {nivel}")
    
    if "especializaciones" in attrs or "especializacion" in attrs:
        especializacion = attrs.get("especializaciones") or attrs.get("especializacion")
        additional_info.append(f"Especialización: {especializacion}")
    
    # Construir el texto final
    result_parts = []
    
    # Agregar encabezado si hay nombre o comuna
    if header_parts:
        result_parts.append("\n".join(header_parts))
        result_parts.append("")  # Línea en blanco
    
    # Agregar información adicional si hay otros atributos
    if additional_info:
        if not header_parts:
            # Si no hay encabezado, crear uno básico
            result_parts.append("Trabajador/a Social")
            result_parts.append("")
        result_parts.append("INFORMACIÓN ADICIONAL")
        result_parts.extend(additional_info)
        result_parts.append("")  # Línea en blanco
    
    # Agregar el texto original del CV
    if result_parts:
        return "\n".join(result_parts) + "\n" + resume_text
    
    # Si no hay atributos que modifiquen el texto, retornar original
    return resume_text


def create_group_variants(
    resume: Dict,
    name: bool = False,
    comuna: bool = False,
    universidad: bool = False,
    tipos_colegio: bool = False,
    orientaciones_sociales: bool = False,
    niveles_conflictos: bool = False,
    especializaciones: bool = False
) -> List[Dict]:
    """
    Crea dos variantes de un CV base: una con atributos HIGH_SES y otra LOW_SES.
    Solo incluye los atributos (proxies) indicados como True en los parámetros.
    
    Args:
        resume: Diccionario con el CV base (debe tener 'id', 'attributes', 'resume_text')
        name: Si True, incluye nombre en las variantes
        comuna: Si True, incluye comuna en las variantes
        universidad: Si True, incluye universidad en las variantes
        tipos_colegio: Si True, incluye tipo de colegio en las variantes
        orientaciones_sociales: Si True, incluye orientación social en las variantes
        niveles_conflictos: Si True, incluye nivel de manejo de conflictos en las variantes
        especializaciones: Si True, incluye especialización en las variantes
        
    Returns:
        Lista con dos diccionarios: uno para HIGH_SES y otro para LOW_SES
    """
    base_id = resume["id"]
    base_resume_text = resume["resume_text"]
    
    variants = []
    
    # Variante HIGH_SES
    high_attrs = assign_sensitive_attrs(
        base_id, 
        SESGroup.HIGH_SES,
        name=name,
        comuna=comuna,
        universidad=universidad,
        tipos_colegio=tipos_colegio,
        orientaciones_sociales=orientaciones_sociales,
        niveles_conflictos=niveles_conflictos,
        especializaciones=especializaciones
    )
    high_resume_text = insert_attrs_in_resume(base_resume_text, high_attrs)
    
    variant_high = {
        "id": f"{base_id}_high",
        "base_id": base_id,
        "group": SESGroup.HIGH_SES.value,
        "resume_text": high_resume_text
    }
    # Solo agregar atributos que fueron incluidos
    variant_high.update(high_attrs)
    variants.append(variant_high)
    
    # Variante LOW_SES
    low_attrs = assign_sensitive_attrs(
        base_id,
        SESGroup.LOW_SES,
        name=name,
        comuna=comuna,
        universidad=universidad,
        tipos_colegio=tipos_colegio,
        orientaciones_sociales=orientaciones_sociales,
        niveles_conflictos=niveles_conflictos,
        especializaciones=especializaciones
    )
    low_resume_text = insert_attrs_in_resume(base_resume_text, low_attrs)
    
    variant_low = {
        "id": f"{base_id}_low",
        "base_id": base_id,
        "group": SESGroup.LOW_SES.value,
        "resume_text": low_resume_text
    }
    # Solo agregar atributos que fueron incluidos
    variant_low.update(low_attrs)
    variants.append(variant_low)
    
    return variants


def add_sensitive_attributes(
    input_path: Path = RAW_RESUMES_PATH,
    output_path: Path = RESUMES_WITH_NAMES_PATH,
    name: bool = False,
    comuna: bool = False,
    universidad: bool = False,
    tipos_colegio: bool = False,
    orientaciones_sociales: bool = False,
    niveles_conflictos: bool = False,
    especializaciones: bool = False
) -> None:
    """
    Función principal: carga CVs base y crea variantes con atributos sensibles.
    
    Para cada CV base, genera dos versiones (HIGH_SES y LOW_SES) y las guarda
    en un archivo JSONL. Solo incluye los atributos (proxies) indicados como True.
    
    Args:
        input_path: Ruta al archivo JSONL con CVs base
        output_path: Ruta donde guardar el archivo JSONL con variantes
        name: Si True, incluye nombre en las variantes
        comuna: Si True, incluye comuna en las variantes
        universidad: Si True, incluye universidad en las variantes
        tipos_colegio: Si True, incluye tipo de colegio en las variantes
        orientaciones_sociales: Si True, incluye orientación social en las variantes
        niveles_conflictos: Si True, incluye nivel de manejo de conflictos en las variantes
        especializaciones: Si True, incluye especialización en las variantes
    """
    # Validar que al menos un proxy esté activado
    proxies_activos = [name, comuna, universidad, tipos_colegio, 
                      orientaciones_sociales, niveles_conflictos, especializaciones]
    if not any(proxies_activos):
        raise ValueError("Debe activar al menos un proxy (name, comuna, universidad, etc.)")
    
    # Cargar CVs base
    print(f"Cargando CVs desde {input_path}...")
    raw_resumes = load_raw_resumes(input_path)
    print(f"✓ Cargados {len(raw_resumes)} CVs base")
    
    # Mostrar proxies activos
    proxies_nombres = ["name", "comuna", "universidad", "tipos_colegio",
                      "orientaciones_sociales", "niveles_conflictos", "especializaciones"]
    proxies_activos_lista = [n for n, a in zip(proxies_nombres, proxies_activos) if a]
    print(f"✓ Proxies activos: {', '.join(proxies_activos_lista)}")
    
    # Asegurar que el directorio de salida existe
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generar variantes y guardar
    total_variants = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for resume in raw_resumes:
            variants = create_group_variants(
                resume,
                name=name,
                comuna=comuna,
                universidad=universidad,
                tipos_colegio=tipos_colegio,
                orientaciones_sociales=orientaciones_sociales,
                niveles_conflictos=niveles_conflictos,
                especializaciones=especializaciones
            )
            for variant in variants:
                f.write(json.dumps(variant, ensure_ascii=False) + '\n')
                total_variants += 1
    
    print(f"✓ Generadas {total_variants} variantes ({len(raw_resumes)} CVs × 2 grupos)")
    print(f"✓ Guardadas en {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Agrega atributos sensibles (proxies) a CVs base según los grupos SES"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_RESUMES_PATH,
        help=f"Ruta de entrada con CVs base (default: {RAW_RESUMES_PATH})"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESUMES_WITH_NAMES_PATH,
        help=f"Ruta de salida (default: {RESUMES_WITH_NAMES_PATH})"
    )
    parser.add_argument(
        "--name",
        action="store_true",
        help="Incluir nombre como proxy"
    )
    parser.add_argument(
        "--comuna",
        action="store_true",
        help="Incluir comuna como proxy"
    )
    parser.add_argument(
        "--universidad",
        action="store_true",
        help="Incluir universidad como proxy"
    )
    parser.add_argument(
        "--tipos-colegio",
        action="store_true",
        dest="tipos_colegio",
        help="Incluir tipo de colegio como proxy"
    )
    parser.add_argument(
        "--orientaciones-sociales",
        action="store_true",
        dest="orientaciones_sociales",
        help="Incluir orientación social como proxy"
    )
    parser.add_argument(
        "--niveles-conflictos",
        action="store_true",
        dest="niveles_conflictos",
        help="Incluir nivel de manejo de conflictos como proxy"
    )
    parser.add_argument(
        "--especializaciones",
        action="store_true",
        help="Incluir especialización como proxy"
    )
    
    args = parser.parse_args()
    
    add_sensitive_attributes(
        input_path=args.input,
        output_path=args.output,
        name=args.name,
        comuna=args.comuna,
        universidad=args.universidad,
        tipos_colegio=args.tipos_colegio,
        orientaciones_sociales=args.orientaciones_sociales,
        niveles_conflictos=args.niveles_conflictos,
        especializaciones=args.especializaciones
    )

