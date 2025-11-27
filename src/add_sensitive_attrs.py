"""
Pipeline 2: Agregar atributos sensibles (nombres y comunas) a CVs base.

Este módulo toma CVs sin atributos sensibles y crea variantes con nombres
y comunas asociadas a grupos socioeconómicos altos (HIGH_SES) y bajos (LOW_SES).
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
    LOW_SES_COMUNAS
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


def assign_sensitive_attrs(base_id: str, group: SESGroup) -> Dict[str, str]:
    """
    Asigna atributos sensibles (nombre, comuna, email) según el grupo socioeconómico.
    
    Args:
        base_id: ID del CV base
        group: Grupo socioeconómico (HIGH_SES o LOW_SES)
        
    Returns:
        Diccionario con name, comuna, y email
    """
    if group == SESGroup.HIGH_SES:
        name = random.choice(HIGH_SES_NAMES)
        comuna = random.choice(HIGH_SES_COMUNAS)
    elif group == SESGroup.LOW_SES:
        name = random.choice(LOW_SES_NAMES)
        comuna = random.choice(LOW_SES_COMUNAS)
    else:
        raise ValueError(f"Grupo no válido: {group}")
    
    # Generar email derivado del nombre (formato simple)
    nombre_parts = name.lower().split()
    apellido = nombre_parts[-1] if len(nombre_parts) > 1 else nombre_parts[0]
    # Remover acentos y caracteres especiales para email
    apellido_clean = apellido.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
    email = f"{nombre_parts[0].lower()}.{apellido_clean}@gmail.com"
    
    return {
        "name": name,
        "comuna": comuna,
        "email": email
    }


def insert_name_and_comuna_in_resume(resume_text: str, name: str, comuna: str) -> str:
    """
    Inserta el nombre y la comuna en el texto del CV de forma coherente.
    
    Agrega un encabezado al inicio del CV con nombre, título y comuna.
    
    Args:
        resume_text: Texto original del CV
        name: Nombre completo
        comuna: Comuna
        
    Returns:
        Texto del CV con nombre y comuna insertados
    """
    # Crear encabezado profesional
    header = f"{name.upper()}\nTrabajador/a Social\n{comuna}, Región Metropolitana\n\n"
    
    # Insertar al inicio del CV
    return header + resume_text


def create_group_variants(resume: Dict) -> List[Dict]:
    """
    Crea dos variantes de un CV base: una con atributos HIGH_SES y otra LOW_SES.
    
    Args:
        resume: Diccionario con el CV base (debe tener 'id', 'attributes', 'resume_text')
        
    Returns:
        Lista con dos diccionarios: uno para HIGH_SES y otro para LOW_SES
    """
    base_id = resume["id"]
    base_resume_text = resume["resume_text"]
    
    variants = []
    
    # Variante HIGH_SES
    high_attrs = assign_sensitive_attrs(base_id, SESGroup.HIGH_SES)
    high_resume_text = insert_name_and_comuna_in_resume(
        base_resume_text,
        high_attrs["name"],
        high_attrs["comuna"]
    )
    
    variants.append({
        "id": f"{base_id}_high",
        "base_id": base_id,
        "group": SESGroup.HIGH_SES.value,
        "name": high_attrs["name"],
        "comuna": high_attrs["comuna"],
        "email": high_attrs["email"],
        "resume_text": high_resume_text
    })
    
    # Variante LOW_SES
    low_attrs = assign_sensitive_attrs(base_id, SESGroup.LOW_SES)
    low_resume_text = insert_name_and_comuna_in_resume(
        base_resume_text,
        low_attrs["name"],
        low_attrs["comuna"]
    )
    
    variants.append({
        "id": f"{base_id}_low",
        "base_id": base_id,
        "group": SESGroup.LOW_SES.value,
        "name": low_attrs["name"],
        "comuna": low_attrs["comuna"],
        "email": low_attrs["email"],
        "resume_text": low_resume_text
    })
    
    return variants


def add_sensitive_attributes(
    input_path: Path = RAW_RESUMES_PATH,
    output_path: Path = RESUMES_WITH_NAMES_PATH
) -> None:
    """
    Función principal: carga CVs base y crea variantes con atributos sensibles.
    
    Para cada CV base, genera dos versiones (HIGH_SES y LOW_SES) y las guarda
    en un archivo JSONL.
    
    Args:
        input_path: Ruta al archivo JSONL con CVs base
        output_path: Ruta donde guardar el archivo JSONL con variantes
    """
    # Cargar CVs base
    print(f"Cargando CVs desde {input_path}...")
    raw_resumes = load_raw_resumes(input_path)
    print(f"✓ Cargados {len(raw_resumes)} CVs base")
    
    # Asegurar que el directorio de salida existe
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generar variantes y guardar
    total_variants = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for resume in raw_resumes:
            variants = create_group_variants(resume)
            for variant in variants:
                f.write(json.dumps(variant, ensure_ascii=False) + '\n')
                total_variants += 1
    
    print(f"✓ Generadas {total_variants} variantes ({len(raw_resumes)} CVs × 2 grupos)")
    print(f"✓ Guardadas en {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Agrega atributos sensibles (nombres y comunas) a CVs base"
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
    
    args = parser.parse_args()
    
    add_sensitive_attributes(
        input_path=args.input,
        output_path=args.output
    )

