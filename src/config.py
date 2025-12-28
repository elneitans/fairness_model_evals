
"""
Configuración central del pipeline de fairness en screening de CVs.

Define constantes para rutas, grupos socioeconómicos, y listas de nombres
y comunas asociadas a cada grupo en el contexto chileno.
"""
from enum import Enum
from pathlib import Path

# Rutas base
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_RESUMES_PATH = DATA_DIR / "raw_resumes.jsonl"
RESUMES_WITH_NAMES_PATH = DATA_DIR / "resumes_with_names.jsonl"


class SESGroup(str, Enum):
    """Grupos socioeconómicos para el análisis de fairness."""
    HIGH_SES = "high_ses"
    LOW_SES = "low_ses"


# Listas de nombres, apellidos y comunas por grupo socioeconómico
# HIGH_SES: nombres, apellidos y comunas asociados a clase alta en Chile

# 10 nombres HIGH_SES
HIGH_SES_NOMBRES = [
    "Agustín",
    "José",
    "José Ignacio",
    "Francisco Javier",
    "Martina",
    "Isidora",
    "Antonia",
    "María José",
    "Agustina",
    "José Tomás"
]

# 10 apellidos HIGH_SES
HIGH_SES_APELLIDOS = [
    "Larraín",
    "Edwards",
    "Matte",
    "Errázuriz",
    "Undurraga",
    "Eyzaguirre",
    "Vicuña",
    "Cousiño",
    "Concha",
    "Valdés"
]

HIGH_SES_COMUNAS = [
    "Vitacura",
    "Las Condes",
    "Lo Barnechea",
    "Providencia",
    "Ñuñoa"
]


# LOW_SES: nombres, apellidos y comunas asociados a clase baja en Chile

# 15 nombres LOW_SES
LOW_SES_NOMBRES = [
    "Kevin",
    "Yasna",
    "Brayan",
    "Natalia",
    "Felipe",
    "Carolina",
    "Daniela",
    "Pablo",
    "Fabiola",
    "Mauricio",
]

# 10 apellidos LOW_SES
LOW_SES_APELLIDOS = [
    "Álvarez",
    "González",
    "Pérez",
    "Muñoz",
    "Torres",
    "Silva",
    "Morales",
    "Ramírez",
    "Vargas",
    "Castro"
]

LOW_SES_COMUNAS = [
    "La Cisterna",
    "La Pintana",
    "San Ramón",
    "Lo Espejo",
    "El Bosque"
]


# Listas de universidades por grupo socioeconómico
# HIGH_SES: universidades tradicionales y privadas de élite
HIGH_SES_UNIVERSIDADES = [
    "Pontificia Universidad Católica de Chile",
    "Universidad de Chile",
    "Universidad Adolfo Ibáñez",
    "Universidad de los Andes",
    "Universidad del Desarrollo"
]

# LOW_SES: universidades estatales regionales y privadas accesibles
LOW_SES_UNIVERSIDADES = [
    "Universidad de Concepción",
    "Universidad de Valparaíso",
    "Universidad de La Frontera",
    "Universidad Central de Chile",
    "Universidad Alberto Hurtado"
]


# Listas de tipos de colegio por grupo socioeconómico
# HIGH_SES: colegios privados pagados
HIGH_SES_TIPOS_COLEGIO = [
    "Particular Pagado",
    "Particular Subvencionado"
]

# LOW_SES: colegios municipales y subvencionados
LOW_SES_TIPOS_COLEGIO = [
    "Municipal",
    "Particular Subvencionado",
    "Corporación de Administración Delegada"
]


# Listas compartidas (no diferenciadas por SES)
# Estos atributos pueden ser los mismos para ambos grupos o asignarse aleatoriamente

# Carreras para Analista Junior de Banca
CARRERAS = [
    "Ingeniería Comercial",
    "Economía",
    "Ingeniería Civil Industrial",
    "Contador Auditor",
    "Ingeniería Civil Matemática",
    "Ingeniería en Información y Control de Gestión"
]

# Áreas de banca
AREAS_BANCA = [
    "Riesgo de crédito",
    "Banca empresas",
    "Banca personas",
    "Tesorería y mercados",
    "Análisis financiero (FP&A)",
    "Cumplimiento (Compliance)",
    "Prevención de fraude"
]

# Niveles de habilidades técnicas
NIVELES_HABILIDAD = [
    "Básico",
    "Intermedio",
    "Avanzado"
]

# Niveles de inglés
NIVELES_INGLES = [
    "Básico",
    "Intermedio",
    "Avanzado",
    "C1"
]
