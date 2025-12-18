
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


# Listas de nombres y comunas por grupo socioeconómico
# HIGH_SES: comunas y nombres asociados a clase alta en Chile
HIGH_SES_NAMES = [
    "Sofía Larraín",
    "Sebastián Edwards",
    "Isidora Matte",
    "Tomás Errázuriz",
    "Constanza Undurraga"
]

HIGH_SES_COMUNAS = [
    "Vitacura",
    "Las Condes",
    "Lo Barnechea",
    "Providencia",
    "Ñuñoa"
]


# LOW_SES: comunas y nombres asociados a clase baja en Chile
LOW_SES_NAMES = [
    "Camila Álvarez",
    "Kevin González",
    "Yasna Pérez",
    "Brayan Muñoz",
    "Javiera Torres"
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
ORIENTACIONES_SOCIALES = [
    "Trabajo con familias en situación de vulnerabilidad",
    "Intervención comunitaria",
    "Salud mental y adicciones",
    "Infancia y adolescencia",
    "Adultos mayores",
    "Violencia intrafamiliar"
]

NIVELES_CONFLICTOS = [
    "Alto",
    "Medio",
    "Bajo"
]

ESPECIALIZACIONES = [
    "Intervención familiar",
    "Políticas sociales",
    "Desarrollo comunitario",
    "Salud pública",
    "Educación social"
]

