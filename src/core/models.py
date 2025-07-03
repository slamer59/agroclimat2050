"""
Modèles de données pour l'application agroclimatique
Utilise Pydantic pour la validation et sérialisation
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class AnimalCategory(Enum):
    """Catégories d'animaux"""
    BOVINS = "bovins"
    VOLAILLES = "volailles"
    PORCINS = "porcins"
    OVINS = "ovins"


class WeatherModel(Enum):
    """Modèles météorologiques disponibles"""
    AROME = "arome"
    ARPEGE = "arpege" 
    GFS = "gfs"


class AnimalType(BaseModel):
    """Type d'animal avec ses caractéristiques"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: str
    name: str
    category: AnimalCategory
    optimal_temp: float
    optimal_humidity: float
    temp_tolerance: float
    humidity_tolerance: float
    races: List[str] = Field(default_factory=list)


class IndicatorDefinition(BaseModel):
    """Définition d'un indicateur agroclimatique"""
    id: str
    name: str
    description: str
    unit: str
    category: AnimalCategory
    animal_types: List[str]
    calculation_function: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class WeatherModelInfo(BaseModel):
    """Informations sur un modèle météorologique"""
    id: str
    name: str
    description: str
    resolution: str
    forecast_days: int
    update_frequency: str


class FilterState(BaseModel):
    """État actuel des filtres sélectionnés"""
    category: Optional[AnimalCategory] = None
    indicator_id: Optional[str] = None
    animal_type_id: Optional[str] = None
    race: Optional[str] = None
    weather_model: Optional[WeatherModel] = None
    temperature_threshold: float = 30.0
    humidity_threshold: float = 60.0
    forecast_date: Optional[str] = None


class MapData(BaseModel):
    """Données pour l'affichage de la carte"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    lons: np.ndarray
    lats: np.ndarray
    values: np.ndarray
    title: str
    unit: str
    colormap: Dict[int, str]
    labels: Dict[int, str]


# Configuration des types d'animaux
ANIMAL_TYPES = {
    "vache_laitiere": AnimalType(
        id="vache_laitiere",
        name="Vache laitière",
        category=AnimalCategory.BOVINS,
        optimal_temp=18.0,
        optimal_humidity=65.0,
        temp_tolerance=5.0,
        humidity_tolerance=15.0,
        races=["Prim'Holstein", "Montbéliarde", "Normande", "Simmental"]
    ),
    "vache_allaitante": AnimalType(
        id="vache_allaitante", 
        name="Vache allaitante",
        category=AnimalCategory.BOVINS,
        optimal_temp=20.0,
        optimal_humidity=60.0,
        temp_tolerance=8.0,
        humidity_tolerance=20.0,
        races=["Charolaise", "Limousine", "Salers", "Aubrac"]
    ),
    "bovin_engraissement": AnimalType(
        id="bovin_engraissement",
        name="Bovin à l'engraissement", 
        category=AnimalCategory.BOVINS,
        optimal_temp=16.0,
        optimal_humidity=65.0,
        temp_tolerance=6.0,
        humidity_tolerance=15.0,
        races=["Charolais", "Limousin", "Blonde d'Aquitaine"]
    ),
    "poule_pondeuse": AnimalType(
        id="poule_pondeuse",
        name="Poule pondeuse",
        category=AnimalCategory.VOLAILLES,
        optimal_temp=20.0,
        optimal_humidity=60.0,
        temp_tolerance=5.0,
        humidity_tolerance=10.0,
        races=["ISA Brown", "Lohmann Brown", "Hy-Line"]
    ),
    "poulet_chair": AnimalType(
        id="poulet_chair",
        name="Poulet de chair",
        category=AnimalCategory.VOLAILLES,
        optimal_temp=22.0,
        optimal_humidity=65.0,
        temp_tolerance=4.0,
        humidity_tolerance=10.0,
        races=["Ross 308", "Cobb 500", "Hubbard"]
    )
}

# Configuration des indicateurs
INDICATORS = {
    "stress_thermique_max": IndicatorDefinition(
        id="stress_thermique_max",
        name="Stress thermique maximal",
        description="Niveau de stress basé sur la température maximale",
        unit="Niveau (0-4)",
        category=AnimalCategory.BOVINS,
        animal_types=["vache_laitiere", "vache_allaitante", "bovin_engraissement"],
        calculation_function="calculate_heat_stress_max"
    ),
    "stress_thermique_moyen": IndicatorDefinition(
        id="stress_thermique_moyen", 
        name="Stress thermique moyen",
        description="Niveau de stress basé sur la température moyenne",
        unit="Niveau (0-4)",
        category=AnimalCategory.BOVINS,
        animal_types=["vache_laitiere", "vache_allaitante", "bovin_engraissement"],
        calculation_function="calculate_heat_stress_avg"
    ),
    "perte_ponte": IndicatorDefinition(
        id="perte_ponte",
        name="Perte de ponte",
        description="Réduction de la production d'œufs due au stress climatique",
        unit="Pourcentage (%)",
        category=AnimalCategory.VOLAILLES,
        animal_types=["poule_pondeuse"],
        calculation_function="calculate_laying_loss"
    ),
    "perte_lait": IndicatorDefinition(
        id="perte_lait",
        name="Perte de production laitière",
        description="Réduction de la production de lait due au stress thermique",
        unit="Pourcentage (%)",
        category=AnimalCategory.BOVINS,
        animal_types=["vache_laitiere"],
        calculation_function="calculate_milk_production_loss"
    ),
    "perte_gmq": IndicatorDefinition(
        id="perte_gmq",
        name="Perte de GMQ",
        description="Réduction du gain de masse quotidien",
        unit="Pourcentage (%)",
        category=AnimalCategory.BOVINS,
        animal_types=["bovin_engraissement"],
        calculation_function="calculate_daily_weight_gain_loss"
    )
}

# Configuration des modèles météorologiques
WEATHER_MODELS = {
    WeatherModel.AROME: WeatherModelInfo(
        id="arome",
        name="AROME",
        description="Haute résolution",
        resolution="2 jours",
        forecast_days=2,
        update_frequency="4 fois/jour"
    ),
    WeatherModel.ARPEGE: WeatherModelInfo(
        id="arpege", 
        name="ARPEGE",
        description="Moyen terme",
        resolution="4 jours",
        forecast_days=4,
        update_frequency="2 fois/jour"
    ),
    WeatherModel.GFS: WeatherModelInfo(
        id="gfs",
        name="GFS", 
        description="Global",
        resolution="10 jours",
        forecast_days=10,
        update_frequency="4 fois/jour"
    )
}
