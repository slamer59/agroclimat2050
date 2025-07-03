from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import numpy as np


class AnimalCategory(Enum):
    BOVINS = "BOVINS"
    VOLAILLES = "VOLAILLES"

@dataclass
class AnimalType:
    id: str
    name: str
    category: AnimalCategory
    races: List[str]

ANIMAL_TYPES = {
    "VACHE_LAITIERE": AnimalType(id="VACHE_LAITIERE", name="Vache laitière", category=AnimalCategory.BOVINS, races=["PRIM'HOLSTEIN", "MONTBÉLIARDE", "NORMANDE"]),
    "VACHE_ALLAITANTE": AnimalType(id="VACHE_ALLAITANTE", name="Vache allaitante", category=AnimalCategory.BOVINS, races=["CHAROLAISE", "LIMOUSINE", "HEREFORD"]),
    "POULES_PONDEUSES": AnimalType(id="POULES_PONDEUSES", name="Poules pondeuses", category=AnimalCategory.VOLAILLES, races=["ISA BROWN", "HY-LINE BROWN", "Lohmann Brown"]),
    "BOVINS_A_L_ENGRAISSEMENT": AnimalType(id="BOVINS_A_L_ENGRAISSEMENT", name="Bovins à l'engraissement", category=AnimalCategory.BOVINS, races=["CHAROLAISE", "LIMOUSINE", "HEREFORD"]),
}

@dataclass
class Indicator:
    id: str
    name: str
    category: AnimalCategory
    calculation_function: str

INDICATORS = {
    "STRESS_THERMIQUE_MAXIMAL": Indicator(id="STRESS_THERMIQUE_MAXIMAL", name="Stress thermique maximal", category=AnimalCategory.BOVINS, calculation_function="calculate_heat_stress_max"),
    "PERTE_DE_PONTE": Indicator(id="PERTE_DE_PONTE", name="Perte de ponte (%)", category=AnimalCategory.VOLAILLES, calculation_function="calculate_laying_loss"),
    "PERTE_DE_PRODUCTION_DE_LAIT": Indicator(id="PERTE_DE_PRODUCTION_DE_LAIT", name="Perte de production de lait (%)", category=AnimalCategory.BOVINS, calculation_function="calculate_milk_production_loss"),
    "PERTE_DE_GMQ": Indicator(id="PERTE_DE_GMQ", name="Perte de GMQ - Gain en masse quotidien (%)", category=AnimalCategory.BOVINS, calculation_function="calculate_daily_weight_gain_loss"),
}

@dataclass
class WeatherModel:
    id: str
    name: str
    description: str
    resolution: str

WEATHER_MODELS = {
    "AROME": WeatherModel(id="AROME", name="AROME", description="Modèle de prévision numérique à haute résolution", resolution="1 km"),
    "ARPEGE": WeatherModel(id="ARPEGE", name="ARPEGE", description="Modèle de prévision numérique à moyenne résolution", resolution="10 km"),
    "GFS": WeatherModel(id="GFS", name="GFS", description="Modèle de prévision numérique global", resolution="25 km"),
}

@dataclass
class FilterState:
    category: AnimalCategory = AnimalCategory.BOVINS
    indicator_id: str = ""
    animal_type_id: str = ""
    race: str = ""
    weather_model: str = "AROME"
    temperature_threshold: float = 30.0

@dataclass
class MapData:
    lons: list
    lats: list
    temp_data: np.ndarray
    humidity_data: np.ndarray
