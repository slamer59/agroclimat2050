"""
Fonctions de calcul des indicateurs agroclimatiques
Approche fonctionnelle pure sans effets de bord
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .models import ANIMAL_TYPES, AnimalType


def calculate_heat_stress_max(
    temp_data: np.ndarray, 
    animal_type_id: str,
    threshold: Optional[float] = None
) -> np.ndarray:
    """
    Calcule le stress thermique maximal pour un type d'animal
    
    Args:
        temp_data: Données de température
        animal_type_id: Identifiant du type d'animal
        threshold: Seuil personnalisé (optionnel)
    
    Returns:
        Tableau des niveaux de stress (0-4)
    """
    animal = ANIMAL_TYPES[animal_type_id]
    
    # Utiliser le seuil personnalisé ou la température optimale + tolérance
    if threshold is None:
        threshold = animal.optimal_temp + animal.temp_tolerance
    
    # Calcul du stress relatif
    stress_values = np.where(
        temp_data > threshold,
        (temp_data - threshold) / threshold * 100,
        0
    )
    
    # Classification en 5 niveaux
    stress_classes = np.select([
        stress_values <= 0,
        (stress_values > 0) & (stress_values <= 10),
        (stress_values > 10) & (stress_values <= 25),
        (stress_values > 25) & (stress_values <= 50),
        stress_values > 50
    ], [0, 1, 2, 3, 4], default=0)
    
    return stress_classes


def calculate_heat_stress_avg(
    temp_data: np.ndarray,
    animal_type_id: str, 
    threshold: Optional[float] = None
) -> np.ndarray:
    """
    Calcule le stress thermique moyen pour un type d'animal
    """
    animal = ANIMAL_TYPES[animal_type_id]
    
    if threshold is None:
        threshold = animal.optimal_temp
    
    stress_values = np.where(
        temp_data > threshold,
        (temp_data - threshold) / threshold * 50,
        0
    )
    
    stress_classes = np.select([
        stress_values <= 0,
        (stress_values > 0) & (stress_values <= 5),
        (stress_values > 5) & (stress_values <= 15),
        (stress_values > 15) & (stress_values <= 30),
        stress_values > 30
    ], [0, 1, 2, 3, 4], default=0)
    
    return stress_classes


def calculate_laying_loss(
    temp_data: np.ndarray,
    humidity_data: np.ndarray,
    animal_type_id: str
) -> np.ndarray:
    """
    Calcule la perte de ponte pour les volailles
    """
    animal = ANIMAL_TYPES[animal_type_id]
    
    # Calcul des écarts par rapport aux conditions optimales
    temp_stress = np.abs(temp_data - animal.optimal_temp) / animal.optimal_temp
    humidity_stress = np.abs(humidity_data - animal.optimal_humidity) / animal.optimal_humidity
    
    # Stress combiné
    total_stress = (temp_stress + humidity_stress) * 50
    
    stress_classes = np.select([
        total_stress <= 5,
        (total_stress > 5) & (total_stress <= 15),
        (total_stress > 15) & (total_stress <= 30),
        (total_stress > 30) & (total_stress <= 50),
        total_stress > 50
    ], [0, 1, 2, 3, 4], default=0)
    
    return stress_classes


def calculate_milk_production_loss(
    temp_data: np.ndarray,
    animal_type_id: str
) -> np.ndarray:
    """
    Calcule la perte de production laitière
    """
    animal = ANIMAL_TYPES[animal_type_id]
    optimal_temp = animal.optimal_temp
    
    stress_values = np.where(
        temp_data > optimal_temp,
        (temp_data - optimal_temp) / optimal_temp * 60,
        0
    )
    
    stress_classes = np.select([
        stress_values <= 0,
        (stress_values > 0) & (stress_values <= 10),
        (stress_values > 10) & (stress_values <= 25),
        (stress_values > 25) & (stress_values <= 40),
        stress_values > 40
    ], [0, 1, 2, 3, 4], default=0)
    
    return stress_classes


def calculate_daily_weight_gain_loss(
    temp_data: np.ndarray,
    humidity_data: np.ndarray,
    animal_type_id: str
) -> np.ndarray:
    """
    Calcule la perte de gain de masse quotidien
    """
    animal = ANIMAL_TYPES[animal_type_id]
    
    temp_factor = np.where(
        temp_data > animal.optimal_temp,
        (temp_data - animal.optimal_temp) / animal.optimal_temp,
        0
    )
    
    humidity_factor = np.where(
        humidity_data > animal.optimal_humidity,
        (humidity_data - animal.optimal_humidity) / animal.optimal_humidity,
        0
    )
    
    combined_stress = (temp_factor + humidity_factor) * 40
    
    stress_classes = np.select([
        combined_stress <= 0,
        (combined_stress > 0) & (combined_stress <= 8),
        (combined_stress > 8) & (combined_stress <= 20),
        (combined_stress > 20) & (combined_stress <= 35),
        combined_stress > 35
    ], [0, 1, 2, 3, 4], default=0)
    
    return stress_classes


def get_indicator_function(function_name: str):
    """
    Retourne la fonction de calcul d'indicateur par son nom
    """
    functions = {
        'calculate_heat_stress_max': calculate_heat_stress_max,
        'calculate_heat_stress_avg': calculate_heat_stress_avg,
        'calculate_laying_loss': calculate_laying_loss,
        'calculate_milk_production_loss': calculate_milk_production_loss,
        'calculate_daily_weight_gain_loss': calculate_daily_weight_gain_loss
    }
    
    return functions.get(function_name)


def get_stress_colors() -> Dict[int, str]:
    """Retourne la palette de couleurs pour les niveaux de stress"""
    return {
        0: '#00ff00',  # Vert - Aucun stress
        1: '#ffff00',  # Jaune - Faible
        2: '#ffa500',  # Orange - Modéré
        3: '#ff4500',  # Rouge orangé - Fort
        4: '#8b0000'   # Rouge foncé - Très sévère
    }


def get_stress_labels() -> Dict[int, str]:
    """Retourne les labels pour les niveaux de stress"""
    return {
        0: 'Aucun stress',
        1: 'Faible',
        2: 'Modéré',
        3: 'Fort', 
        4: 'Très sévère'
    }
