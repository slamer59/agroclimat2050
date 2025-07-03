import numpy as np


def calculate_heat_stress_max(temp_data, threshold=30):
    """Calcule le stress thermique maximal"""
    stress_values = np.where(temp_data > threshold, (temp_data - threshold) / threshold * 100, 0)
    stress_classes = np.select([
        stress_values <= 0,
        (stress_values > 0) & (stress_values <= 10),
        (stress_values > 10) & (stress_values <= 25),
        (stress_values > 25) & (stress_values <= 50),
        stress_values > 50
    ], [0, 1, 2, 3, 4], default=0)
    return stress_classes

def calculate_heat_stress_avg(temp_data, threshold=25):
    """Calcule le stress thermique moyen"""
    stress_values = np.where(temp_data > threshold, (temp_data - threshold) / threshold * 50, 0)
    stress_classes = np.select([
        stress_values <= 0,
        (stress_values > 0) & (stress_values <= 5),
        (stress_values > 5) & (stress_values <= 15),
        (stress_values > 15) & (stress_values <= 30),
        stress_values > 30
    ], [0, 1, 2, 3, 4], default=0)
    return stress_classes

def calculate_laying_loss(temp_data, humidity_data):
    """Calcule la perte de ponte"""
    comfort_temp = 20
    comfort_humidity = 60

    temp_stress = np.abs(temp_data - comfort_temp) / comfort_temp
    humidity_stress = np.abs(humidity_data - comfort_humidity) / comfort_humidity

    total_stress = (temp_stress + humidity_stress) * 50

    stress_classes = np.select([
        total_stress <= 5,
        (total_stress > 5) & (total_stress <= 15),
        (total_stress > 15) & (total_stress <= 30),
        (total_stress > 30) & (total_stress <= 50),
        total_stress > 50
    ], [0, 1, 2, 3, 4], default=0)
    return stress_classes

def calculate_milk_production_loss(temp_data):
    """Calcule la perte de production de lait"""
    optimal_temp = 18
    stress_values = np.where(temp_data > optimal_temp, (temp_data - optimal_temp) / optimal_temp * 60, 0)
    stress_classes = np.select([
        stress_values <= 0,
        (stress_values > 0) & (stress_values <= 10),
        (stress_values > 10) & (stress_values <= 25),
        (stress_values > 25) & (stress_values <= 40),
        stress_values > 40
    ], [0, 1, 2, 3, 4], default=0)
    return stress_classes

def calculate_daily_weight_gain_loss(temp_data, humidity_data):
    """Calcule la perte de GMQ (Gain de Masse Quotidien)"""
    optimal_temp = 16
    optimal_humidity = 65

    temp_factor = np.where(temp_data > optimal_temp, (temp_data - optimal_temp) / optimal_temp, 0)
    humidity_factor = np.where(humidity_data > optimal_humidity, (humidity_data - optimal_humidity) / optimal_humidity, 0)

    combined_stress = (temp_factor + humidity_factor) * 40

    stress_classes = np.select([
        combined_stress <= 0,
        (combined_stress > 0) & (combined_stress <= 8),
        (combined_stress > 8) & (combined_stress <= 20),
        (combined_stress > 20) & (combined_stress <= 35),
        combined_stress > 35
    ], [0, 1, 2, 3, 4], default=0)
    return stress_classes

def get_stress_colors():
    return {
        0: '#00ff00',  # Vert - Aucun stress (0.0-68.0)
        1: '#ffff00',  # Jaune - Faible (68.0-72.0)
        2: '#ffa500',  # Orange - Modéré (72.0-80.0)
        3: '#ff4500',  # Rouge orangé - Fort (80.0-90.0)
        4: '#8b0000'   # Rouge foncé - Très sévère (90.0-99.0)
    }

def get_stress_labels():
    return {
        0: '0.0-68.0 : Aucun stress',
        1: '68.0-72.0 : Faible',
        2: '72.0-80.0 : Modéré',
        3: '80.0-90.0 : Fort',
        4: '90.0-99.0 : Très sévère'
    }


def get_indicator_function(function_name):
    """Retourne la fonction de calcul d'indicateur par son nom"""
    function_map = {
        'calculate_heat_stress_max': calculate_heat_stress_max,
        'calculate_heat_stress_avg': calculate_heat_stress_avg,
        'calculate_laying_loss': calculate_laying_loss,
        'calculate_milk_production_loss': calculate_milk_production_loss,
        'calculate_daily_weight_gain_loss': calculate_daily_weight_gain_loss
    }
    return function_map.get(function_name)
