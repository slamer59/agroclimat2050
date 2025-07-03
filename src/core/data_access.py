"""
Fonctions d'accès aux données météorologiques et de cache
Approche fonctionnelle pour la gestion des données
"""

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def setup_data_directories(data_dir: str = "data", cache_dir: str = "cache") -> Tuple[Path, Path]:
    """
    Initialise les répertoires de données et cache
    
    Returns:
        Tuple des chemins data_dir et cache_dir
    """
    data_path = Path(data_dir)
    cache_path = Path(cache_dir)
    
    data_path.mkdir(exist_ok=True)
    cache_path.mkdir(exist_ok=True)
    
    return data_path, cache_path


def save_to_parquet(df: pd.DataFrame, filename: str, data_dir: str = "data") -> Path:
    """
    Sauvegarde un DataFrame en format Parquet
    
    Args:
        df: DataFrame à sauvegarder
        filename: Nom du fichier (sans extension)
        data_dir: Répertoire de destination
        
    Returns:
        Chemin du fichier sauvegardé
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    filepath = data_path / f"{filename}.parquet"
    df.to_parquet(filepath, compression='snappy')
    
    return filepath


def load_from_parquet(filename: str, data_dir: str = "data") -> Optional[pd.DataFrame]:
    """
    Charge un DataFrame depuis un fichier Parquet
    
    Args:
        filename: Nom du fichier (sans extension)
        data_dir: Répertoire source
        
    Returns:
        DataFrame ou None si le fichier n'existe pas
    """
    filepath = Path(data_dir) / f"{filename}.parquet"
    
    if filepath.exists():
        return pd.read_parquet(filepath)
    
    return None


@lru_cache(maxsize=32)
def get_cached_data(cache_key: str, cache_dir: str = "cache") -> Optional[Any]:
    """
    Récupère des données du cache avec LRU
    
    Args:
        cache_key: Clé de cache
        cache_dir: Répertoire de cache
        
    Returns:
        Données mises en cache ou None
    """
    cache_file = Path(cache_dir) / f"{cache_key}.pkl"
    
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    return None


def set_cache(cache_key: str, data: Any, cache_dir: str = "cache") -> None:
    """
    Met des données en cache
    
    Args:
        cache_key: Clé de cache
        data: Données à mettre en cache
        cache_dir: Répertoire de cache
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    
    cache_file = cache_path / f"{cache_key}.pkl"
    
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)


def generate_sample_weather_data(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Génère des données météorologiques d'exemple pour la France
    
    Args:
        seed: Graine pour la reproductibilité
        
    Returns:
        Tuple (lons, lats, temp_data, humidity_data)
    """
    np.random.seed(seed)
    
    # Coordonnées de la France
    france_bounds = (-5.5, 9.6, 41.0, 51.2)  # ouest, est, sud, nord
    
    # Grille de coordonnées
    lons = np.linspace(france_bounds[0], france_bounds[1], 50)
    lats = np.linspace(france_bounds[2], france_bounds[3], 40)
    
    # Données simulées avec gradient géographique
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Température avec gradient nord-sud
    base_temp = 15 + (lat_grid - france_bounds[2]) / (france_bounds[3] - france_bounds[2]) * 10
    temp_data = base_temp + 5 * np.random.random(lon_grid.shape)
    
    # Humidité avec variabilité
    humidity_data = 40 + 40 * np.random.random(lon_grid.shape)
    
    return lons, lats, temp_data, humidity_data


def load_or_generate_weather_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge les données météo depuis le cache ou les génère
    
    Returns:
        Tuple (lons, lats, temp_data, humidity_data)
    """
    # Essayer de charger depuis Parquet
    temp_df = load_from_parquet("temperature_data")
    humidity_df = load_from_parquet("humidity_data")
    
    if temp_df is not None and humidity_df is not None:
        # Reconstruire les coordonnées
        lons, lats, _, _ = generate_sample_weather_data()
        temp_data = temp_df.values
        humidity_data = humidity_df.values
    else:
        # Générer de nouvelles données
        lons, lats, temp_data, humidity_data = generate_sample_weather_data()
        
        # Sauvegarder pour la prochaine fois
        save_to_parquet(pd.DataFrame(temp_data), "temperature_data")
        save_to_parquet(pd.DataFrame(humidity_data), "humidity_data")
    
    return lons, lats, temp_data, humidity_data


def get_france_bounds() -> Tuple[float, float, float, float]:
    """
    Retourne les limites géographiques de la France
    
    Returns:
        Tuple (ouest, est, sud, nord)
    """
    return (-5.5, 9.6, 41.0, 51.2)
