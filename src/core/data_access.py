import pickle
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


class DataManager:
    """Gestionnaire de données avec cache et stockage Parquet"""

    def __init__(self, data_dir="data", cache_dir="cache"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

    def save_to_parquet(self, df, filename):
        """Sauvegarde DataFrame en Parquet"""
        filepath = self.data_dir / f"{filename}.parquet"
        df.to_parquet(filepath, compression='snappy')
        return filepath

    def load_from_parquet(self, filename):
        """Charge DataFrame depuis Parquet"""
        filepath = self.data_dir / f"{filename}.parquet"
        if filepath.exists():
            return pd.read_parquet(filepath)
        return None

    @lru_cache(maxsize=32)
    def get_cached_data(self, cache_key):
        """Récupère des données du cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def set_cache(self, cache_key, data):
        """Met en cache des données"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

    def generate_sample_data(self):
        """Génère des données d'exemple"""
        # Créer une grille de données météorologiques simulées
        lons = np.linspace(-5.5, 9.6, 50)
        lats = np.linspace(41.0, 51.2, 40)

        # Données de température (simulation)
        np.random.seed(42)
        temp_data = 15 + 15 * np.random.random((len(lats), len(lons)))

        # Données d'humidité (simulation)
        humidity_data = 40 + 40 * np.random.random((len(lats), len(lons)))

        # Stocker les coordonnées
        self.__dict__.update({'lons':lons, 'lats':lats})

        # Sauvegarder en Parquet pour optimisation
        temp_df = pd.DataFrame(temp_data)
        humidity_df = pd.DataFrame(humidity_data)

        self.save_to_parquet(temp_df, "temperature_data")
        self.save_to_parquet(humidity_df, "humidity_data")
        return lons, lats, temp_data, humidity_data


def load_or_generate_weather_data():
    """Charge ou génère les données météorologiques"""
    data_manager = DataManager()
    
    # Essayer de charger depuis le cache
    cached_data = data_manager.get_cached_data("weather_data")
    if cached_data is not None:
        return cached_data
    
    # Générer de nouvelles données
    lons, lats, temp_data, humidity_data = data_manager.generate_sample_data()
    
    # Mettre en cache
    weather_data = (lons, lats, temp_data, humidity_data)
    data_manager.set_cache("weather_data", weather_data)
    
    return weather_data
