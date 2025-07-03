"""
Application Panel pour visualiser des indicateurs agroclimatiques
Architecture modulaire avec séparation UI/calculs et cache efficace
"""

import pickle
import warnings
from functools import lru_cache
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geoviews as gv
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
import xarray as xr
from holoviews import opts

warnings.filterwarnings('ignore')

# Configuration Panel
pn.extension('tabulator')
hv.extension('bokeh')
gv.extension('bokeh')

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

class IndicatorCalculator:
    """Calculateur d'indicateurs agroclimatiques"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        
    def calculate_heat_stress_max(self, temp_data, threshold=30):
        """Calcule le stress thermique maximal"""
        cache_key = f"heat_stress_max_{threshold}"
        cached = self.data_manager.get_cached_data(cache_key)
        if cached is not None:
            return cached
            
        # Simulation de calcul de stress thermique
        stress_values = np.where(temp_data > threshold, 
                               (temp_data - threshold) / threshold * 100, 0)
        
        # Classification du stress
        stress_classes = np.select([
            stress_values <= 0,
            (stress_values > 0) & (stress_values <= 10),
            (stress_values > 10) & (stress_values <= 25),
            (stress_values > 25) & (stress_values <= 50),
            stress_values > 50
        ], [0, 1, 2, 3, 4], default=0)
        
        result = stress_classes
        self.data_manager.set_cache(cache_key, result)
        return result
    
    def calculate_heat_stress_avg(self, temp_data, threshold=25):
        """Calcule le stress thermique moyen"""
        cache_key = f"heat_stress_avg_{threshold}"
        cached = self.data_manager.get_cached_data(cache_key)
        if cached is not None:
            return cached
            
        # Simulation de calcul de stress thermique moyen
        stress_values = np.where(temp_data > threshold, 
                               (temp_data - threshold) / threshold * 50, 0)
        
        stress_classes = np.select([
            stress_values <= 0,
            (stress_values > 0) & (stress_values <= 5),
            (stress_values > 5) & (stress_values <= 15),
            (stress_values > 15) & (stress_values <= 30),
            stress_values > 30
        ], [0, 1, 2, 3, 4], default=0)
        
        result = stress_classes
        self.data_manager.set_cache(cache_key, result)
        return result
    
    def calculate_laying_loss(self, temp_data, humidity_data):
        """Calcule la perte de ponte"""
        cache_key = "laying_loss"
        cached = self.data_manager.get_cached_data(cache_key)
        if cached is not None:
            return cached
            
        # Simulation basée sur température et humidité
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
        
        result = stress_classes
        self.data_manager.set_cache(cache_key, result)
        return result
    
    def calculate_milk_production_loss(self, temp_data):
        """Calcule la perte de production de lait"""
        cache_key = "milk_production_loss"
        cached = self.data_manager.get_cached_data(cache_key)
        if cached is not None:
            return cached
            
        # Simulation pour bovins laitiers
        optimal_temp = 18
        stress_values = np.where(temp_data > optimal_temp,
                               (temp_data - optimal_temp) / optimal_temp * 60, 0)
        
        stress_classes = np.select([
            stress_values <= 0,
            (stress_values > 0) & (stress_values <= 10),
            (stress_values > 10) & (stress_values <= 25),
            (stress_values > 25) & (stress_values <= 40),
            stress_values > 40
        ], [0, 1, 2, 3, 4], default=0)
        
        result = stress_classes
        self.data_manager.set_cache(cache_key, result)
        return result
    
    def calculate_daily_weight_gain_loss(self, temp_data, humidity_data):
        """Calcule la perte de GMQ (Gain de Masse Quotidien)"""
        cache_key = "daily_weight_gain_loss"
        cached = self.data_manager.get_cached_data(cache_key)
        if cached is not None:
            return cached
            
        # Simulation pour bovins à l'engraissement
        optimal_temp = 16
        optimal_humidity = 65
        
        temp_factor = np.where(temp_data > optimal_temp,
                             (temp_data - optimal_temp) / optimal_temp, 0)
        humidity_factor = np.where(humidity_data > optimal_humidity,
                                 (humidity_data - optimal_humidity) / optimal_humidity, 0)
        
        combined_stress = (temp_factor + humidity_factor) * 40
        
        stress_classes = np.select([
            combined_stress <= 0,
            (combined_stress > 0) & (combined_stress <= 8),
            (combined_stress > 8) & (combined_stress <= 20),
            (combined_stress > 20) & (combined_stress <= 35),
            combined_stress > 35
        ], [0, 1, 2, 3, 4], default=0)
        
        result = stress_classes
        self.data_manager.set_cache(cache_key, result)
        return result

class MapVisualizer:
    """Visualisateur de cartes pour les indicateurs"""
    
    def __init__(self):
        self.stress_colors = {
            0: '#00ff00',  # Vert - Aucun stress
            1: '#ffff00',  # Jaune - Faible
            2: '#ffa500',  # Orange - Modéré
            3: '#ff4500',  # Rouge orangé - Fort
            4: '#8b0000'   # Rouge foncé - Très sévère
        }
        
        self.stress_labels = {
            0: 'Aucun stress',
            1: 'Faible',
            2: 'Modéré', 
            3: 'Fort',
            4: 'Très sévère'
        }
    
    def create_base_map(self):
        """Crée une carte de base de la France"""
        # Coordonnées approximatives de la France
        france_bounds = (-5.5, 9.6, 41.0, 51.2)  # ouest, est, sud, nord
        
        # Créer une grille pour la France
        lons = np.linspace(france_bounds[0], france_bounds[1], 50)
        lats = np.linspace(france_bounds[2], france_bounds[3], 40)
        
        return lons, lats, france_bounds
    
    def create_indicator_map(self, indicator_data, title, lons, lats):
        """Crée une carte d'indicateur"""
        # Création d'un dataset xarray
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # S'assurer que indicator_data a la bonne forme
        if indicator_data.shape != lon_grid.shape:
            indicator_data = np.resize(indicator_data, lon_grid.shape)
        
        # Créer le dataset
        ds = xr.Dataset({
            'indicator': (['lat', 'lon'], indicator_data),
            'lon': (['lat', 'lon'], lon_grid),
            'lat': (['lat', 'lon'], lat_grid)
        })
        
        # Créer la visualisation avec geoviews
        map_data = gv.QuadMesh((ds.lon, ds.lat, ds.indicator), 
                              crs=ccrs.PlateCarree())
        
        # Configuration des couleurs discrètes
        color_map = [self.stress_colors[i] for i in range(5)]
        
        map_viz = map_data.opts(
            opts.QuadMesh(
                cmap=color_map,
                clim=(0, 4),
                colorbar=True,
                colorbar_opts={'title': 'Niveau de stress'},
                width=800,
                height=600,
                title=title,
                tools=['hover'],
                projection=ccrs.PlateCarree()
            )
        )
        
        # Ajouter les contours géographiques
        coastline = gv.feature.coastline().opts(line_color='black', line_width=1)
        borders = gv.feature.borders().opts(line_color='gray', line_width=0.5)
        
        return map_viz * coastline * borders

class AgroclimaticApp(param.Parameterized):
    """Application principale des indicateurs agroclimatiques"""
    
    # Paramètres de l'interface
    selected_indicator = param.Selector(
        default="Stress thermique maximal",
        objects=[
            "Stress thermique maximal",
            "Stress thermique moyen", 
            "Perte de ponte (%)",
            "Perte de production de lait (%)",
            "Perte de GMQ - Gain en masse quotidien (%)"
        ],
        doc="Indicateur à afficher"
    )
    
    temperature_threshold = param.Number(
        default=30.0,
        bounds=(15.0, 40.0),
        step=0.5,
        doc="Seuil de température (°C)"
    )
    
    def __init__(self, **params):
        super().__init__(**params)
        
        # Initialisation des composants
        self.data_manager = DataManager()
        self.calculator = IndicatorCalculator(self.data_manager)
        self.visualizer = MapVisualizer()
        
        # Génération des données de base
        self._generate_sample_data()
        
        # Interface utilisateur
        self._create_ui()
    
    def _generate_sample_data(self):
        """Génère des données d'exemple"""
        # Créer une grille de données météorologiques simulées
        lons, lats, bounds = self.visualizer.create_base_map()
        
        # Données de température (simulation)
        np.random.seed(42)
        self.temp_data = 15 + 15 * np.random.random((len(lats), len(lons)))
        
        # Données d'humidité (simulation)
        self.humidity_data = 40 + 40 * np.random.random((len(lats), len(lons)))
        
        # Stocker les coordonnées
        self.lons = lons
        self.lats = lats
        
        # Sauvegarder en Parquet pour optimisation
        temp_df = pd.DataFrame(self.temp_data)
        humidity_df = pd.DataFrame(self.humidity_data)
        
        self.data_manager.save_to_parquet(temp_df, "temperature_data")
        self.data_manager.save_to_parquet(humidity_df, "humidity_data")
    
    def _create_ui(self):
        """Crée l'interface utilisateur"""
        # Panneau de contrôle
        controls = pn.Param(
            self,
            parameters=['selected_indicator', 'temperature_threshold'],
            widgets={
                'selected_indicator': pn.widgets.Select,
                'temperature_threshold': pn.widgets.FloatSlider
            },
            width=300,
            sizing_mode='fixed'
        )
        
        # Zone d'information
        info_pane = pn.pane.Markdown("""
        ## Indicateurs Agroclimatiques
        
        Cette application permet de visualiser différents indicateurs de stress 
        climatique pour l'élevage:
        
        - **Stress thermique**: Impact de la température sur les animaux
        - **Perte de ponte**: Réduction de la production d'œufs
        - **Perte de production laitière**: Impact sur les bovins laitiers
        - **Perte de GMQ**: Impact sur la croissance des bovins
        
        Survolez un point pour avoir plus d'informations.
        """, width=300)
        
        # Panneau latéral
        sidebar = pn.Column(
            "## Paramètres",
            controls,
            info_pane,
            width=320,
            sizing_mode='fixed'
        )
        
        # Zone principale pour la carte
        self.map_pane = pn.pane.HoloViews(
            self._create_map(),
            sizing_mode='stretch_width',
            height=700
        )
        
        # Layout principal
        self.layout = pn.template.MaterialTemplate(
            title="Indicateurs Agroclimatiques - France",
            sidebar=[sidebar],
            main=[self.map_pane],
            header_background='#2596be',
        )
    
    @param.depends('selected_indicator', 'temperature_threshold', watch=True)
    def _update_map(self):
        """Met à jour la carte quand les paramètres changent"""
        self.map_pane.object = self._create_map()
    
    def _create_map(self):
        """Crée la carte selon l'indicateur sélectionné"""
        # Calculer l'indicateur selon la sélection
        if self.selected_indicator == "Stress thermique maximal":
            indicator_data = self.calculator.calculate_heat_stress_max(
                self.temp_data, self.temperature_threshold
            )
        elif self.selected_indicator == "Stress thermique moyen":
            indicator_data = self.calculator.calculate_heat_stress_avg(
                self.temp_data, self.temperature_threshold
            )
        elif self.selected_indicator == "Perte de ponte (%)":
            indicator_data = self.calculator.calculate_laying_loss(
                self.temp_data, self.humidity_data
            )
        elif self.selected_indicator == "Perte de production de lait (%)":
            indicator_data = self.calculator.calculate_milk_production_loss(
                self.temp_data
            )
        else:  # Perte de GMQ
            indicator_data = self.calculator.calculate_daily_weight_gain_loss(
                self.temp_data, self.humidity_data
            )
        
        # Créer la carte
        return self.visualizer.create_indicator_map(
            indicator_data,
            self.selected_indicator,
            self.lons,
            self.lats
        )
    
    def serve(self, port=5007, show=True):
        """Lance l'application"""
        return pn.serve(self.layout, port=port, show=show, autoreload=True)

# Point d'entrée principal
if __name__ == "__main__":
    app = AgroclimaticApp()
    # Alternative : servir directement le layout
    pn.serve(app.layout, port=5007, show=True, autoreload=True)