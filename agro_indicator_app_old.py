"""
Application Panel pour visualiser des indicateurs agroclimatiques
Interface moderne avec cartes et application automatique des paramètres
"""

import pickle
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    
    def __init__(self, data_dir: str = "data", cache_dir: str = "cache"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
    def save_to_parquet(self, df: pd.DataFrame, filename: str) -> Path:
        """Sauvegarde DataFrame en Parquet"""
        filepath = self.data_dir / f"{filename}.parquet"
        df.to_parquet(filepath, compression='snappy')
        return filepath
    
    def load_from_parquet(self, filename: str) -> Optional[pd.DataFrame]:
        """Charge DataFrame depuis Parquet"""
        filepath = self.data_dir / f"{filename}.parquet"
        if filepath.exists():
            return pd.read_parquet(filepath)
        return None
    
    @lru_cache(maxsize=32)
    def get_cached_data(self, cache_key: str) -> Any:
        """Récupère des données du cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def set_cache(self, cache_key: str, data: Any) -> None:
        """Met en cache des données"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

class IndicatorCalculator:
    """Calculateur d'indicateurs agroclimatiques"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        
    def calculate_heat_stress_max(self, temp_data: np.ndarray, threshold: float = 30) -> np.ndarray:
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
    
    def calculate_heat_stress_avg(self, temp_data: np.ndarray, threshold: float = 25) -> np.ndarray:
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
    
    def calculate_laying_loss(self, temp_data: np.ndarray, humidity_data: np.ndarray) -> np.ndarray:
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
    
    def calculate_milk_production_loss(self, temp_data: np.ndarray) -> np.ndarray:
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
    
    def calculate_daily_weight_gain_loss(self, temp_data: np.ndarray, humidity_data: np.ndarray) -> np.ndarray:
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
            0: '#4CAF50',  # Vert - Aucun stress
            1: '#FFEB3B',  # Jaune - Faible
            2: '#FF9800',  # Orange - Modéré
            3: '#FF5722',  # Rouge orangé - Fort
            4: '#B71C1C'   # Rouge foncé - Très sévère
        }
        
        self.stress_labels = {
            0: 'Aucun stress',
            1: 'Faible',
            2: 'Modéré', 
            3: 'Fort',
            4: 'Très sévère'
        }
        
        # Légende avec valeurs numériques
        self.stress_ranges = {
            0: '0.0-68.0',
            1: '68.0-72.0',
            2: '72.0-80.0',
            3: '80.0-90.0',
            4: '90.0-99.0'
        }
    
    def create_base_map(self):
        """Crée une carte de base de la France"""
        # Coordonnées approximatives de la France
        france_bounds = (-5.5, 9.6, 41.0, 51.2)  # ouest, est, sud, nord
        
        # Créer une grille pour la France
        lons = np.linspace(france_bounds[0], france_bounds[1], 50)
        lats = np.linspace(france_bounds[2], france_bounds[3], 40)
        
        return lons, lats, france_bounds
    
    def create_indicator_map(self, indicator_data: np.ndarray, title: str, lons: np.ndarray, lats: np.ndarray):
        """Crée une carte d'indicateur avec légende améliorée"""
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
                colorbar_opts={
                    'title': 'Niveau de stress',
                    'ticker': [0, 1, 2, 3, 4],
                    'major_label_overrides': {
                        0: f'{self.stress_ranges[0]} : {self.stress_labels[0]}',
                        1: f'{self.stress_ranges[1]} : {self.stress_labels[1]}',
                        2: f'{self.stress_ranges[2]} : {self.stress_labels[2]}',
                        3: f'{self.stress_ranges[3]} : {self.stress_labels[3]}',
                        4: f'{self.stress_ranges[4]} : {self.stress_labels[4]}'
                    }
                },
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
    category = param.Selector(
        default=None,
        objects=[None, "ANIMAUX", "FEUX DE FORÊT", "MALADIES", "PRATIQUES AGRICOLES", "RAVAGEURS", "POLLENS", "VÉGÉTAUX"],
        doc="Catégorie sélectionnée"
    )
    
    selected_indicator = param.Selector(
        default=None,
        objects=[None],
        doc="Indicateur sélectionné"
    )
    
    animal_type = param.Selector(
        default=None,
        objects=[None],
        doc="Type d'animal"
    )
    
    race = param.Selector(
        default=None,
        objects=[None],
        doc="Race"
    )
    
    weather_model = param.Selector(
        default=None,
        objects=[None, "AROME", "ARPEGE", "GFS"],
        doc="Modèle météorologique"
    )
    
    def __init__(self, **params):
        super().__init__(**params)
        
        # Initialisation des composants
        self.data_manager = DataManager()
        self.calculator = IndicatorCalculator(self.data_manager)
        self.visualizer = MapVisualizer()
        
        # Données de configuration
        self.indicators_config = {
            "ANIMAUX": {
                "STRESS THERMIQUE MAXIMAL": {
                    "animals": ["VACHE LAITIÈRE", "POULE PONDEUSE", "BOVIN VIANDE"],
                    "races": {
                        "VACHE LAITIÈRE": ["PRIM'HOLSTEIN", "NORMANDE", "MONTBÉLIARDE"],
                        "POULE PONDEUSE": ["SUSSEX", "RHODE ISLAND", "LEGHORN"],
                        "BOVIN VIANDE": ["CHAROLAISE", "LIMOUSINE", "BLONDE D'AQUITAINE"]
                    }
                },
                "PERTE DE PONTE (%)": {
                    "animals": ["POULE PONDEUSE"],
                    "races": {
                        "POULE PONDEUSE": ["SUSSEX", "RHODE ISLAND", "LEGHORN"]
                    }
                },
                "PERTE DE PRODUCTION DE LAIT (%)": {
                    "animals": ["VACHE LAITIÈRE"],
                    "races": {
                        "VACHE LAITIÈRE": ["PRIM'HOLSTEIN", "NORMANDE", "MONTBÉLIARDE"]
                    }
                },
                "PERTE DE GMQ - GAIN EN MASSE QUOTIDIEN (%)": {
                    "animals": ["BOVIN VIANDE"],
                    "races": {
                        "BOVIN VIANDE": ["CHAROLAISE", "LIMOUSINE", "BLONDE D'AQUITAINE"]
                    }
                }
            }
        }
        
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
    
    def _create_category_card(self, name: str, icon: str, enabled: bool = True) -> pn.pane.HTML:
        """Crée une carte de catégorie"""
        status_class = "" if enabled else "disabled"
        status_badge = "" if enabled else '<span class="status-badge">BIENTÔT</span>'
        
        card_html = f"""
        <div class="category-card {status_class}" data-category="{name}">
            <div class="card-icon">{icon}</div>
            <div class="card-title">{name}</div>
            {status_badge}
        </div>
        """
        return pn.pane.HTML(card_html, sizing_mode='fixed', width=140, height=120)
    
    def _create_indicator_card(self, name: str, icon: str, description: str = "") -> pn.pane.HTML:
        """Crée une carte d'indicateur"""
        card_html = f"""
        <div class="indicator-card" data-indicator="{name}">
            <div class="indicator-icon">{icon}</div>
            <div class="indicator-title">{name}</div>
            {f'<div class="indicator-description">{description}</div>' if description else ''}
        </div>
        """
        return pn.pane.HTML(card_html, sizing_mode='stretch_width', height=100)
    
    def _create_weather_model_card(self, name: str, icon: str, description: str, duration: str) -> pn.pane.HTML:
        """Crée une carte de modèle météorologique"""
        card_html = f"""
        <div class="weather-card" data-model="{name}">
            <div class="weather-icon">{icon}</div>
            <div class="weather-name">{name}</div>
            <div class="weather-description">{description}</div>
            <div class="weather-duration">{duration}</div>
        </div>
        """
        return pn.pane.HTML(card_html, sizing_mode='fixed', width=140, height=120)
    
    def _create_ui(self):
        """Crée l'interface utilisateur moderne"""
        
        # CSS personnalisé
        custom_css = """
        <style>
        .main-title {
            font-size: 2.5em;
            font-weight: bold;
            color: #2E7D32;
            text-align: center;
            margin: 20px 0;
        }
        
        .agro-title { color: #8BC34A; }
        .climat-title { color: #2E7D32; }
        
        .section-title {
            font-size: 1.2em;
            font-weight: bold;
            color: #424242;
            margin: 20px 0 10px 0;
        }
        
        .category-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }
        
        .category-card {
            background: white;
            border: 2px solid #E0E0E0;
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
        }
        
        .category-card:hover {
            border-color: #4CAF50;
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.2);
            transform: translateY(-2px);
        }
        
        .category-card.selected {
            border-color: #4CAF50;
            background: #E8F5E8;
        }
        
        .category-card.disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .category-card.disabled:hover {
            transform: none;
            box-shadow: none;
            border-color: #E0E0E0;
        }
        
        .card-icon {
            font-size: 2.5em;
            margin-bottom: 8px;
        }
        
        .card-title {
            font-size: 0.9em;
            font-weight: bold;
            color: #424242;
            line-height: 1.2;
        }
        
        .status-badge {
            position: absolute;
            top: 5px;
            right: 5px;
            background: #FFC107;
            color: white;
            font-size: 0.7em;
            padding: 2px 6px;
            border-radius: 8px;
            font-weight: bold;
        }
        
        .indicator-card {
            background: white;
            border: 2px solid #E0E0E0;
            border-radius: 12px;
            padding: 15px;
            margin: 10px 0;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .indicator-card:hover {
            border-color: #2196F3;
            box-shadow: 0 4px 12px rgba(33, 150, 243, 0.2);
        }
        
        .indicator-card.selected {
            border-color: #2196F3;
            background: #E3F2FD;
        }
        
        .indicator-icon {
            font-size: 2em;
            min-width: 50px;
        }
        
        .indicator-title {
            font-weight: bold;
            color: #424242;
            flex: 1;
        }
        
        .indicator-description {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }
        
        .weather-card {
            background: white;
            border: 2px solid #E0E0E0;
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .weather-card:hover {
            border-color: #FF9800;
            box-shadow: 0 4px 12px rgba(255, 152, 0, 0.2);
            transform: translateY(-2px);
        }
        
        .weather-card.selected {
            border-color: #FF9800;
            background: #FFF3E0;
        }
        
        .weather-icon {
            font-size: 2em;
            margin-bottom: 8px;
        }
        
        .weather-name {
            font-weight: bold;
            color: #424242;
            margin-bottom: 5px;
        }
        
        .weather-description {
            font-size: 0.8em;
            color: #666;
            margin-bottom: 5px;
        }
        
        .weather-duration {
            font-size: 0.8em;
            color: #FF9800;
            font-weight: bold;
        }
        
        .parameters-section {
            background: #F5F5F5;
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .parameter-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 10px 0;
        }
        
        .parameter-label {
            font-weight: bold;
            color: #424242;
            min-width: 120px;
        }
        
        .parameter-value {
            color: #2196F3;
            font-weight: bold;
        }
        
        .download-section {
            text-align: center;
            margin: 20px 0;
        }
        
        .download-btn {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: bold;
            cursor: pointer;
            transition: background 0.3s ease;
        }
        
        .download-btn:hover {
            background: #45A049;
        }
        </style>
        """
        
        # Titre principal
        title_html = """
        <div class="main-title">
            <span class="agro-title">AGRO</span><span class="climat-title">CLIMAT</span>
        </div>
        """
        
        # Section 1 - Catégories
        categories_html = '<div class="section-title">1 - Choisir une catégorie</div>'
        
        category_cards = pn.Row(
            self._create_category_card("ANIMAUX", "🐄", True),
            self._create_category_card("FEUX DE FORÊT", "🔥", False),
            self._create_category_card("MALADIES", "🦠", False),
            self._create_category_card("PRATIQUES AGRICOLES", "🚜", False),
            self._create_category_card("RAVAGEURS", "🐛", False),
            self._create_category_card("POLLENS", "🌸", False),
            self._create_category_card("VÉGÉTAUX", "🌱", False),
            sizing_mode='stretch_width'
        )
        
        # Section 2 - Indicateurs (initialement vide)
        self.indicators_section = pn.Column(
            pn.pane.HTML('<div class="section-title">2 - Choisir un indicateur</div>'),
            sizing_mode='stretch_width'
        )
        
        # Section 3 - Paramètres (initialement vide)
        self.parameters_section = pn.Column(
            pn.pane.HTML('<div class="section-title">3 - Paramètres</div>'),
            sizing_mode='stretch_width'
        )
        
        # Section 4 - Modèle météorologique (initialement vide)
        self.weather_section = pn.Column(
            pn.pane.HTML('<div class="section-title">4 - Modèle météorologique</div>'),
            sizing_mode='stretch_width'
        )
        
        # Zone principale (carte)
        self.map_pane = pn.pane.HTML(
            "<div style='text-align: center; padding: 100px; color: #666;'>"
            "<h2>🗺️ Carte des indicateurs agroclimatiques</h2>"
            "<p>Sélectionnez les paramètres pour afficher la carte</p>"
            "</div>",
            sizing_mode='stretch_width',
            height=700
        )
        
        # Panneau de configuration
        config_panel = pn.Column(
            pn.pane.HTML(custom_css),
            pn.pane.HTML(title_html),
            pn.pane.HTML(categories_html),
            category_cards,
            self.indicators_section,
            self.parameters_section,
            self.weather_section,
            width=400,
            sizing_mode='fixed',
            scroll=True
        )
        
        # Layout principal
        self.layout = pn.template.MaterialTemplate(
            title="🌡️ Indicateurs Agroclimatiques - France",
            sidebar=[config_panel],
            main=[self.map_pane],
            header_background='#4CAF50',
            sidebar_width=420
        )
        
        # Ajouter les événements JavaScript pour l'interactivité
        self._add_javascript_events()
    
    def _add_javascript_events(self):
        """Ajoute les événements JavaScript pour l'interactivité"""
        js_code = """
        <script>
        // Gestion des clics sur les catégories
        document.addEventListener('click', function(e) {
            if (e.target.closest('.category-card') && !e.target.closest('.category-card').classList.contains('disabled')) {
                // Désélectionner toutes les autres catégories
                document.querySelectorAll('.category-card').forEach(card => {
                    card.classList.remove('selected');
                });
                
                // Sélectionner la catégorie cliquée
                e.target.closest('.category-card').classList.add('selected');
                
                // Déclencher l'événement de changement de catégorie
                const category = e.target.closest('.category-card').dataset.category;
                window.parent.postMessage({type: 'category_change', value: category}, '*');
            }
            
            // Gestion des clics sur les indicateurs
            if (e.target.closest('.indicator-card')) {
                document.querySelectorAll('.indicator-card').forEach(card => {
                    card.classList.remove('selected');
                });
                
                e.target.closest('.indicator-card').classList.add('selected');
                
                const indicator = e.target.closest('.indicator-card').dataset.indicator;
                window.parent.postMessage({type: 'indicator_change', value: indicator}, '*');
            }
            
            // Gestion des clics sur les modèles météo
            if (e.target.closest('.weather-card')) {
                document.querySelectorAll('.weather-card').forEach(card => {
                    card.classList.remove('selected');
                });
                
                e.target.closest('.weather-card').classList.add('selected');
                
                const model = e.target.closest('.weather-card').dataset.model;
                window.parent.postMessage({type: 'weather_change', value: model}, '*');
            }
        });
        </script>
        """
        
        # Ajouter le JavaScript au panneau
        self.layout.sidebar.append(pn.pane.HTML(js_code))
    
    def update_indicators(self, category: str):
        """Met à jour la liste des indicateurs en fonction de la catégorie."""
        if not category or category not in self.indicators_config:
            self.indicators_section.objects = [
                pn.pane.HTML('<div class="section-title">2 - Choisir un indicateur</div>', sizing_mode='stretch_width')
            ]
            self.selected_indicator = None
            self.param.selected_indicator.objects = [None]
            return

        indicators = self.indicators_config[category]
        self.param.selected_indicator.objects = [None] + list(indicators.keys())
        
        indicator_cards = [
            pn.pane.HTML('<div class="section-title">2 - Choisir un indicateur</div>', sizing_mode='stretch_width')
        ]
        
        indicator_icons = {
            "STRESS THERMIQUE MAXIMAL": "🌡️",
            "PERTE DE PONTE (%)": "🥚",
            "PERTE DE PRODUCTION DE LAIT (%)": "🥛",
            "PERTE DE GMQ - GAIN EN MASSE QUOTIDIEN (%)": "⚖️"
        }

        for name in indicators:
            icon = indicator_icons.get(name, "❓")
            card = self._create_indicator_card(name, icon)
            indicator_cards.append(card)
        
        self.indicators_section.objects = indicator_cards
        self.selected_indicator = None

    def update_parameters(self, indicator: str):
        """Met à jour les paramètres (animal, race) en fonction de l'indicateur."""
        if not indicator or not self.category or self.category not in self.indicators_config:
            self.parameters_section.objects = [pn.pane.HTML('<div class="section-title">3 - Paramètres</div>', sizing_mode='stretch_width')]
            self.animal_type = None
            self.race = None
            return

        config = self.indicators_config[self.category].get(indicator)
        if not config or "animals" not in config:
            self.parameters_section.objects = [pn.pane.HTML('<div class="section-title">3 - Paramètres</div>', sizing_mode='stretch_width')]
            self.animal_type = None
            self.race = None
            return

        self.param.animal_type.objects = [None] + config["animals"]
        self.animal_type = config["animals"][0] if config["animals"] else None
        
        animal_selector = pn.widgets.Select.from_param(
            self.param.animal_type, name="Type d'animal", sizing_mode='stretch_width'
        )

        race_selector = pn.widgets.Select.from_param(
            self.param.race, name="Race", sizing_mode='stretch_width'
        )

        @pn.depends(self.param.animal_type, watch=True)
        def _update_races(animal):
            if animal and animal in config.get("races", {}):
                races = [None] + config["races"][animal]
                self.param.race.objects = races
                self.race = races[1] if len(races) > 1 else None
            else:
                self.param.race.objects = [None]
                self.race = None
        
        _update_races(self.animal_type)

        self.parameters_section.objects = [
            pn.pane.HTML('<div class="section-title">3 - Paramètres</div>', sizing_mode='stretch_width'),
            pn.Column(animal_selector, race_selector, sizing_mode='stretch_width')
        ]

    def update_weather_models(self):
        """Affiche les cartes de modèles météo."""
        weather_cards = pn.Row(
            self._create_weather_model_card("AROME", "🌀", "Maille fine", "J+2"),
            self._create_weather_model_card("ARPEGE", "🌍", "Global", "J+4"),
            self._create_weather_model_card("GFS", "🇺🇸", "Américain", "J+16"),
            sizing_mode='stretch_width',
            css_classes=['category-grid']
        )
        self.weather_section.objects = [
            pn.pane.HTML('<div class="section-title">4 - Modèle météorologique</div>', sizing_mode='stretch_width'),
            weather_cards
        ]

    def _initial_map_message(self):
        return pn.pane.HTML(
            "<div style='text-align: center; padding: 100px; color: #666;'>"
            "<h2>🗺️ Carte des indicateurs agroclimatiques</h2>"
            "<p>Sélectionnez les paramètres pour afficher la carte</p>"
            "</div>",
            sizing_mode='stretch_width',
            height=700
        )

    def update_map(self):
        """Met à jour la carte avec les données de l'indicateur."""
        if not all([self.category, self.selected_indicator, self.animal_type, self.race, self.weather_model]):
             self.map_pane.object = self._initial_map_message()
             return

        indicator = self.selected_indicator
        title = f"{indicator} - {self.animal_type} ({self.race}) - Modèle {self.weather_model}"
        
        indicator_data = None
        if indicator == "STRESS THERMIQUE MAXIMAL":
            indicator_data = self.calculator.calculate_heat_stress_max(self.temp_data, threshold=30.0)
        elif indicator == "PERTE DE PONTE (%)":
            indicator_data = self.calculator.calculate_laying_loss(self.temp_data, self.humidity_data)
        elif indicator == "PERTE DE PRODUCTION DE LAIT (%)":
            indicator_data = self.calculator.calculate_milk_production_loss(self.temp_data)
        elif indicator == "PERTE DE GMQ - GAIN EN MASSE QUOTIDIEN (%)":
            indicator_data = self.calculator.calculate_daily_weight_gain_loss(self.temp_data, self.humidity_data)
        
        if indicator_data is not None:
            map_viz = self.visualizer.create_indicator_map(indicator_data, title, self.lons, self.lats)
            self.map_pane.object = map_viz
        else:
            self.map_pane.object = pn.pane.HTML(
                f"<div style='text-align: center; padding: 50px;'>Indicateur '{indicator}' non implémenté.</div>",
                sizing_mode='stretch_width'
            )

    def _setup_js_comms(self):
        """Met en place la communication JavaScript -> Python."""
        def message_handler(msg):
            msg_type = msg.get('type')
            value = msg.get('value')
            
            if msg_type == 'category_change':
                self.category = value
                self.selected_indicator = None
                self.animal_type = None
                self.race = None
                self.weather_model = None
                self.update_indicators(value)
                self.update_parameters(None)
                self.weather_section.objects = [pn.pane.HTML('<div class="section-title">4 - Modèle météorologique</div>', sizing_mode='stretch_width')]
                self.map_pane.object = self._initial_map_message()

            elif msg_type == 'indicator_change':
                self.selected_indicator = value
                self.animal_type = None
                self.race = None
                self.weather_model = None
                self.update_parameters(value)
                if value:
                    self.update_weather_models()
                else:
                    self.weather_section.objects = [pn.pane.HTML('<div class="section-title">4 - Modèle météorologique</div>', sizing_mode='stretch_width')]
                self.map_pane.object = self._initial_map_message()

            elif msg_type == 'weather_change':
                self.weather_model = value
                self.update_map()

        if pn.state.curdoc:
            pn.state.js_on_message(message_handler)

    def view(self):
        """Retourne le layout de l'application."""
        self._setup_js_comms()
        # Reset state on first view
        self.map_pane.object = self._initial_map_message()
        self.update_indicators(None)
        self.update_parameters(None)
        self.weather_section.objects = [pn.pane.HTML('<div class="section-title">4 - Modèle météorologique</div>', sizing_mode='stretch_width')]
        return self.layout

# if __name__ == "__main__":
# Pour lancer l'application: panel serve agro_indicator_app.py --autoreload
app = AgroclimaticApp()
app.view().servable()
