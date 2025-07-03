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
    
    # Étape 1: Sélection de catégorie
    selected_category = param.Selector(
        default="ANIMAUX",
        objects=[
            "ANIMAUX",
            "FEUX DE FORÊT", 
            "MALADIES",
            "PRATIQUES AGRICOLES",
            "RAVAGEURS",
            "POLLENS",
            "VÉGÉTAUX"
        ],
        doc="Catégorie d'activité agricole"
    )
    
    # Étape 2: Sélection d'indicateur (KPI) - dépend de la catégorie
    selected_indicator = param.Selector(
        default="STRESS THERMIQUE MAXIMAL",
        objects=[
            "STRESS THERMIQUE MAXIMAL",
            "PERTE DE PONTE (%)",
            "PERTE DE PRODUCTION DE LAIT (%)",
            "PERTE DE GMQ - GAIN EN MASSE QUOTIDIEN (%)"
        ],
        doc="Indicateur à afficher"
    )
    
    # Étape 3: Paramètres - Type d'animal
    animal_type = param.Selector(
        default="VACHE LAITIÈRE",
        objects=[
            "VACHE LAITIÈRE",
            "VACHE ALLAITANTE",
            "POULES PONDEUSES",
            "BOVINS À L'ENGRAISSEMENT"
        ],
        doc="Type d'animal"
    )
    
    # Race d'animal
    animal_breed = param.Selector(
        default="PRIM'HOLSTEIN",
        objects=[
            "PRIM'HOLSTEIN",
            "MONTBÉLIARDE",
            "NORMANDE",
            "CHAROLAISE"
        ],
        doc="Race de l'animal"
    )
    
    # Étape 4: Modèle météorologique
    weather_model = param.Selector(
        default="AROME",
        objects=[
            "AROME",
            "ARPEGE",
            "GFS"
        ],
        doc="Modèle météorologique"
    )
    
    # Paramètres techniques
    temperature_threshold = param.Number(
        default=30.0,
        bounds=(15.0, 40.0),
        step=0.5,
        doc="Seuil de température (°C)"
    )
    
    # État de l'interface - pour contrôler les étapes
    current_step = param.Integer(default=1, bounds=(1, 5))
    show_step_2 = param.Boolean(default=False)
    show_step_3 = param.Boolean(default=False)
    show_step_4 = param.Boolean(default=False)
    
    def __init__(self, **params):
        super().__init__(**params)
        
        # Initialisation des composants
        self.data_manager = DataManager()
        self.calculator = IndicatorCalculator(self.data_manager)
        self.visualizer = MapVisualizer()
        
        # Définition des indicateurs par catégorie
        self.indicators_by_category = {
            "ANIMAUX": [
                "STRESS THERMIQUE MAXIMAL",
                "PERTE DE PONTE (%)",
                "PERTE DE PRODUCTION DE LAIT (%)",
                "PERTE DE GMQ - GAIN EN MASSE QUOTIDIEN (%)"
            ],
            "FEUX DE FORÊT": [
                "RISQUE D'INCENDIE",
                "INDICE MÉTÉOROLOGIQUE"
            ],
            "MALADIES": [
                "PROPAGATION PATHOGÈNES",
                "CONDITIONS FAVORABLES"
            ],
            "PRATIQUES AGRICOLES": [
                "FENÊTRE DE TIR",
                "CONDITIONS DE TRAVAIL"
            ],
            "RAVAGEURS": [
                "DÉVELOPPEMENT INSECTES",
                "CYCLES BIOLOGIQUES"
            ],
            "POLLENS": [
                "CONCENTRATION POLLENS",
                "ALLERGÈNES"
            ],
            "VÉGÉTAUX": [
                "STRESS HYDRIQUE",
                "ÉCHAUDAGE",
                "GEL"
            ]
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
    
    def _create_ui(self):
        """Crée l'interface utilisateur step-by-step"""
        
        # Étape 1: Choisir une catégorie
        step1_title = pn.pane.Markdown("## 1 - Choisir une catégorie", margin=(10, 5))
        step1_widget = pn.Param(
            self,
            parameters=['selected_category'],
            widgets={'selected_category': pn.widgets.RadioButtonGroup},
            width=300,
            show_name=False
        )
        
        # Étape 2: Choisir un indicateur (initialement masqué)
        step2_title = pn.pane.Markdown("## 2 - Choisir un indicateur", margin=(10, 5))
        step2_widget = pn.Param(
            self,
            parameters=['selected_indicator'],
            widgets={'selected_indicator': pn.widgets.RadioButtonGroup},
            width=300,
            show_name=False
        )
        
        # Étape 3: Paramètres (initialement masqué)
        step3_title = pn.pane.Markdown("## 3 - Paramètres", margin=(10, 5))
        step3_animal_type = pn.Param(
            self,
            parameters=['animal_type'],
            widgets={'animal_type': pn.widgets.Select},
            width=300,
            show_name=False
        )
        step3_animal_breed = pn.Param(
            self,
            parameters=['animal_breed'],
            widgets={'animal_breed': pn.widgets.Select},
            width=300,
            show_name=False
        )
        
        # Étape 4: Modèle météorologique (initialement masqué)
        step4_title = pn.pane.Markdown("## 4 - Modèle météorologique", margin=(10, 5))
        step4_widget = pn.Param(
            self,
            parameters=['weather_model'],
            widgets={'weather_model': pn.widgets.RadioButtonGroup},
            width=300,
            show_name=False
        )
        
        # Conteneurs conditionnels pour les étapes
        self.step2_container = pn.Column(step2_title, step2_widget, visible=False)
        self.step3_container = pn.Column(step3_title, step3_animal_type, step3_animal_breed, visible=False)
        self.step4_container = pn.Column(step4_title, step4_widget, visible=False)
        
        # Zone d'information
        info_pane = pn.pane.Markdown("""
        ## AGRO CLIMAT
        
        Suivez les étapes pour configurer votre analyse:
        
        1. **Catégorie**: Choisissez le domaine d'activité
        2. **Indicateur**: Sélectionnez l'indicateur à analyser  
        3. **Paramètres**: Configurez les paramètres spécifiques
        4. **Modèle**: Choisissez le modèle météorologique
        
        La carte se mettra à jour automatiquement.
        """, width=300)
        
        # Panneau latéral avec les étapes
        sidebar = pn.Column(
            step1_title,
            step1_widget,
            self.step2_container,
            self.step3_container, 
            self.step4_container,
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
            title="AGRO CLIMAT - Indicateurs Agroclimatiques",
            sidebar=[sidebar],
            main=[self.map_pane],
            header_background='#2596be',
        )
    
    @param.depends('selected_category', watch=True)
    def _update_category(self):
        """Met à jour les indicateurs disponibles selon la catégorie sélectionnée"""
        # Mettre à jour les indicateurs disponibles
        available_indicators = self.indicators_by_category.get(self.selected_category, [])
        self.param.selected_indicator.objects = available_indicators
        if available_indicators:
            self.selected_indicator = available_indicators[0]
        
        # Afficher l'étape 2
        self.step2_container.visible = True
        
    @param.depends('selected_indicator', watch=True)
    def _update_indicator(self):
        """Met à jour l'interface selon l'indicateur sélectionné"""
        if self.selected_indicator:
            # Afficher l'étape 3 si on est dans la catégorie ANIMAUX
            if self.selected_category == "ANIMAUX":
                self.step3_container.visible = True
            else:
                self.step3_container.visible = False
                # Afficher directement l'étape 4 pour les autres catégories
                self.step4_container.visible = True
        
        # Mettre à jour la carte
        self.map_pane.object = self._create_map()
    
    @param.depends('animal_type', 'animal_breed', watch=True)
    def _update_parameters(self):
        """Met à jour l'interface selon les paramètres sélectionnés"""
        if self.animal_type and self.animal_breed:
            # Afficher l'étape 4
            self.step4_container.visible = True
        
        # Mettre à jour la carte
        self.map_pane.object = self._create_map()
    
    @param.depends('weather_model', watch=True)
    def _update_weather_model(self):
        """Met à jour la carte selon le modèle météorologique"""
        self.map_pane.object = self._create_map()
    
    def _create_map(self):
        """Crée la carte selon l'indicateur sélectionné"""
        # Calculer l'indicateur selon la sélection
        if self.selected_indicator == "STRESS THERMIQUE MAXIMAL":
            indicator_data = self.calculator.calculate_heat_stress_max(
                self.temp_data, self.temperature_threshold
            )
        elif self.selected_indicator == "PERTE DE PONTE (%)":
            indicator_data = self.calculator.calculate_laying_loss(
                self.temp_data, self.humidity_data
            )
        elif self.selected_indicator == "PERTE DE PRODUCTION DE LAIT (%)":
            indicator_data = self.calculator.calculate_milk_production_loss(
                self.temp_data
            )
        elif self.selected_indicator == "PERTE DE GMQ - GAIN EN MASSE QUOTIDIEN (%)":
            indicator_data = self.calculator.calculate_daily_weight_gain_loss(
                self.temp_data, self.humidity_data
            )
        else:
            # Pour les autres indicateurs, utiliser des données simulées
            indicator_data = self.calculator.calculate_heat_stress_max(
                self.temp_data, self.temperature_threshold
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

app = AgroclimaticApp()
# Alternative : servir directement le layout
pn.serve(app.layout, port=5007, show=True, autoreload=True)
