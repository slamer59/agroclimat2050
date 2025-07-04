"""
Application Panel pour visualiser des indicateurs agroclimatiques
Architecture modulaire avec séparation UI/calculs et cache efficace
"""

import datetime
import threading
import time
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
from panel.io.profile import profile

# Debug mode flag - Set to False for production
DEBUG_PROFILING = True  # Change to False to disable profiling

warnings.filterwarnings('ignore')

# Configuration Panel
pn.extension('tabulator')
hv.extension('bokeh')
gv.extension('bokeh')

# Base classes for shared parameters
class AnimalParams(param.Parameterized):
    animal_type = param.Selector(default="VACHE LAITIÈRE", objects=[
        "VACHE LAITIÈRE", "Vache allaitante", "Poule de chair", "Poule pondeuse"
    ])
    simulation_mode = param.Boolean(default=False)
    temperature_offset = param.Number(default=0, bounds=(-5, 5))

class DiseaseParams(param.Parameterized):
    growth_factor = param.Number(default=1.0, bounds=(0.5, 2.0))

# Subclass for specific animal types
class DairyCowParams(AnimalParams):
    breed = param.Selector(default="PRIM'HOLSTEIN", objects=[
        "PRIM'HOLSTEIN", "JERSEY", "HOLSTEIN", "NORMANDE"
    ])
    lactation_stage = param.Selector(default="Milieu", objects=[
        "Début", "Milieu", "Fin"
    ])

class PoultryParams(AnimalParams):
    housing_type = param.Selector(default="Cage", objects=[
        "Cage", "Plein air", "Bio"
    ])
    flock_size = param.Integer(default=10000, bounds=(100, 100000))


class DataManager:
    """Gestionnaire de données avec cache et stockage Parquet optimisé"""
    
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
    
    @lru_cache(maxsize=512)  # Augmenté pour plus d'efficacité
    def get_cached_data(self, cache_key):
        """Récupère des données du cache Parquet"""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                return df.values  # Retourner directement le numpy array
            except Exception:
                return None
        return None
    
    def set_cache(self, cache_key, data):
        """Met en cache des données en Parquet"""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        try:
            # Convertir en DataFrame si nécessaire
            if hasattr(data, 'shape'):  # numpy array
                df = pd.DataFrame(data.reshape(-1, 1) if data.ndim == 1 else data)
            else:
                df = pd.DataFrame([data])
            df.to_parquet(cache_file, compression='snappy')
        except Exception as e:
            print(f"⚠️ Erreur de cache pour {cache_key}: {e}")
    
    def clear_cache(self):
        """Vide le cache Parquet"""
        for cache_file in self.cache_dir.glob("*.parquet"):
            cache_file.unlink()
        print("🗑️ Cache Parquet vidé")

class IndicatorCalculator:
    """Calculateur d'indicateurs agroclimatiques"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        
    @profile('heat_stress_max_calculation', engine='pyinstrument') if DEBUG_PROFILING else lambda f: f
    def calculate_heat_stress_max(self, temp_data, threshold=30):
        """Calcule le stress thermique maximal - Version ultra-optimisée"""
        # Cache key simplifié pour de meilleures performances
        cache_key = f"heat_stress_max_{threshold:.1f}_{temp_data.shape}"
        cached = self.data_manager.get_cached_data(cache_key)
        if cached is not None:
            # Reshape pour correspondre à la forme originale
            return cached.reshape(temp_data.shape).astype(np.int8)
            
        # Optimisation ultra-rapide avec masquage direct
        stress_factor = (temp_data - threshold) / threshold * 100
        
        # Classification vectorisée en une seule opération
        stress_classes = (
            (stress_factor > 0).astype(np.int8) +
            (stress_factor > 10).astype(np.int8) +
            (stress_factor > 25).astype(np.int8) +
            (stress_factor > 50).astype(np.int8)
        )
        
        self.data_manager.set_cache(cache_key, stress_classes)
        return stress_classes
    
    @profile('heat_stress_avg_calculation', engine='pyinstrument') if DEBUG_PROFILING else lambda f: f
    def calculate_heat_stress_avg(self, temp_data, threshold=25):
        """Calcule le stress thermique moyen - Version ultra-optimisée"""
        # Cache key simplifié
        cache_key = f"heat_stress_avg_{threshold:.1f}_{temp_data.shape}"
        cached = self.data_manager.get_cached_data(cache_key)
        if cached is not None:
            return cached.reshape(temp_data.shape).astype(np.int8)
            
        # Calcul direct optimisé
        stress_factor = (temp_data - threshold) / threshold * 50
        
        # Classification vectorisée ultra-rapide
        stress_classes = (
            (stress_factor > 0).astype(np.int8) +
            (stress_factor > 5).astype(np.int8) +
            (stress_factor > 15).astype(np.int8) +
            (stress_factor > 30).astype(np.int8)
        )
        
        self.data_manager.set_cache(cache_key, stress_classes)
        return stress_classes
    
    @profile('laying_loss_calculation', engine='pyinstrument') if DEBUG_PROFILING else lambda f: f
    def calculate_laying_loss(self, temp_data, humidity_data):
        """Calcule la perte de ponte - Version ultra-optimisée"""
        # Cache key simplifié
        cache_key = f"laying_loss_{temp_data.shape}_{humidity_data.shape}"
        cached = self.data_manager.get_cached_data(cache_key)
        if cached is not None:
            return cached.reshape(temp_data.shape).astype(np.int8)
            
        # Calcul direct optimisé - éviter les allocations intermédiaires
        total_stress = (np.abs(temp_data - 20) / 20 + np.abs(humidity_data - 60) / 60) * 50
        
        # Classification ultra-rapide
        stress_classes = (
            (total_stress > 5).astype(np.int8) +
            (total_stress > 15).astype(np.int8) +
            (total_stress > 30).astype(np.int8) +
            (total_stress > 50).astype(np.int8)
        )
        
        self.data_manager.set_cache(cache_key, stress_classes)
        return stress_classes
    
    @profile('milk_production_loss_calculation', engine='pyinstrument') if DEBUG_PROFILING else lambda f: f
    def calculate_milk_production_loss(self, temp_data):
        """Calcule la perte de production de lait"""
        # Enhanced cache key
        data_hash = hash(temp_data.tobytes()) if hasattr(temp_data, 'tobytes') else hash(str(temp_data))
        cache_key = f"milk_production_loss_{data_hash}_{temp_data.shape}"
        cached = self.data_manager.get_cached_data(cache_key)
        if cached is not None:
            return cached
            
        # Vectorized calculation
        optimal_temp = 18
        stress_values = np.where(temp_data > optimal_temp,
                               (temp_data - optimal_temp) / optimal_temp * 60, 0)
        
        # Optimized vectorized classification
        stress_classes = np.zeros_like(stress_values, dtype=np.int8)
        stress_classes[stress_values > 0] = 1
        stress_classes[stress_values > 10] = 2
        stress_classes[stress_values > 25] = 3
        stress_classes[stress_values > 40] = 4
        
        self.data_manager.set_cache(cache_key, stress_classes)
        return stress_classes
    
    @profile('daily_weight_gain_loss_calculation', engine='pyinstrument') if DEBUG_PROFILING else lambda f: f
    def calculate_daily_weight_gain_loss(self, temp_data, humidity_data):
        """Calcule la perte de GMQ (Gain de Masse Quotidien)"""
        # Enhanced cache key
        temp_hash = hash(temp_data.tobytes()) if hasattr(temp_data, 'tobytes') else hash(str(temp_data))
        humid_hash = hash(humidity_data.tobytes()) if hasattr(humidity_data, 'tobytes') else hash(str(humidity_data))
        cache_key = f"daily_weight_gain_loss_{temp_hash}_{humid_hash}_{temp_data.shape}"
        cached = self.data_manager.get_cached_data(cache_key)
        if cached is not None:
            return cached
            
        # Vectorized calculation
        optimal_temp = 16
        optimal_humidity = 65
        
        temp_factor = np.where(temp_data > optimal_temp,
                             (temp_data - optimal_temp) / optimal_temp, 0)
        humidity_factor = np.where(humidity_data > optimal_humidity,
                                 (humidity_data - optimal_humidity) / optimal_humidity, 0)
        
        combined_stress = (temp_factor + humidity_factor) * 40
        
        # Optimized vectorized classification
        stress_classes = np.zeros_like(combined_stress, dtype=np.int8)
        stress_classes[combined_stress > 0] = 1
        stress_classes[combined_stress > 8] = 2
        stress_classes[combined_stress > 20] = 3
        stress_classes[combined_stress > 35] = 4
        
        self.data_manager.set_cache(cache_key, stress_classes)
        return stress_classes

class MapVisualizer:
    """Visualisateur de cartes pour les indicateurs"""
    
    def __init__(self):
        # Couleurs selon l'image de légende
        self.stress_colors = {
            0: '#00ff00',  # Vert - Aucun stress (0.0-68.0)
            1: '#ffff00',  # Jaune - Faible (68.0-72.0)
            2: '#ffa500',  # Orange - Modéré (72.0-80.0)
            3: '#ff4500',  # Rouge orangé - Fort (80.0-90.0)
            4: '#8b0000'   # Rouge foncé - Très sévère (90.0-99.0)
        }
        
        # Labels avec valeurs selon l'image
        self.stress_labels = {
            0: '0.0-68.0 : Aucun stress',
            1: '68.0-72.0 : Faible',
            2: '72.0-80.0 : Modéré', 
            3: '80.0-90.0 : Fort',
            4: '90.0-99.0 : Très sévère'
        }
        
        # Valeurs numériques pour le hover
        self.stress_ranges = {
            0: (0.0, 68.0),
            1: (68.0, 72.0),
            2: (72.0, 80.0),
            3: (80.0, 90.0),
            4: (90.0, 99.0)
        }
    
    def create_base_map(self):
        """Crée une carte de base de la France"""
        # Coordonnées approximatives de la France
        france_bounds = (-5.5, 9.6, 41.0, 51.2)  # ouest, est, sud, nord
        
        # Créer une grille pour la France
        lons = np.linspace(france_bounds[0], france_bounds[1], 50)
        lats = np.linspace(france_bounds[2], france_bounds[3], 40)
        
        return lons, lats, france_bounds
    
    @profile('indicator_map_creation', engine='pyinstrument') if DEBUG_PROFILING else lambda f: f
    def create_indicator_map(self, indicator_data, title, lons, lats):
        """Crée une carte d'indicateur avec hover values"""
        # Création d'un dataset xarray
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # S'assurer que indicator_data a la bonne forme
        if indicator_data.shape != lon_grid.shape:
            indicator_data = np.resize(indicator_data, lon_grid.shape)
        
        # Générer des valeurs réelles pour le hover basées sur les classes
        real_values = np.zeros_like(indicator_data, dtype=float)
        for i in range(5):
            mask = indicator_data == i
            min_val, max_val = self.stress_ranges[i]
            # Générer des valeurs aléatoires dans la plage appropriée
            real_values[mask] = np.random.uniform(min_val, max_val, np.sum(mask))
        
        # Créer la visualisation avec geoviews - format simplifié
        map_data = gv.QuadMesh((lons, lats, indicator_data), 
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
        objects=["STRESS THERMIQUE MAXIMAL", "STRESS THERMIQUE MOYEN", 
                         "PERTE DE PONTE (%)", "PERTE DE PRODUCTION DE LAIT (%)"],
        doc="Indicateur à afficher"
    )
    
    # Étape 3: Paramètres - Type d'animal
    animal_params = param.ClassSelector(class_=AnimalParams, default=AnimalParams())
    disease_params = param.ClassSelector(class_=DiseaseParams, default=DiseaseParams())
    
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
    
    # Debouncing for parameter updates
    _update_timer = param.Parameter(default=None)
    _pending_update = param.Boolean(default=False)
    
    # Attributes
    data_manager = param.ClassSelector(DataManager)
    calculator = param.ClassSelector(IndicatorCalculator)
    visualizer = param.ClassSelector(MapVisualizer)
    animal_params_pane = param.Parameter()
    step2_container = param.Parameter()
    step3_container = param.Parameter()
    step4_container = param.Parameter()
    map_pane = param.Parameter()
    layout = param.Parameter()
    temp_data = param.Parameter()
    humidity_data = param.Parameter()
    lons = param.Parameter()
    lats = param.Parameter()

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
        self._update_indicators()
    
    def _generate_sample_data(self):
        """Charge ou génère des données d'exemple optimisées"""
        # Essayer de charger les données existantes depuis Parquet
        temp_df = self.data_manager.load_from_parquet("temperature_data")
        humidity_df = self.data_manager.load_from_parquet("humidity_data")
        coords_df = self.data_manager.load_from_parquet("coordinates_data")
        
        if temp_df is not None and humidity_df is not None and coords_df is not None:
            print("📁 Chargement des données depuis le cache Parquet...")
            # Données trouvées dans le cache
            self.temp_data = temp_df.values.astype(np.float32)
            self.humidity_data = humidity_df.values.astype(np.float32)
            
            # Reconstituer les coordonnées
            lons_len = int(coords_df['lons_len'].iloc[0])
            lats_len = int(coords_df['lats_len'].iloc[0])
            lons_start = coords_df['lons_start'].iloc[0]
            lons_end = coords_df['lons_end'].iloc[0]
            lats_start = coords_df['lats_start'].iloc[0]
            lats_end = coords_df['lats_end'].iloc[0]
            
            self.lons = np.linspace(lons_start, lons_end, lons_len)
            self.lats = np.linspace(lats_start, lats_end, lats_len)
            print("✅ Données chargées depuis le cache - démarrage instantané!")
        else:
            print("🛠️ Génération des données d'exemple...")
            # Créer une grille plus petite pour des calculs plus rapides
            lons, lats, bounds = self.visualizer.create_base_map()
            
            # Réduire la taille de la grille pour des calculs plus rapides
            # 50x40 -> 25x20 (4x moins de points)
            if len(lons) > 25:
                lons = lons[::2]  # Prendre un point sur deux
            if len(lats) > 20:
                lats = lats[::2]  # Prendre un point sur deux
            
            # Données de température (simulation) - dtype optimisé
            np.random.seed(42)
            self.temp_data = (15 + 15 * np.random.random((len(lats), len(lons)))).astype(np.float32)
            
            # Données d'humidité (simulation) - dtype optimisé
            self.humidity_data = (40 + 40 * np.random.random((len(lats), len(lons)))).astype(np.float32)
            
            # Stocker les coordonnées
            self.lons = lons
            self.lats = lats
            
            # Sauvegarder en Parquet pour la prochaine fois
            temp_df = pd.DataFrame(self.temp_data)
            humidity_df = pd.DataFrame(self.humidity_data)
            
            # Créer un DataFrame pour les coordonnées avec la bonne structure
            coords_df = pd.DataFrame({
                'lons_len': [len(self.lons)],
                'lats_len': [len(self.lats)],
                'lons_start': [self.lons[0]],
                'lons_end': [self.lons[-1]], 
                'lats_start': [self.lats[0]],
                'lats_end': [self.lats[-1]]
            })
            
            self.data_manager.save_to_parquet(temp_df, "temperature_data")
            self.data_manager.save_to_parquet(humidity_df, "humidity_data")
            self.data_manager.save_to_parquet(coords_df, "coordinates_data")
            print("💾 Données sauvegardées en cache Parquet")
        
        # Précalculer les résultats pour les seuils communs
        self._precompute_common_scenarios()
    
    def _precompute_common_scenarios(self):
        """Précalcule les scénarios courants pour des réponses instantanées"""
        print("🛠️ Précalcul des scénarios courants...")
        
        # Seuils de température courants
        common_thresholds = [20, 25, 30, 35]
        
        for threshold in common_thresholds:
            # Précalculer le stress thermique maximal
            self.calculator.calculate_heat_stress_max(self.temp_data, threshold)
            # Précalculer le stress thermique moyen
            self.calculator.calculate_heat_stress_avg(self.temp_data, threshold)
        
        # Précalculer les autres indicateurs
        self.calculator.calculate_laying_loss(self.temp_data, self.humidity_data)
        self.calculator.calculate_milk_production_loss(self.temp_data)
        self.calculator.calculate_daily_weight_gain_loss(self.temp_data, self.humidity_data)
        
        print("\u2705 Précalcul terminé - réponses instantanées disponibles")
    
    def _create_ui(self):
        """Crée l'interface utilisateur step-by-step"""
        
        # Étape 1: Choisir une catégorie
        step1_widget = pn.Param(
            self,
            parameters=['selected_category'],
            widgets={'selected_category': pn.widgets.RadioButtonGroup},
            show_name=False,
            width=280,
            margin=(5, 5)
        )
        
        step1_card = pn.Card(
            step1_widget,
            title="1 - Choisir une catégorie",
            width=320,
            margin=(5, 5),
            styles={'background': '#f8f9fa'}
        )
        
        # Étape 2: Choisir un indicateur (initialement masqué)
        step2_widget = pn.Param(
            self,
            parameters=['selected_indicator'],
            widgets={'selected_indicator': pn.widgets.RadioButtonGroup},
            show_name=False,
            width=280,
            margin=(5, 5)
        )
        
        step2_card = pn.Card(
            step2_widget,
            title="2 - Choisir un indicateur",
            width=320,
            margin=(5, 5),
            styles={'background': '#f8f9fa'}
        )
        
        # Étape 3: Paramètres (initialement masqué)
        self.animal_params_pane = self._create_animal_params_panel()
        step3_card = pn.Card(
            self.animal_params_pane,
            title="3 - Paramètres",
            width=320,
            margin=(5, 5),
            styles={'background': '#f8f9fa'}
        )
        
        # Conteneurs conditionnels pour les étapes
        self.step2_container = pn.Column(step2_card, visible=False)
        self.step3_container = pn.Column(step3_card, visible=False)
        self.step4_container = pn.Column(
            pn.Card(
                pn.Param(self, parameters=['weather_model'], show_name=False, width=280),
                title="4 - Modèle météorologique",
                width=320,
                margin=(5, 5),
                styles={'background': '#f8f9fa'}
            ), 
            visible=False
        )
        
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
            step1_card,
            self.step2_container,
            self.step3_container, 
            self.step4_container,
            info_pane,
            width=340,
            sizing_mode='fixed'
        )
        
        # Zone principale pour la carte
        self.map_pane = pn.pane.HoloViews(
            self._create_map(),
            sizing_mode='stretch_width',
            height=700
        )
        
        # Layout principal avec style amélioré
        self.layout = pn.template.MaterialTemplate(
            title="AGRO CLIMAT - Indicateurs Agroclimatiques",
            sidebar=[sidebar],
            main=[self.map_pane],
            header_background='#2596be',
            sidebar_width=340
        )
    
    @param.depends('selected_category', watch=True)
    @profile('indicator_update', engine='pyinstrument') if DEBUG_PROFILING else lambda f: f
    def _update_indicators(self):
        """Met à jour les indicateurs disponibles selon la catégorie sélectionnée"""
        if self.selected_category == "ANIMAUX":
            self.param.selected_indicator.objects = self.indicators_by_category.get(self.selected_category, [])
            if not self.selected_indicator or self.selected_indicator not in self.param.selected_indicator.objects:
                self.selected_indicator = self.param.selected_indicator.objects[0] if self.param.selected_indicator.objects else None
        elif self.selected_category == "MALADIES":
            self.param.selected_indicator.objects = self.indicators_by_category.get(self.selected_category, [])
            if not self.selected_indicator or self.selected_indicator not in self.param.selected_indicator.objects:
                self.selected_indicator = self.param.selected_indicator.objects[0] if self.param.selected_indicator.objects else None
        else:
            self.param.selected_indicator.objects = []
            self.selected_indicator = None
        
        # Afficher l'étape 2
        self.step2_container.visible = True
        
    @profile('debounced_map_update', engine='pyinstrument') if DEBUG_PROFILING else lambda f: f
    def _debounced_update_map(self):
        """Met à jour la carte après un délai (debouncing)"""
        if self._update_timer is not None:
            self._update_timer.cancel()
        
        def delayed_update():
            time.sleep(0.3)  # 300ms delay
            if self._pending_update:
                self.map_pane.object = self._create_map()
                self._pending_update = False
        
        self._pending_update = True
        self._update_timer = threading.Timer(0.3, delayed_update)
        self._update_timer.start()
    
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
        
        # Mettre à jour la carte avec debouncing
        self._debounced_update_map()
    
    def _create_animal_params_panel(self):
        """Crée le panneau de paramètres animaux selon le design"""
        # Animal type dropdown with styling
        animal_type_select = pn.widgets.Select(
            name="Animal type",
            value="VACHE LAITIÈRE",
            options=[
                "VACHE LAITIÈRE",
                "Vache allaitante", 
                "Poule de chair",
                "Poule pondeuse"
            ],
            width=280,
            margin=(5, 5),
            styles={'background': '#ffffff'}
        )
        
        # Simulation mode checkbox
        simulation_checkbox = pn.widgets.Checkbox(
            name="Simulation mode",
            value=False,
            margin=(5, 5),
            styles={'color': '#495057'}
        )
        
        # Temperature offset slider with blue styling
        temp_offset_slider = pn.widgets.FloatSlider(
            name="Temperature offset: 0",
            start=-5,
            end=5,
            step=0.1,
            value=0,
            width=280,
            margin=(5, 5),
            styles={'color': '#007bff'}
        )
        
        # Store references for updates
        self.animal_type_select = animal_type_select
        self.simulation_checkbox = simulation_checkbox
        self.temp_offset_slider = temp_offset_slider
        
        return pn.Column(
            animal_type_select,
            simulation_checkbox,
            temp_offset_slider,
            width=300,
            margin=(5, 5)
        )
    
    
    @param.depends('weather_model', watch=True)
    def _update_weather_model(self):
        """Met à jour la carte selon le modèle météorologique"""
        self._debounced_update_map()
    
    @param.depends('temperature_threshold', watch=True)
    def _update_temperature_threshold(self):
        """Met à jour la carte selon le seuil de température"""
        self._debounced_update_map()
    
    @param.depends('animal_params.animal_type', watch=True)
    def _update_animal_type(self):
        """Met à jour les paramètres spécifiques à l'animal"""
        current_params = self.animal_params
        if current_params.animal_type == "VACHE LAITIÈRE":
            if not isinstance(current_params, DairyCowParams):
                self.animal_params = DairyCowParams(
                    animal_type=current_params.animal_type,
                    simulation_mode=current_params.simulation_mode,
                    temperature_offset=current_params.temperature_offset
                )
        elif current_params.animal_type in ["Poule de chair", "Poule pondeuse"]:
            if not isinstance(current_params, PoultryParams):
                self.animal_params = PoultryParams(
                    animal_type=current_params.animal_type,
                    simulation_mode=current_params.simulation_mode,
                    temperature_offset=current_params.temperature_offset
                )
        else:
            if not type(current_params) is AnimalParams:
                self.animal_params = AnimalParams(
                    animal_type=current_params.animal_type,
                    simulation_mode=current_params.simulation_mode,
                    temperature_offset=current_params.temperature_offset
                )
        
        if self.animal_params is not current_params:
            self.animal_params_pane.object = self.animal_params
    
    @profile('map_creation', engine='pyinstrument') if DEBUG_PROFILING else lambda f: f
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
            self.selected_indicator if self.selected_indicator else "",
            self.lons,
            self.lats
        )
    
app = AgroclimaticApp()

# Debug profiling utilities
if DEBUG_PROFILING:
    def get_performance_report():
        """Get performance profiling report for all tracked functions"""
        profiles = [
            'heat_stress_max_calculation',
            'heat_stress_avg_calculation', 
            'laying_loss_calculation',
            'milk_production_loss_calculation',
            'daily_weight_gain_loss_calculation',
            'map_creation',
            'debounced_map_update',
            'indicator_update',
            'indicator_map_creation'
        ]
        
        print("=== PERFORMANCE PROFILING REPORT ===")
        for profile_name in profiles:
            try:
                result = pn.state.get_profile(profile_name)
                if result:
                    print(f"\n{profile_name.upper()}:")
                    print(f"  Calls: {len(result) if hasattr(result, '__len__') else 'N/A'}")
                else:
                    print(f"\n{profile_name.upper()}: No data yet")
            except Exception as e:
                print(f"\n{profile_name.upper()}: Error - {e}")
        print("=" * 40)
    
    # Make profiling function available globally
    app.get_performance_report = get_performance_report
    
    print("🔍 DEBUG PROFILING ENABLED")
    print("Access performance data with:")
    print("  - Individual: pn.state.get_profile('heat_stress_max_calculation')")
    print("  - All reports: app.get_performance_report()")
    print("  - Admin panel: /admin (when running panel serve)")

# Cache management utilities
def clear_all_cache():
    """Vide tous les caches pour forcer la régénération"""
    app.data_manager.clear_cache()
    # Supprimer aussi les données de base
    for filename in ["temperature_data", "humidity_data", "coordinates_data", "precomputed_scenarios"]:
        filepath = app.data_manager.data_dir / f"{filename}.parquet"
        if filepath.exists():
            filepath.unlink()
    print("🗑️ Tous les caches ont été vidés")

def cache_info():
    """Affiche des informations sur le cache"""
    cache_files = list(app.data_manager.cache_dir.glob("*.parquet"))
    data_files = list(app.data_manager.data_dir.glob("*.parquet"))
    
    print(f"📊 CACHE INFO:")
    print(f"  Cache files: {len(cache_files)}")
    print(f"  Data files: {len(data_files)}")
    
    total_size = sum(f.stat().st_size for f in cache_files + data_files)
    print(f"  Total size: {total_size / 1024 / 1024:.2f} MB")

# Make utilities available
app.clear_all_cache = clear_all_cache
app.cache_info = cache_info

print("💾 PARQUET CACHE SYSTEM ENABLED")
print("Cache utilities:")
print("  - app.clear_all_cache() - Vider tous les caches")
print("  - app.cache_info() - Informations sur le cache")

app.layout.servable()
