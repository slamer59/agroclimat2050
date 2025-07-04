"""
Application Panel pour visualiser des indicateurs agroclimatiques
Architecture modulaire avec séparation UI/calculs et cache efficace
VERSION HVPLOT
"""

import datetime
import threading
import time
import warnings
from functools import lru_cache
from pathlib import Path

import hvplot.pandas  # Import hvplot for pandas
import numpy as np
import pandas as pd
import panel as pn
import param
from panel.io.profile import profile

# Debug mode flag - Set to False for production
DEBUG_PROFILING = True  # Change to False to disable profiling

warnings.filterwarnings('ignore')

# Configuration Panel
pn.extension('tabulator')

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
        stress_classes = np.zeros(stress_values.shape, dtype=np.int8)
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
        stress_classes = np.zeros(combined_stress.shape, dtype=np.int8)
        stress_classes[combined_stress > 0] = 1
        stress_classes[combined_stress > 8] = 2
        stress_classes[combined_stress > 20] = 3
        stress_classes[combined_stress > 35] = 4
        
        self.data_manager.set_cache(cache_key, stress_classes)
        return stress_classes

class HvPlotVisualizer:
    """Visualisateur de cartes pour les indicateurs utilisant hvPlot"""
    
    def __init__(self):
        # Couleurs selon l'image de légende
        self.stress_colors = ['#00ff00', '#ffff00', '#ffa500', '#ff4500', '#8b0000']
        
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
        """Crée une carte d'indicateur avec hvPlot"""
        # Créer les grilles de coordonnées
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # S'assurer que indicator_data a la bonne forme
        if indicator_data.shape != lon_grid.shape:
            indicator_data = np.resize(indicator_data, lon_grid.shape)
        
        # Générer des valeurs réelles pour le hover basées sur les classes
        real_values = np.zeros(indicator_data.shape, dtype=float)
        for i in range(5):
            mask = indicator_data == i
            if np.sum(mask) > 0:
                min_val, max_val = self.stress_ranges[i]
                real_values[mask] = np.random.uniform(min_val, max_val, np.sum(mask))
        
        # Créer un DataFrame pour hvPlot (plus stable que xarray)
        df_data = []
        for i in range(len(lats)):
            for j in range(len(lons)):
                df_data.append({
                    'lon': lons[j],
                    'lat': lats[i],
                    'stress_level': int(indicator_data[i, j]),
                    'stress_value': float(real_values[i, j]),
                    'label': self.stress_labels[int(indicator_data[i, j])]
                })
        
        df = pd.DataFrame(df_data)
        
        # Créer la carte avec hvPlot - version scatter (plus stable)
        plot = df.hvplot.scatter(
            x='lon', 
            y='lat', 
            c='stress_level',
            cmap=self.stress_colors,
            clim=(0, 4),
            title=title,
            width=800,
            height=600,
            colorbar=True,
            colorbar_opts={'title': 'Niveau de stress'},
            hover_cols=['stress_value', 'label'],
            size=40,
            alpha=0.8
        )
        
        return plot

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

    def __init__(self, **params):
        super().__init__(**params)
        
        # Initialisation des composants
        self.data_manager = DataManager()
        self.calculator = IndicatorCalculator(self.data_manager)
        self.visualizer = HvPlotVisualizer()
        
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
        
        print("✅ Précalcul terminé - réponses instantanées disponibles")
    
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
        
        # Zone principale pour la carte - utilise maintenant hvPlot
        self.map_pane = pn.pane.HoloViews(
            self._create_map(),
            sizing_mode='stretch_width',
            height=700
        )
        
        # Layout principal avec style amélioré
        self.layout = pn.template.MaterialTemplate(
            title="AGRO CLIMAT - Indicateurs Agroclimatiques (hvPlot)",
            sidebar=[sidebar],
            main=[self.map_pane],
            header_background='#2596be',
            sidebar_width=340
        )
    
    @param.depends('selected_category', watch=True)
    @profile('indicator_update', engine='pyinstrument') if DEBUG_PROFILING else lambda f: f
    def _update_indicators(self):
        """Met à jour les indicateurs disponibles selon la catégorie sélectionnée"""
        # Mettre à jour la liste des indicateurs disponibles
        available_indicators = self.indicators_by_category.get(self.selected_category, [])
        self.param.selected_indicator.objects = available_indicators
        
        # Sélectionner le premier indicateur par défaut
        if available_indicators:
            self.selected_indicator = available_indicators[0]
        
        # Afficher l'étape 2
        self.step2_container.visible = True
        self.step3_container.visible = True
        self.step4_container.visible = True
        
        # Mettre à jour la carte
        self._schedule_map_update()
    
    @param.depends('selected_indicator', 'temperature_threshold', 'weather_model', watch=True)
    def _schedule_map_update(self):
        """Programme une mise à jour de la carte avec debouncing"""
        if self._update_timer is not None:
            self._update_timer.cancel()
        
        self._update_timer = threading.Timer(0.3, self._update_map)
        self._update_timer.start()
    
    @profile('map_update', engine='pyinstrument') if DEBUG_PROFILING else lambda f: f
    def _update_map(self):
        """Met à jour la carte avec les nouveaux paramètres"""
        try:
            new_plot = self._create_map()
            self.map_pane.object = new_plot
        except Exception as e:
            print(f"⚠️ Erreur lors de la mise à jour de la carte: {e}")
    
    def _create_animal_params_panel(self):
        """Crée le panneau de paramètres pour les animaux"""
        # Paramètres de base pour tous les animaux
        base_params = pn.Param(
            self.animal_params,
            parameters=['animal_type', 'simulation_mode', 'temperature_offset'],
            show_name=False,
            width=280
        )
        
        # Paramètres spécifiques selon le type d'animal
        specific_params = pn.pane.Markdown("Sélectionnez un type d'animal pour voir les paramètres spécifiques.")
        
        return pn.Column(base_params, specific_params)
    
    @profile('map_creation', engine='pyinstrument') if DEBUG_PROFILING
