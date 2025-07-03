"""
Application Panel refactorisée avec système de state management réactif
Architecture basée sur param.Parameterized pour une gestion d'état robuste
"""

import os
import sys
import warnings
from pathlib import Path

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.insert(0, src_dir)

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geoviews as gv
import holoviews as hv
import numpy as np
import panel as pn
import param
import xarray as xr
from holoviews import opts

from core.data_access import load_or_generate_weather_data
from core.indicators import (get_indicator_function, get_stress_colors,
                             get_stress_labels)
from core.models import (ANIMAL_TYPES, INDICATORS, WEATHER_MODELS,
                         AnimalCategory, FilterState, MapData, WeatherModel)

warnings.filterwarnings('ignore')

# Configuration Panel
pn.extension('tabulator')
hv.extension('bokeh')
gv.extension('bokeh')


class ReactiveAgroclimaticApp(param.Parameterized):
    """Application principale avec gestion d'état réactive"""
    
    # État réactif
    category = param.Parameter(default=AnimalCategory.BOVINS)
    indicator_id = param.Parameter(default="")
    animal_type_id = param.Parameter(default="")
    race = param.Parameter(default="")
    weather_model = param.Parameter(default="AROME")
    temperature_threshold = param.Number(default=30.0, bounds=(15.0, 40.0))
    
    def __init__(self, **params):
        super().__init__(**params)
        
        # Charger les données météorologiques
        self.lons, self.lats, self.temp_data, self.humidity_data = load_or_generate_weather_data()
        
        # Initialiser l'état avec des valeurs par défaut
        self._initialize_default_state()
        
        # Créer l'interface utilisateur
        self._create_ui()
    
    def _initialize_default_state(self):
        """Initialise l'état avec des valeurs par défaut cohérentes"""
        # Auto-sélectionner le premier indicateur pour la catégorie par défaut
        available_indicators = {
            k: v for k, v in INDICATORS.items() 
            if v.category == self.category
        }
        if available_indicators and not self.indicator_id:
            self.indicator_id = list(available_indicators.keys())[0]
        
        # Auto-sélectionner le premier type d'animal pour la catégorie par défaut
        available_animals = {
            k: v for k, v in ANIMAL_TYPES.items()
            if v.category == self.category
        }
        if available_animals and not self.animal_type_id:
            self.animal_type_id = list(available_animals.keys())[0]
            
        # Auto-sélectionner la première race
        if self.animal_type_id and not self.race:
            animal = ANIMAL_TYPES[self.animal_type_id]
            if animal.races:
                self.race = animal.races[0]
    
    @param.depends('category', watch=True)
    def _on_category_change(self):
        """Réagit aux changements de catégorie"""
        # Auto-sélectionner le premier indicateur disponible
        available_indicators = {
            k: v for k, v in INDICATORS.items() 
            if v.category == self.category
        }
        if available_indicators:
            self.indicator_id = list(available_indicators.keys())[0]
        else:
            self.indicator_id = ""
        
        # Auto-sélectionner le premier type d'animal disponible
        available_animals = {
            k: v for k, v in ANIMAL_TYPES.items()
            if v.category == self.category
        }
        if available_animals:
            self.animal_type_id = list(available_animals.keys())[0]
        else:
            self.animal_type_id = ""
        
        # Réinitialiser la race
        self.race = ""
        
        # Mettre à jour l'interface
        self._update_ui_components()
    
    @param.depends('animal_type_id', watch=True)
    def _on_animal_type_change(self):
        """Réagit aux changements de type d'animal"""
        if self.animal_type_id and self.animal_type_id in ANIMAL_TYPES:
            animal = ANIMAL_TYPES[self.animal_type_id]
            if animal.races:
                self.race = animal.races[0]
            else:
                self.race = ""
        else:
            self.race = ""
    
    def _create_ui(self):
        """Crée l'interface utilisateur"""
        # Créer les composants
        self.category_filter = self._create_category_filter()
        self.indicator_filter = self._create_indicator_filter()
        self.animal_params = self._create_animal_parameters()
        self.weather_filter = self._create_weather_model_filter()
        self.info_panel = self._create_info_panel()
        self.map_viz = self._create_map_visualization()
        
        # Panneau latéral (filtres)
        self.sidebar = pn.Column(
            self.category_filter,
            self.indicator_filter,
            self.animal_params,
            self.weather_filter,
            self.info_panel,
            width=320,
            sizing_mode='fixed',
            scroll=True
        )
        
        # Zone principale (carte)
        self.main_area = pn.Column(
            self.map_viz,
            sizing_mode='stretch_width'
        )
        
        # Layout principal
        self.layout = pn.template.MaterialTemplate(
            title="🌡️ Indicateurs Agroclimatiques - France",
            sidebar=[self.sidebar],
            main=[self.main_area],
            header_background='#2596be',
            sidebar_width=350
        )
    
    def _create_category_filter(self) -> pn.Card:
        """Crée le filtre de catégorie d'animaux"""
        category_options = [
            ("🐄 BOVINS", AnimalCategory.BOVINS),
            ("🐔 VOLAILLES", AnimalCategory.VOLAILLES)
        ]
        
        category_select = pn.widgets.RadioButtonGroup(
            options=category_options,
            value=self.category,
            button_type='primary',
            orientation='vertical',
            button_style='outline',
            width=250
        )
        
        def on_category_change(event):
            self.category = event.new
        
        category_select.param.watch(on_category_change, 'value')
        
        return pn.Card(
            category_select,
            title="1 - Choisir une catégorie",
            width=300,
            margin=(10, 10),
            styles={'background': '#f8f9fa'}
        )
    
    @param.depends('category', 'indicator_id')
    def _create_indicator_filter(self) -> pn.Card:
        """Crée le filtre d'indicateur (réactif)"""
        if not self.category:
            return pn.Card(
                pn.pane.HTML(
                    "<div style='text-align: center; color: #6c757d; padding: 20px;'>"
                    "<i class='fas fa-arrow-up'></i><br>"
                    "Sélectionnez d'abord une catégorie"
                    "</div>"
                ),
                title="2 - Choisir un indicateur",
                width=300,
                margin=(10, 10),
                styles={'background': '#f8f9fa', 'opacity': '0.6'}
            )
        
        # Filtrer les indicateurs par catégorie
        available_indicators = {
            k: v for k, v in INDICATORS.items() 
            if v.category == self.category
        }
        
        if not available_indicators:
            return pn.Card(
                pn.pane.HTML("<p><i>Aucun indicateur disponible</i></p>"),
                title="2 - Choisir un indicateur",
                width=300,
                margin=(10, 10),
                styles={'background': '#f8f9fa'}
            )
        
        options = [(indicator.name, indicator_id) 
                   for indicator_id, indicator in available_indicators.items()]
        
        indicator_select = pn.widgets.RadioButtonGroup(
            options=options,
            value=self.indicator_id if self.indicator_id in available_indicators else None,
            button_type='primary',
            orientation='vertical',
            button_style='outline',
            width=250
        )
        
        def on_indicator_change(event):
            if event.new:
                self.indicator_id = event.new
        
        indicator_select.param.watch(on_indicator_change, 'value')
        
        return pn.Card(
            indicator_select,
            title="2 - Choisir un indicateur",
            width=300,
            margin=(10, 10),
            styles={'background': '#f8f9fa'}
        )
    
    @param.depends('category', 'animal_type_id', 'race')
    def _create_animal_parameters(self) -> pn.Column:
        """Crée les paramètres d'animal (réactif)"""
        if not self.category:
            return pn.Column(
                "## 3 - Paramètres",
                pn.pane.HTML("<p><i>Sélectionnez d'abord une catégorie</i></p>"),
                width=300,
                sizing_mode='fixed'
            )
        
        # Filtrer les types d'animaux par catégorie
        available_animals = {
            k: v for k, v in ANIMAL_TYPES.items()
            if v.category == self.category
        }
        
        if not available_animals:
            return pn.Column(
                "## 3 - Paramètres",
                pn.pane.HTML("<p><i>Aucun type d'animal disponible</i></p>"),
                width=300,
                sizing_mode='fixed'
            )
        
        # Sélecteur de type d'animal
        animal_options = [(animal.name, animal_id) 
                         for animal_id, animal in available_animals.items()]
        
        animal_select = pn.widgets.Select(
            name="Type d'animal",
            options=animal_options,
            value=self.animal_type_id if self.animal_type_id in available_animals else None,
            width=280
        )
        
        # Sélecteur de race
        race_options = []
        if self.animal_type_id and self.animal_type_id in available_animals:
            animal = available_animals[self.animal_type_id]
            race_options = [(race, race) for race in animal.races]
        
        race_select = pn.widgets.Select(
            name="Race",
            options=race_options,
            value=self.race if self.race in [r[1] for r in race_options] else None,
            width=280
        )
        
        def on_animal_change(event):
            if event.new:
                self.animal_type_id = event.new
        
        def on_race_change(event):
            if event.new:
                self.race = event.new
        
        animal_select.param.watch(on_animal_change, 'value')
        race_select.param.watch(on_race_change, 'value')
        
        return pn.Column(
            "## 3 - Paramètres",
            animal_select,
            race_select,
            width=300,
            sizing_mode='fixed'
        )
    
    def _create_weather_model_filter(self) -> pn.Column:
        """Crée le filtre de modèle météorologique"""
        models_info = WEATHER_MODELS
        
        # Créer des boutons pour chaque modèle
        model_buttons = []
        for model, info in models_info.items():
            button = pn.widgets.Button(
                name=f"{info.name}\n{info.description}\n{info.resolution}",
                button_type='primary' if self.weather_model == model else 'default',
                width=90,
                height=80
            )
            
            def make_callback(m):
                def callback(event):
                    self.weather_model = m
                    # Mettre à jour l'apparence des boutons
                    for i, (model_key, _) in enumerate(models_info.items()):
                        model_buttons[i].button_type = 'primary' if model_key == m else 'default'
                return callback
            
            button.on_click(make_callback(model))
            model_buttons.append(button)
        
        # Seuil de température
        temp_slider = pn.widgets.FloatSlider(
            name="Seuil de température (°C)",
            start=15.0,
            end=40.0,
            step=0.5,
            value=self.temperature_threshold,
            width=280
        )
        
        def on_temp_change(event):
            self.temperature_threshold = event.new
        
        temp_slider.param.watch(on_temp_change, 'value')
        
        return pn.Column(
            "## 4 - Modèle météorologique",
            pn.Row(*model_buttons),
            temp_slider,
            width=300,
            sizing_mode='fixed'
        )
    
    @param.depends('indicator_id', 'animal_type_id', 'temperature_threshold')
    def _create_map_visualization(self):
        """Crée la visualisation de la carte (réactive)"""
        if not self.indicator_id or not self.animal_type_id:
            return pn.pane.HTML(
                "<div style='text-align: center; padding: 50px;'>"
                "<h3>Sélectionnez un indicateur et un type d'animal pour afficher la carte</h3>"
                "</div>",
                width=800,
                height=600
            )
        
        # Obtenir la fonction de calcul
        indicator = INDICATORS[self.indicator_id]
        calc_function = get_indicator_function(indicator.calculation_function)
        
        if not calc_function:
            return pn.pane.HTML("<p>Fonction de calcul non trouvée</p>")
        
        # Calculer l'indicateur
        try:
            if indicator.calculation_function == 'calculate_laying_loss':
                indicator_data = calc_function(self.temp_data, self.humidity_data)
            elif indicator.calculation_function == 'calculate_daily_weight_gain_loss':
                indicator_data = calc_function(self.temp_data, self.humidity_data)
            elif indicator.calculation_function == 'calculate_heat_stress_max':
                indicator_data = calc_function(self.temp_data, self.temperature_threshold)
            elif indicator.calculation_function == 'calculate_heat_stress_avg':
                indicator_data = calc_function(self.temp_data, self.temperature_threshold)
            elif indicator.calculation_function == 'calculate_milk_production_loss':
                indicator_data = calc_function(self.temp_data)
            else:
                indicator_data = calc_function(self.temp_data)
        except Exception as e:
            return pn.pane.HTML(f"<p>Erreur de calcul: {e}</p>")
        
        # Créer la carte
        lon_grid, lat_grid = np.meshgrid(self.lons, self.lats)
        
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
        
        # Configuration des couleurs
        stress_colors = get_stress_colors()
        color_map = [stress_colors[i] for i in range(5)]
        
        map_viz = map_data.opts(
            opts.QuadMesh(
                cmap=color_map,
                clim=(0, 4),
                colorbar=True,
                colorbar_opts={'title': 'Niveau de stress'},
                width=800,
                height=600,
                title=f"{indicator.name} - {ANIMAL_TYPES[self.animal_type_id].name}",
                tools=['hover'],
                projection=ccrs.PlateCarree()
            )
        )
        
        # Ajouter les contours géographiques
        coastline = gv.feature.coastline().opts(line_color='black', line_width=1)
        borders = gv.feature.borders().opts(line_color='gray', line_width=0.5)
        
        final_map = map_viz * coastline * borders
        
        return pn.pane.HoloViews(final_map, sizing_mode='stretch_width')
    
    def _create_info_panel(self) -> pn.pane.Markdown:
        """Crée le panneau d'informations"""
        return pn.pane.Markdown("""
        ## Indicateurs Agroclimatiques
        
        Cette application permet de visualiser différents indicateurs de stress 
        climatique pour l'élevage:
        
        ### Bovins
        - **Stress thermique maximal/moyen**: Impact de la température
        - **Perte de production laitière**: Réduction du lait (vaches laitières)
        - **Perte de GMQ**: Réduction du gain de masse (engraissement)
        
        ### Volailles  
        - **Perte de ponte**: Réduction de la production d'œufs
        
        ### Niveaux de stress
        - 🟢 **Aucun stress** (0)
        - 🟡 **Faible** (1) 
        - 🟠 **Modéré** (2)
        - 🔴 **Fort** (3)
        - 🔴 **Très sévère** (4)
        
        Survolez la carte pour plus d'informations.
        """, width=300)
    
    def _update_ui_components(self):
        """Met à jour les composants de l'interface utilisateur"""
        # Les composants réactifs se mettront à jour automatiquement
        # grâce aux décorateurs @param.depends
        pass
    
    def serve(self, port=5008, show=True):
        """Lance l'application"""
        return pn.serve(self.layout, port=port, show=show, autoreload=True,
                       title="Indicateurs Agroclimatiques")


# Créer et lancer l'application
app = ReactiveAgroclimaticApp()

# Pour lancement direct
if __name__ == "__main__":
    app.serve()
else:
    # Pour panel serve
    pn.serve(app.layout, port=5008, show=True, autoreload=True, 
             title="Indicateurs Agroclimatiques")
