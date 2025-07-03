"""
Application Panel refactorisée pour visualiser des indicateurs agroclimatiques
Architecture fonctionnelle avec séparation claire UI/métier
"""

import sys
import warnings
from pathlib import Path

# Ajouter le répertoire src au path pour les imports
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geoviews as gv
import holoviews as hv
import numpy as np
import panel as pn
import xarray as xr
from holoviews import opts

# Imports relatifs depuis src
try:
    from core.data_access import load_or_generate_weather_data
    from core.indicators import (get_indicator_function, get_stress_colors,
                                 get_stress_labels)
    from core.models import (ANIMAL_TYPES, INDICATORS, WEATHER_MODELS,
                             AnimalCategory, FilterState, MapData,
                             WeatherModel)
except ImportError:
    # Fallback si les imports relatifs ne marchent pas
    import importlib.util

    # Charger les modules manuellement
    models_spec = importlib.util.spec_from_file_location("models", src_dir / "core" / "models.py")
    models_module = importlib.util.module_from_spec(models_spec)
    models_spec.loader.exec_module(models_module)
    
    indicators_spec = importlib.util.spec_from_file_location("indicators", src_dir / "core" / "indicators.py")
    indicators_module = importlib.util.module_from_spec(indicators_spec)
    indicators_spec.loader.exec_module(indicators_module)
    
    data_access_spec = importlib.util.spec_from_file_location("data_access", src_dir / "core" / "data_access.py")
    data_access_module = importlib.util.module_from_spec(data_access_spec)
    data_access_spec.loader.exec_module(data_access_module)
    
    # Importer les objets nécessaires
    from data_access import load_or_generate_weather_data
    from indicators import (get_indicator_function, get_stress_colors,
                            get_stress_labels)
    from models import (ANIMAL_TYPES, INDICATORS, WEATHER_MODELS,
                        AnimalCategory, FilterState, MapData, WeatherModel)

warnings.filterwarnings('ignore')

# Configuration Panel
pn.extension('tabulator')
hv.extension('bokeh')
gv.extension('bokeh')


def create_category_filter(state: FilterState, update_callback) -> pn.Card:
    """Crée le filtre de catégorie d'animaux"""
    
    # Options disponibles
    category_options = [
        ("🐄 BOVINS", AnimalCategory.BOVINS),
        ("🐔 VOLAILLES", AnimalCategory.VOLAILLES)
    ]
    
    category_select = pn.widgets.RadioButtonGroup(
        options=category_options,
        value=state.category,
        button_type='primary',
        orientation='vertical',
        button_style='outline',
        width=250
    )
    
    def on_category_change(event):
        state.category = event.new
        state.indicator_id = None  # Reset dependent filters
        state.animal_type_id = None
        state.race = None
        update_callback()
    
    category_select.param.watch(on_category_change, 'value')
    
    return pn.Card(
        category_select,
        title="1 - Choisir une catégorie",
        width=300,
        margin=(10, 10),
        styles={'background': '#f8f9fa'}
    )


def create_indicator_filter(state: FilterState, update_callback) -> pn.Card:
    """Crée le filtre d'indicateur"""
    
    if not state.category:
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
        if v.category == state.category
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
        value=state.indicator_id,
        button_type='primary',
        orientation='vertical',
        button_style='outline',
        width=250
    )
    
    def on_indicator_change(event):
        state.indicator_id = event.new
        update_callback()
    
    indicator_select.param.watch(on_indicator_change, 'value')
    
    return pn.Card(
        indicator_select,
        title="2 - Choisir un indicateur",
        width=300,
        margin=(10, 10),
        styles={'background': '#f8f9fa'}
    )


def create_animal_parameters(state: FilterState, update_callback) -> pn.Column:
    """Crée les paramètres d'animal"""
    
    if not state.category:
        return pn.pane.HTML("<p><i>Sélectionnez d'abord une catégorie</i></p>")
    
    # Filtrer les types d'animaux par catégorie
    available_animals = {
        k: v for k, v in ANIMAL_TYPES.items()
        if v.category == state.category
    }
    
    if not available_animals:
        return pn.pane.HTML("<p><i>Aucun type d'animal disponible</i></p>")
    
    # Sélecteur de type d'animal
    animal_options = [(animal.name, animal_id) 
                     for animal_id, animal in available_animals.items()]
    
    animal_select = pn.widgets.Select(
        name="Type d'animal",
        options=animal_options,
        value=state.animal_type_id,
        width=280
    )
    
    # Sélecteur de race
    race_select = pn.widgets.Select(
        name="Race",
        options=[],
        width=280
    )
    
    def update_races():
        """Met à jour les races disponibles"""
        if state.animal_type_id and state.animal_type_id in available_animals:
            animal = available_animals[state.animal_type_id]
            race_options = [(race, race) for race in animal.races]
            race_select.options = race_options
            if race_options and not state.race:
                state.race = race_options[0][1]
                race_select.value = state.race
    
    def on_animal_change(event):
        state.animal_type_id = event.new
        update_races()
        update_callback()
    
    def on_race_change(event):
        state.race = event.new
        update_callback()
    
    animal_select.param.watch(on_animal_change, 'value')
    race_select.param.watch(on_race_change, 'value')
    
    # Initialiser les races
    update_races()
    
    return pn.Column(
        "## 3 - Paramètres",
        animal_select,
        race_select,
        width=300,
        sizing_mode='fixed'
    )


def create_weather_model_filter(state: FilterState, update_callback) -> pn.Column:
    """Crée le filtre de modèle météorologique"""
    
    models_info = WEATHER_MODELS
    
    # Créer des boutons pour chaque modèle
    model_buttons = []
    for model, info in models_info.items():
        button = pn.widgets.Button(
            name=f"{info.name}\n{info.description}\n{info.resolution}",
            button_type='primary' if state.weather_model == model else 'default',
            width=90,
            height=80
        )
        
        def make_callback(m):
            def callback(event):
                state.weather_model = m
                # Mettre à jour l'apparence des boutons
                for i, (model_key, _) in enumerate(models_info.items()):
                    model_buttons[i].button_type = 'primary' if model_key == m else 'default'
                update_callback()
            return callback
        
        button.on_click(make_callback(model))
        model_buttons.append(button)
    
    # Seuil de température
    temp_slider = pn.widgets.FloatSlider(
        name="Seuil de température (°C)",
        start=15.0,
        end=40.0,
        step=0.5,
        value=state.temperature_threshold,
        width=280
    )
    
    def on_temp_change(event):
        state.temperature_threshold = event.new
        update_callback()
    
    temp_slider.param.watch(on_temp_change, 'value')
    
    return pn.Column(
        "## 4 - Modèle météorologique",
        pn.Row(*model_buttons),
        temp_slider,
        width=300,
        sizing_mode='fixed'
    )


def create_map_visualization(state: FilterState, lons, lats, temp_data, humidity_data) -> pn.pane.HoloViews:
    """Crée la visualisation de la carte"""
    
    if not state.indicator_id or not state.animal_type_id:
        return pn.pane.HTML(
            "<div style='text-align: center; padding: 50px;'>"
            "<h3>Sélectionnez un indicateur et un type d'animal pour afficher la carte</h3>"
            "</div>",
            width=800,
            height=600
        )
    
    # Obtenir la fonction de calcul
    indicator = INDICATORS[state.indicator_id]
    calc_function = get_indicator_function(indicator.calculation_function)
    
    if not calc_function:
        return pn.pane.HTML("<p>Fonction de calcul non trouvée</p>")
    
    # Calculer l'indicateur
    try:
        if indicator.calculation_function in ['calculate_laying_loss', 'calculate_daily_weight_gain_loss']:
            indicator_data = calc_function(temp_data, humidity_data, state.animal_type_id)
        elif indicator.calculation_function in ['calculate_heat_stress_max', 'calculate_heat_stress_avg']:
            indicator_data = calc_function(temp_data, state.animal_type_id, state.temperature_threshold)
        else:
            indicator_data = calc_function(temp_data, state.animal_type_id)
    except Exception as e:
        return pn.pane.HTML(f"<p>Erreur de calcul: {e}</p>")
    
    # Créer la carte
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
            title=f"{indicator.name} - {ANIMAL_TYPES[state.animal_type_id].name}",
            tools=['hover'],
            projection=ccrs.PlateCarree()
        )
    )
    
    # Ajouter les contours géographiques
    coastline = gv.feature.coastline().opts(line_color='black', line_width=1)
    borders = gv.feature.borders().opts(line_color='gray', line_width=0.5)
    
    final_map = map_viz * coastline * borders
    
    return pn.pane.HoloViews(final_map, sizing_mode='stretch_width')


def create_info_panel() -> pn.pane.Markdown:
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


def create_app():
    """Crée l'application principale"""
    
    # État initial
    state = FilterState()
    
    # Charger les données météorologiques
    lons, lats, temp_data, humidity_data = load_or_generate_weather_data()
    
    # Créer les composants
    category_filter = create_category_filter(state, lambda: update_ui())
    indicator_filter = create_indicator_filter(state, lambda: update_ui())
    animal_params = create_animal_parameters(state, lambda: update_ui())
    weather_filter = create_weather_model_filter(state, lambda: update_ui())
    info_panel = create_info_panel()
    map_viz = create_map_visualization(state, lons, lats, temp_data, humidity_data)
    
    # Panneau latéral (filtres)
    sidebar = pn.Column(
        category_filter,
        indicator_filter,
        animal_params,
        weather_filter,
        info_panel,
        width=320,
        sizing_mode='fixed',
        scroll=True
    )
    
    # Zone principale (carte)
    main_area = pn.Column(
        map_viz,
        sizing_mode='stretch_width'
    )
    
    def update_ui():
        """Met à jour l'interface utilisateur"""
        # Recréer les composants dépendants
        sidebar[1] = create_indicator_filter(state, lambda: update_ui())
        sidebar[2] = create_animal_parameters(state, lambda: update_ui())
        main_area[0] = create_map_visualization(state, lons, lats, temp_data, humidity_data)
    
    # Layout principal
    template = pn.template.MaterialTemplate(
        title="🌡️ Indicateurs Agroclimatiques - France",
        sidebar=[sidebar],
        main=[main_area],
        header_background='#2596be',
        sidebar_width=350
    )
    
    return template


# Créer l'application pour panel serve
app = create_app()

if __name__ == "__main__":
    # Pour lancement direct
    pn.serve(app, port=5008, show=True, autoreload=True, 
             title="Indicateurs Agroclimatiques")
