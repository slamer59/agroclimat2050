# -*- coding: utf-8 -*-
"""
Application Panel pour visualiser des indicateurs agroclimatiques.
Architecture modernisée avec xarray, hvplot et cache NetCDF,
tout en conservant l'interface utilisateur multi-étapes et tous les indicateurs.
"""
import os
import warnings

import cartopy.crs as ccrs
import hvplot.xarray
import numpy as np
import panel as pn
import param
import xarray as xr
import xyzservices.providers as xyz

# --- Configuration ---
warnings.filterwarnings('ignore')
pn.extension()

# --- Fonctions de calcul des indicateurs ---

class IndicatorCalculator:
    """
    Calcule les indicateurs agroclimatiques à partir de données xarray.
    Les calculs sont vectorisés et retournent des DataArrays.
    """
    def _calculate_stress(self, stress_factor):
        """Helper function to classify stress levels."""
        return (
            (stress_factor > 0).astype(np.float32) +
            (stress_factor > 10).astype(np.float32) +
            (stress_factor > 25).astype(np.float32) +
            (stress_factor > 50).astype(np.float32)
        )

    def calculate_heat_stress_max(self, temp_data, threshold=30):
        """Calcule le stress thermique maximal."""
        stress_factor = (temp_data - threshold) / threshold * 100
        stress_classes = self._calculate_stress(stress_factor)
        stress_classes.attrs['long_name'] = 'Stress Thermique Maximal'
        return stress_classes

    def calculate_laying_loss(self, temp_data, humidity_data):
        """Calcule la perte de ponte."""
        total_stress = (np.abs(temp_data - 20) / 20 + np.abs(humidity_data - 60) / 60) * 50
        stress_classes = self._calculate_stress(total_stress)
        stress_classes.attrs['long_name'] = 'Perte de Ponte (%)'
        return stress_classes

    def calculate_milk_production_loss(self, temp_data):
        """Calcule la perte de production de lait."""
        optimal_temp = 18
        stress_values = xr.where(temp_data > optimal_temp,
                                 (temp_data - optimal_temp) / optimal_temp * 60, 0)
        stress_classes = self._calculate_stress(stress_values)
        stress_classes.attrs['long_name'] = 'Perte de Production de Lait (%)'
        return stress_classes

    def calculate_daily_weight_gain_loss(self, temp_data, humidity_data):
        """Calcule la perte de GMQ (Gain de Masse Quotidien)."""
        optimal_temp = 16
        optimal_humidity = 65
        temp_factor = xr.where(temp_data > optimal_temp, (temp_data - optimal_temp) / optimal_temp, 0)
        humidity_factor = xr.where(humidity_data > optimal_humidity, (humidity_data - optimal_humidity) / optimal_humidity, 0)
        combined_stress = (temp_factor + humidity_factor) * 40
        stress_classes = self._calculate_stress(combined_stress)
        stress_classes.attrs['long_name'] = 'Perte de GMQ (%)'
        return stress_classes

# --- Application Principale ---

class AgroclimaticApp(param.Parameterized):
    """Application principale avec UI multi-étapes et backend xarray."""

    # --- Étape 1: Sélection de catégorie ---
    selected_category = param.Selector(
        default="ANIMAUX",
        objects=["ANIMAUX", "FEUX DE FORÊT", "MALADIES", "PRATIQUES AGRICOLES", "RAVAGEURS", "POLLENS", "VÉGÉTAUX"],
        doc="Catégorie d'activité agricole"
    )

    # --- Étape 2: Sélection d'indicateur (KPI) ---
    selected_indicator = param.Selector(
        default="STRESS THERMIQUE MAXIMAL",
        objects=[], # Sera peuplé dynamiquement
        doc="Indicateur à afficher"
    )

    # --- Étape 3: Paramètres ---
    temperature_threshold = param.Number(
        default=30.0, bounds=(15.0, 40.0), step=0.5,
        doc="Seuil de température (°C) pour le stress thermique"
    )

    # --- État interne ---
    ds = param.Parameter(doc="Le dataset xarray contenant les données et les KPIs.")
    calculator = param.ClassSelector(IndicatorCalculator, default=IndicatorCalculator())
    DATA_FILE = "agro_data.nc"

    # --- Dictionnaires de mapping ---
    INDICATORS_BY_CATEGORY = {
        "ANIMAUX": [
            "STRESS THERMIQUE MAXIMAL",
            "PERTE DE PONTE (%)",
            "PERTE DE PRODUCTION DE LAIT (%)",
            "PERTE DE GMQ - GAIN EN MASSE QUOTIDIEN (%)"
        ],
        "FEUX DE FORÊT": ["RISQUE D'INCENDIE"],
        "MALADIES": ["PROPAGATION PATHOGÈNES"],
        # Ajouter d'autres KPIs ici
    }

    KPI_VAR_MAP = {
        "STRESS THERMIQUE MAXIMAL": "stress_thermique_max",
        "PERTE DE PONTE (%)": "perte_ponte",
        "PERTE DE PRODUCTION DE LAIT (%)": "perte_lait",
        "PERTE DE GMQ - GAIN EN MASSE QUOTIDIEN (%)": "perte_gmq",
        "RISQUE D'INCENDIE": "risque_incendie", # Exemple
        "PROPAGATION PATHOGÈNES": "propagation_pathogenes" # Exemple
    }

    def __init__(self, **params):
        super().__init__(**params)
        self.ds = self._load_or_create_dataset()
        self._update_indicator_options() # Initial call
        self.param.watch(self._update_stress_kpi, 'temperature_threshold')

    def _load_or_create_dataset(self):
        """Charge ou crée le dataset NetCDF."""
        if os.path.exists(self.DATA_FILE):
            return xr.open_dataset(self.DATA_FILE)
        
        print("🛠️ Génération des données d'exemple (une seule fois)...")
        lons = np.linspace(-5.5, 9.6, 100)
        lats = np.linspace(41.0, 51.2, 80)
        ds = xr.Dataset(coords={'lon': lons, 'lat': lats})
        
        np.random.seed(42)
        ds['Tair'] = (('lat', 'lon'), 15 + 15 * np.random.rand(len(lats), len(lons)))
        ds['Tair'].attrs = {'long_name': 'Température de l\'air (°C)', 'units': 'celsius'}
        
        ds['humidity'] = (('lat', 'lon'), 40 + 40 * np.random.rand(len(lats), len(lons)))
        ds['humidity'].attrs = {'long_name': 'Humidité relative (%)', 'units': '%'}
        
        ds.to_netcdf(self.DATA_FILE)
        return ds

    @param.depends('selected_category', watch=True)
    def _update_indicator_options(self):
        """Met à jour la liste des indicateurs en fonction de la catégorie."""
        indicators = self.INDICATORS_BY_CATEGORY.get(self.selected_category, [])
        self.param.selected_indicator.objects = indicators
        if indicators:
            self.selected_indicator = indicators[0]

    def _ensure_kpi_is_calculated(self):
        """Calcule et met en cache le KPI si nécessaire."""
        kpi_name = self.selected_indicator
        kpi_var = self.KPI_VAR_MAP.get(kpi_name)

        if not kpi_var or (kpi_var in self.ds and kpi_name != "STRESS THERMIQUE MAXIMAL"):
            return

        needs_update = False
        print(f"🛠️  Vérification/Calcul pour : {kpi_name}")

        if kpi_name == "STRESS THERMIQUE MAXIMAL":
            kpi_data = self.calculator.calculate_heat_stress_max(self.ds['Tair'], self.temperature_threshold)
            self.ds[kpi_var] = kpi_data
            needs_update = True
        elif kpi_name == "PERTE DE PONTE (%)" and kpi_var not in self.ds:
            kpi_data = self.calculator.calculate_laying_loss(self.ds['Tair'], self.ds['humidity'])
            self.ds[kpi_var] = kpi_data
            needs_update = True
        elif kpi_name == "PERTE DE PRODUCTION DE LAIT (%)" and kpi_var not in self.ds:
            kpi_data = self.calculator.calculate_milk_production_loss(self.ds['Tair'])
            self.ds[kpi_var] = kpi_data
            needs_update = True
        elif kpi_name == "PERTE DE GMQ - GAIN EN MASSE QUOTIDIEN (%)" and kpi_var not in self.ds:
            kpi_data = self.calculator.calculate_daily_weight_gain_loss(self.ds['Tair'], self.ds['humidity'])
            self.ds[kpi_var] = kpi_data
            needs_update = True
        # Ajouter d'autres calculs de KPI ici
        
        if needs_update:
            print(f"💾 Mise à jour du cache dans {self.DATA_FILE}...")
            self.ds.to_netcdf(self.DATA_FILE)
            print("✅ Cache mis à jour.")

    def _update_stress_kpi(self, *events):
        """Force le recalcul du stress thermique lorsque le seuil change."""
        if self.selected_indicator == "STRESS THERMIQUE MAXIMAL":
            self._ensure_kpi_is_calculated()
            self.param.trigger('selected_indicator') # Force map update

    @param.depends('selected_indicator', 'temperature_threshold')
    def get_map_view(self):
        """Génère la carte de l'indicateur sélectionné."""
        self._ensure_kpi_is_calculated()
        
        kpi_var = self.KPI_VAR_MAP.get(self.selected_indicator, 'Tair')
        title = f"{self.selected_indicator}"
        if self.selected_indicator == "STRESS THERMIQUE MAXIMAL":
            title += f" (Seuil: {self.temperature_threshold}°C)"

        print(f"🗺️  Affichage de la carte pour : {title}")
        
        return self.ds.hvplot.quadmesh(
            x='lon', y='lat', z=kpi_var,
            crs=ccrs.PlateCarree(), projection=ccrs.GOOGLE_MERCATOR,
            tiles=xyz.Esri.WorldImagery, project=True, rasterize=True,
            cmap='viridis', width=800, height=600, title=title
        )

    def get_panel(self):
        """Construit le layout final de l'application Panel."""
        
        step1_card = pn.Card(
            self.param.selected_category,
            title="1 - Choisir une catégorie", width=320
        )
        
        step2_card = pn.Card(
            self.param.selected_indicator,
            title="2 - Choisir un indicateur", width=320
        )

        # Le slider de température n'est visible que pour le stress thermique
        params_view = pn.panel(self.param.temperature_threshold, visible=pn.bind(lambda ind: ind == "STRESS THERMIQUE MAXIMAL", self.param.selected_indicator))
        
        step3_card = pn.Card(
            params_view,
            title="3 - Paramètres", width=320
        )

        sidebar = pn.Column(step1_card, step2_card, step3_card)
        
        layout = pn.template.MaterialTemplate(
            title="AGRO CLIMAT - Indicateurs (Version Complète)",
            sidebar=[sidebar],
            main=[self.get_map_view],
            header_background='#2596be', sidebar_width=340
        )
        return layout

# --- Point d'entrée ---

# Supprimer l'ancien cache pour éviter les conflits
if os.path.exists(AgroclimaticApp.DATA_FILE):
    print("🗑️ Suppression de l'ancien fichier de cache pour assurer la compatibilité.")
    os.remove(AgroclimaticApp.DATA_FILE)
    
app = AgroclimaticApp()
app.get_panel().servable()
