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

# --- Arbre de Décision ---
DECISION_TREE = {
    "ANIMAUX": {
        "STRESS THERMIQUE MAXIMAL": {
            "variable_name": "stress_thermique_max",
            "type": {
                "VACHE LAITIÈRE": {
                    "race": ["PRIM'HOLSTEIN", "HOLSTEIN", "NORMANDE"],
                    "weather_models": {
                        "AROME": {"label": "🔍 AROME — HD court terme (2 jours)", "max_days": 2},
                        "ARPEGE": {"label": "🌐 ARPEGE — moyen terme (4 jours)", "max_days": 4}
                    }
                },
                "PORC": {
                    "race": ["PIÉTRAIN", "LARGE WHITE"],
                    "weather_models": {
                        "ARPEGE": {"label": "🌐 ARPEGE (4 jours)", "max_days": 4},
                        "GFS": {"label": "🌍 GFS — global (10 jours)", "max_days": 10}
                    }
                }
            }
        },
        "PERTE DE PONTE (%)": {
            "variable_name": "perte_ponte",
            "type": {
                "POULE PONDEUSE": {
                    "race": ["COB", "LEGHORN"],
                    "weather_models": {
                        "ARPEGE": {"label": "🌐 ARPEGE (4 jours)", "max_days": 4},
                        "GFS": {"label": "🌍 GFS — global (10 jours)", "max_days": 10}
                    }
                }
            }
        },
        "PERTE DE PRODUCTION DE LAIT (%)": {
            "variable_name": "perte_lait",
            "type": {
                "VACHE LAITIÈRE": {
                    "race": ["PRIM'HOLSTEIN", "HOLSTEIN", "NORMANDE"],
                    "weather_models": {
                        "ARPEGE": {"label": "🌐 ARPEGE (4 jours)", "max_days": 4},
                    }
                }
            }
        },
        "PERTE DE GMQ - GAIN EN MASSE QUOTIDIEN (%)": {
            "variable_name": "perte_gmq",
            "type": {
                "POULE DE CHAIR": {
                    "race": ["COB", "LEGHORN"],
                    "weather_models": {
                        "ARPEGE": {"label": "🌐 ARPEGE (4 jours)", "max_days": 4}
                    }
                }
            }
        }
    },
    "MALADIE": {
        "CROISSANCE": {
            "variable_name": "maladie_croissance",
            "type": {
                "BLE": {
                    "type_maladie": ["OIDIUM", "ROUGE"],
                    "weather_models": {
                        "AROME": {"label": "🔍 AROME (2 jours)", "max_days": 2},
                        "ARPEGE": {"label": "🌐 ARPEGE (4 jours)", "max_days": 4}
                    }
                },
                "VIGNE": {
                    "type_maladie": ["MILDIOU"],
                    "weather_models": {
                        "AROME": {"label": "🔍 AROME (2 jours)", "max_days": 2},
                        "ARPEGE": {"label": "🌐 ARPEGE (4 jours)", "max_days": 4}
                    }
                }
            }
        }
    },
    "RAVAGEUR": {
        "CROISSANCE": {
            "variable_name": "ravageur_croissance",
            "type": {
                "BLE": {
                    "type_ravageur": ["Puceron sur épis"],
                    "weather_models": {
                        "AROME": {"label": "🔍 AROME (2 jours)", "max_days": 2},
                        "ARPEGE": {"label": "🌐 ARPEGE (4 jours)", "max_days": 4}
                    }
                },
                "POMME DE TERRE": {
                    "type_ravageur": ["Charançon des pommes de terre"],
                    "weather_models": {
                        "AROME": {"label": "🔍 AROME (2 jours)", "max_days": 2},
                    }
                }
            }
        }
    },
    "VÉGÉTAUX": {
        "POTENTIEL DE PERTES PAR GEL (%)": {
            "variable_name": "pertes_gel",
            "parameters": {
                "Espèce": ["ABRICOTIER"],
                "Stade de développement": ["REPOS HIVERNAL"],
                "weather_models": {
                    "AROME": {"label": "🔍 AROME — HD court terme (2 jours)", "max_days": 2},
                    "ARPEGE": {"label": "🌐 ARPEGE — moyen terme (4 jours)", "max_days": 4},
                    "GFS": {"label": "🌡️ GFS — Global (10 jours)", "max_days": 10},
                    "DMI": {"label": "🌎 DMI — Nordique (2 jours)", "max_days": 2},
                    "ICON_monde": {"label": "🌎 ICON — Monde (7 jours)", "max_days": 7},
                    "ICON_europe": {"label": "🌎 ICON — Europe (5 jours)", "max_days": 5}
                }
            }
        },
        "VITESSE DE CROISSANCE": {
            "variable_name": "vitesse_croissance",
            "type": {
                "BLE": {
                    "weather_models": {
                        "AROME": {"label": "🔍 AROME — HD court terme (2 jours)", "max_days": 2},
                        "ARPEGE": {"label": "🌐 ARPEGE — moyen terme (4 jours)", "max_days": 4}
                    }
                }
            }
        },
        "STRESS THERMIQUE": {
            "variable_name": "stress_thermique_vegetaux",
            "type": {
                "BLE": {
                    "weather_models": {
                        "AROME": {"label": "🔍 AROME — HD court terme (2 jours)", "max_days": 2},
                        "ARPEGE": {"label": "🌐 ARPEGE — moyen terme (4 jours)", "max_days": 4}
                    }
                }
            }
        }
    }
}

# --- Modèle de Données ---
class AgroModel(param.Parameterized):
    category = param.ObjectSelector(default="ANIMAUX", objects=list(DECISION_TREE.keys()))
    indicator = param.ObjectSelector(objects=[], allow_None=True)
    type = param.ObjectSelector(objects=[], allow_None=True)
    parameters = param.ObjectSelector(objects=[], allow_None=True)
    race = param.ObjectSelector(objects=[], allow_None=True)
    weather_model = param.ObjectSelector(objects=[], allow_None=True)

    def __init__(self, **params):
        super().__init__(**params)
        self._update_indicator()

    @param.depends('category', watch=True)
    def _update_indicator(self):
        indicators = list(DECISION_TREE.get(self.category, {}).keys())
        self.param.indicator.objects = indicators
        self.indicator = indicators[0] if indicators else None

    @param.depends('indicator', watch=True)
    def _update_type(self):
        data = DECISION_TREE.get(self.category, {}).get(self.indicator, {})
        types = list(data.get('type', {}).keys())
        self.param.type.objects = types
        self.type = types[0] if types else None

    @param.depends('type', watch=True)
    def _update_parameters(self):
        data = DECISION_TREE.get(self.category, {}).get(self.indicator, {})
        if 'parameters' in data:
            params = list(data.get('parameters', {}).keys())
            self.param.parameters.objects = params
            self.parameters = params[0] if params else None
        else:
            self.param.parameters.objects = []
            self.parameters = None

    @param.depends('parameters', 'type', watch=True)
    def _update_race_and_weather(self):
        data = DECISION_TREE.get(self.category, {}).get(self.indicator, {})
        if self.type and 'type' in data:
            sub_data = data['type'].get(self.type, {})
        elif self.parameters and 'parameters' in data:
            sub_data = data['parameters']
        else:
            sub_data = {}

        races = sub_data.get('race', [])
        self.param.race.objects = races
        self.race = races[0] if races else None

        weather_models = list(sub_data.get('weather_models', {}).keys())
        self.param.weather_model.objects = weather_models
        self.weather_model = weather_models[0] if weather_models else None

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
    """Application principale avec UI dynamique et backend xarray."""

    # --- Modèle de données et UI ---
    agro_model = param.ClassSelector(AgroModel, default=AgroModel())

    # --- Paramètres spécifiques à l'indicateur ---
    temperature_threshold = param.Number(
        default=30.0, bounds=(15.0, 40.0), step=0.5,
        doc="Seuil de température (°C) pour le stress thermique"
    )

    # --- État interne ---
    ds = param.Parameter(doc="Le dataset xarray contenant les données et les KPIs.")
    calculator = param.ClassSelector(IndicatorCalculator, default=IndicatorCalculator())
    map_pane = param.Parameter(doc="Le conteneur stable pour la carte afin d'éviter le flash.")
    DATA_FILE = "agro_data.nc"

    def __init__(self, **params):
        super().__init__(**params)
        self.ds = self._load_or_create_dataset()
        self.map_pane = pn.pane.HoloViews(None, width=800, height=600)
        
        # Watchers for dynamic updates
        self.agro_model.param.watch(self._update_map_view, 'indicator')
        self.param.watch(self._update_map_view, 'temperature_threshold')
        
        self._update_map_view() # Initial call to display the map

    def _load_or_create_dataset(self):
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

    def _ensure_kpi_is_calculated(self, kpi_name, kpi_var):
        if kpi_var and (kpi_var in self.ds and kpi_name != "STRESS THERMIQUE MAXIMAL"):
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
        
        if needs_update:
            print(f"💾 Mise à jour du cache dans {self.DATA_FILE}...")
            self.ds.to_netcdf(self.DATA_FILE)
            print("✅ Cache mis à jour.")

    def _get_map_object(self):
        kpi_name = self.agro_model.indicator
        if not kpi_name:
            return pn.pane.Markdown("### Veuillez sélectionner un indicateur.")

        # Retrieve kpi_var directly from DECISION_TREE
        category_data = DECISION_TREE.get(self.agro_model.category, {})
        indicator_data = category_data.get(kpi_name, {})
        kpi_var = indicator_data.get("variable_name")

        if not kpi_var:
             return pn.pane.Markdown(f"### L'indicateur '{kpi_name}' n'est pas encore implémenté ou n'a pas de variable associée.")

        self._ensure_kpi_is_calculated(kpi_name, kpi_var)
        
        title = f"{kpi_name}"
        if kpi_name == "STRESS THERMIQUE MAXIMAL":
            title += f" (Seuil: {self.temperature_threshold}°C)"

        print(f"🗺️  Génération de la vue pour : {title}")
        
        return self.ds.hvplot.quadmesh(
            x='lon', y='lat', z=kpi_var,
            crs=ccrs.PlateCarree(), projection=ccrs.GOOGLE_MERCATOR,
            tiles=xyz.Esri.WorldImagery, project=True, rasterize=True,
            cmap='viridis', title=title
        ).opts(width=800, height=600)

    def _update_map_view(self, *events):
        """Met à jour l'objet dans le conteneur de carte."""
        self.map_pane.object = self._get_map_object()

    def get_panel(self):
        # --- Panneau de contrôle dynamique ---
        selection_panel = pn.Card(
            pn.Param(self.agro_model.param, parameters=['category', 'indicator', 'type', 'parameters', 'race', 'weather_model']),
            title="1 - Sélection de l'indicateur",
            width=320
        )

        # --- Panneau de paramètres contextuels ---
        params_view = pn.panel(
            self.param.temperature_threshold, 
            visible=pn.bind(lambda ind: ind == "STRESS THERMIQUE MAXIMAL", self.agro_model.param.indicator)
        )
        params_card = pn.Card(params_view, title="2 - Paramètres de l'indicateur", width=320)

        sidebar = pn.Column(selection_panel, params_card)
        
        layout = pn.template.MaterialTemplate(
            title="AGRO CLIMAT - Indicateurs (Version Complète)",
            sidebar=[sidebar],
            main=[self.map_pane],
            header_background='#2596be', sidebar_width=340
        )
        return layout

if os.path.exists(AgroclimaticApp.DATA_FILE):
    print("🗑️ Suppression de l'ancien fichier de cache pour assurer la compatibilité.")
    os.remove(AgroclimaticApp.DATA_FILE)

app = AgroclimaticApp()
app.get_panel().servable()