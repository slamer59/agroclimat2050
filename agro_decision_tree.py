import panel as pn
import param

pn.extension()

DECISION_TREE = {
    "ANIMAUX": {
        "STRESS THERMIQUE MAXIMAL": {
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
        "STRESS MOYEN": {
            "type": {
                "VACHE LAITIÈRE": {
                    "race": ["PRIM'HOLSTEIN", "HOLSTEIN", "NORMANDE"],
                    "weather_models": {
                        "ARPEGE": {"label": "🌐 ARPEGE (4 jours)", "max_days": 4},
                        "GFS": {"label": "🌍 GFS — global (10 jours)", "max_days": 10}
                    }
                }
            }
        },
        "PERTE DE PONTE": {
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


class AgroModel(param.Parameterized):
    category = param.ObjectSelector(default="ANIMAUX", objects=list(DECISION_TREE.keys()))
    indicator = param.ObjectSelector(objects=[])
    type = param.ObjectSelector(objects=[])
    parameters = param.ObjectSelector(objects=[])
    race = param.ObjectSelector(objects=[])
    weather_model = param.ObjectSelector(objects=[])


    @param.depends('category', watch=True)
    def _update_indicator(self):
        if self.category in DECISION_TREE:
            self.param['indicator'].objects = list(DECISION_TREE[self.category].keys())
            if self.param['indicator'].objects:
                self.indicator = self.param['indicator'].objects[0]
        else:
            self.param['indicator'].objects = []
        self._update_type()


    @param.depends('indicator', watch=True)
    def _update_type(self):
        if self.indicator and self.category in DECISION_TREE and self.indicator in DECISION_TREE[self.category]:
            data = DECISION_TREE[self.category][self.indicator]
            if 'type' in data:
                self.param['type'].objects = list(data['type'].keys())
                if self.param['type'].objects:
                    self.type = self.param['type'].objects[0]
            else:
                self.param['type'].objects = []
        else:
            self.param['type'].objects = []
        self._update_parameters()


    @param.depends('type', watch=True)
    def _update_parameters(self):
        if self.indicator and self.category in DECISION_TREE and self.indicator in DECISION_TREE[self.category]:
            data = DECISION_TREE[self.category][self.indicator]
            if 'parameters' in data:
                self.param['parameters'].objects = list(data['parameters'].keys())
                if self.param['parameters'].objects:
                    self.parameters = self.param['parameters'].objects[0]
            elif 'type' in data and self.type:
                self.param['parameters'].objects = []
            else:
                self.param['parameters'].objects = []
        else:
            self.param['parameters'].objects = []
        self._update_races_and_models()


    @param.depends('parameters', watch=True)
    def _update_races_and_models(self):
        if self.indicator and self.category in DECISION_TREE and self.indicator in DECISION_TREE[self.category]:
            data = DECISION_TREE[self.category][self.indicator]
            if 'type' in data and self.type:
                data = data['type'][self.type]
            elif 'parameters' in data and self.parameters:
                data = data['parameters']
            else:
                data = {}

            self.param['race'].objects = data.get('race', [])
            weather_models = list(data.get('weather_models', {}).keys())
            self.param['weather_model'].objects = weather_models

            if self.param['race'].objects:
                self.race = self.param['race'].objects[0]

            if weather_models:
                self.weather_model = weather_models[0]
        else:
            self.param['race'].objects = []
            self.param['weather_model'].objects = []


    def get_panel(self):
        return pn.Column(
            pn.widgets.StaticText(value="1 - Choisir une catégorie"),
            pn.Param(self.param.category, widgets={'category': pn.widgets.Select}),
            pn.widgets.StaticText(value="2 - Choisir un indicateur"),
            pn.Param(self.param.indicator, widgets={'indicator': pn.widgets.Select}),
            pn.widgets.StaticText(value="3 - Choisir un type"),
            pn.Param(self.param.type, widgets={'type': pn.widgets.Select}),
            pn.widgets.StaticText(value="4 - Choisir les paramètres"),
            pn.Param(self.param.parameters, widgets={'parameters': pn.widgets.Select}),
            pn.widgets.StaticText(value="5 - Choisir une race"),
            pn.Param(self.param.race, widgets={'race': pn.widgets.Select}),
            pn.widgets.StaticText(value="6 - Choisir un modèle météo"),
            pn.Param(self.param.weather_model, widgets={'weather_model': pn.widgets.Select}),
        )


model = AgroModel()
model.get_panel().servable()
