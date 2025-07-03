import datetime

import panel as pn
import param

pn.extension()

# Base classes for shared parameters
class AnimalParams(param.Parameterized):
    animal_type = param.Selector(default="VACHE LAITIÈRE", objects=[
        "VACHE LAITIÈRE", "Vache allaitante", "Poule de chair", "Poule pondeuse"
    ])
    simulation_mode = param.Boolean(default=False)
    temperature_offset = param.Number(default=0, bounds=(-5, 5))

class DiseaseParams(param.Parameterized):
    growth_factor = param.Number(default=1.0, bounds=(0.5, 2.0))

# Main application class
class CEMATApp(param.Parameterized):
    # 1. Category selection
    category = param.Selector(
        default="ANIMAUX",
        objects=["ANIMAUX", "MALADIES", "FEUX DE FORÊT", "POLLENS", "VÉGÉTAUX"]
    )
    
    # 2. Indicator selection (depends on category)
    animal_indicators = ["STRESS THERMIQUE MAXIMAL", "STRESS THERMIQUE MOYEN", 
                         "PERTE DE PONTE (%)", "PERTE DE PRODUCTION DE LAIT (%)"]
    disease_indicators = ["CROISSANCE"]
    
    indicator = param.Selector()
    
    # 3. Parameters (depend on category and indicator)
    animal_params = param.ClassSelector(class_=AnimalParams, default=AnimalParams())
    disease_params = param.ClassSelector(class_=DiseaseParams, default=DiseaseParams())
    
    # 4. Weather models
    model1_name = param.Selector(default="AROME", objects=["AROME", "ARPEGE", "GFS"])
    model1_term = param.Selector(default="HD court terme", objects=["HD court terme", "HD moyen terme"])
    model1_days = param.Integer(default=2, bounds=(1, 3))
    
    model2_name = param.Selector(default="GFS", objects=["AROME", "ARPEGE", "GFS"])
    model2_term = param.Selector(default="Global", objects=["Global"])
    model2_days = param.Integer(default=10, bounds=(5, 14))
    
    forecast_date = param.Date(default=datetime.date.today())
    
    # Location and result
    location = param.String(default="Nantes")
    result_value = param.Number(default=68.8)

    def __init__(self, **params):
        super().__init__(**params)
        self._update_indicators()
        self._update_dependencies()
        
    @param.depends('category', watch=True)
    def _update_indicators(self):
        if self.category == "ANIMAUX":
            self.param.indicator.objects = self.animal_indicators
            if not self.indicator or self.indicator not in self.animal_indicators:
                self.indicator = self.animal_indicators[0]
        elif self.category == "MALADIES":
            self.param.indicator.objects = self.disease_indicators
            if not self.indicator or self.indicator not in self.disease_indicators:
                self.indicator = self.disease_indicators[0]
        else:
            self.param.indicator.objects = []
            self.indicator = None
    
    @param.depends('model1_name', watch=True)
    def _update_model1_terms(self):
        if self.model1_name == "AROME":
            self.param.model1_term.objects = ["HD court terme", "HD moyen terme"]
            self.model1_term = "HD court terme"
            self.model1_days = 2
        elif self.model1_name == "GFS":
            self.param.model1_term.objects = ["Global"]
            self.model1_term = "Global"
            self.model1_days = 10
    
    @param.depends('model2_name', watch=True)
    def _update_model2_terms(self):
        if self.model2_name == "AROME":
            self.param.model2_term.objects = ["HD court terme", "HD moyen terme"]
            self.model2_term = "HD moyen terme"
            self.model2_days = 3
        elif self.model2_name == "GFS":
            self.param.model2_term.objects = ["Global"]
            self.model2_term = "Global"
            self.model2_days = 14
            
    def _update_dependencies(self):
        self._update_indicators()
        self._update_model1_terms()
        self._update_model2_terms()

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

# Create app instance
# Create app instance
app = CEMATApp()

# Layout for GUI
layout = pn.Column(
    pn.Row(
        app.param.category,
        app.param.indicator
    ),
    pn.Row(
        app.param.model1_name,
        app.param.model1_term,
        app.param.model1_days
    ),
    pn.Row(
        app.param.model2_name,
        app.param.model2_term,
        app.param.model2_days
    ),
    app.param.forecast_date,
    app.param.location,
    pn.indicators.Number(value=app.param.result_value, format='{value}'),
    app.param.animal_params  # Automatically shows animal parameters
)

layout.servable()