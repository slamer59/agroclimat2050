import os
from datetime import datetime

import cartopy.crs as ccrs
import geoviews as gv
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
from holoviews.operation.datashader import rasterize

# Enable extensions
pn.extension()
hv.extension('bokeh')
gv.extension('bokeh')

class ThermalStressDashboard:
    def __init__(self, cache_file='thermal_stress_cache.parquet'):
        self.cache_file = cache_file
        self.df = None
        self.kpis = None
        self.use_cache = True
        
        # Create widgets
        self.cache_toggle = pn.widgets.Toggle(name='Use Cache', value=True)
        self.regenerate_btn = pn.widgets.Button(name='Regenerate Data', button_type='primary')
        self.resolution_slider = pn.widgets.IntSlider(name='Grid Resolution', start=10, end=100, step=10, value=50)
        self.stress_range = pn.widgets.RangeSlider(name='Stress Range', start=0, end=100, value=(30, 70), step=5)
        
        # Set up callbacks
        self.regenerate_btn.on_click(self.regenerate_data)
        self.resolution_slider.param.watch(self.update_plot, 'value')
        self.stress_range.param.watch(self.update_plot, 'value')
        
        # Initialize dashboard
        self.update_data()
        
    def generate_sample_data(self, resolution=50):
        """Generate sample geographic temperature and humidity data"""
        np.random.seed(42)
        
        # Create a geographic grid
        lats = np.linspace(25, 50, resolution)
        lons = np.linspace(-125, -65, resolution)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Flatten for DataFrame
        lats_flat = lat_grid.flatten()
        lons_flat = lon_grid.flatten()
        
        # Generate temperature and humidity data
        base_temp = 25 + (lats_flat - 25) * 0.5  # Temperature increases south to north
        temp_variation = np.random.normal(0, 5, size=lats_flat.shape)
        temperatures = base_temp + temp_variation
        
        base_humidity = 50 + (lats_flat - 25) * 0.8  # Humidity increases south to north
        humidity_variation = np.random.normal(0, 10, size=lats_flat.shape)
        humidities = np.clip(base_humidity + humidity_variation, 20, 100)
        
        # Calculate thermal stress (simplified index)
        thermal_stress = temperatures * (1 + (humidities / 100))
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'latitude': lats_flat,
            'longitude': lons_flat,
            'temperature': temperatures,
            'humidity': humidities,
            'thermal_stress': thermal_stress
        })
        
        # Save to cache
        self.df.to_parquet(self.cache_file)
        
        return self.df
    
    def update_data(self):
        """Load or generate data based on current settings"""
        self.use_cache = self.cache_toggle.value
        
        if self.use_cache and os.path.exists(self.cache_file):
            print("Loading data from cache...")
            self.df = pd.read_parquet(self.cache_file)
        else:
            print("Generating new data...")
            self.generate_sample_data(self.resolution_slider.value)
        
        self.update_kpis()
        self.update_plot()
    
    def regenerate_data(self, event):
        """Force data regeneration"""
        print("Regenerating data...")
        self.generate_sample_data(self.resolution_slider.value)
        self.update_kpis()
        self.update_plot()
    
    def update_kpis(self):
        """Calculate and update KPIs"""
        if self.df is None:
            return
            
        # Calculate KPIs
        max_stress = float(self.df['thermal_stress'].max())
        avg_stress = float(self.df['thermal_stress'].mean())
        
        self.kpis = {
            "STRESS THERMIQUE MAXIMAL": max_stress,
            "STRESS THERMIQUE MOYEN": avg_stress,
            "calculation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_points": len(self.df)
        }
    
    def update_plot(self, *events):
        """Update the map plot based on current settings"""
        if self.df is None:
            return
            
        # Create points and rasterize
        points = gv.Points(
            self.df, 
            ['longitude', 'latitude'], 
            ['thermal_stress']
        ).opts(
            color='thermal_stress',
            cmap='fire_r',
            colorbar=True,
            tools=['hover'],
            size=8,
            alpha=0.7,
            projection=ccrs.PlateCarree()
        )
        
        # Rasterize for better performance
        self.rasterized = rasterize(points).opts(
            width=700,
            height=500,
            clim=(self.stress_range.value[0], self.stress_range.value[1]),
            title=f"Thermal Stress Distribution (Resolution: {self.resolution_slider.value}x{self.resolution_slider.value})"
        )
        
        # Create tiles
        self.tiles = gv.tile_sources.OSM().opts(alpha=0.5)
        
        # Combine tiles and points
        self.plot = self.tiles * self.rasterized
    
    def create_kpi_card(self, title, value):
        """Create a styled KPI card"""
        return pn.Card(
            pn.indicators.Number(
                name=title,
                value=value,
                format='{value:.2f}',
                colors=[(50, 'red'), (100, 'gold'), (float('inf'), 'green')],
                font_size='54pt'
            ),
            styles={'background': '#f0f0f0'},
            hide_header=True,
            margin=5
        )
    
    def view(self):
        """Create the dashboard view"""
        if self.kpis is None:
            self.update_kpis()
        
        # Create KPI cards
        max_stress_card = self.create_kpi_card(
            "STRESS THERMIQUE MAXIMAL", 
            self.kpis["STRESS THERMIQUE MAXIMAL"]
        )
        
        avg_stress_card = self.create_kpi_card(
            "STRESS THERMIQUE MOYEN", 
            self.kpis["STRESS THERMIQUE MOYEN"]
        )
        
        # Create info pane
        info_pane = pn.pane.Markdown(f"""
        **Last calculated:** {self.kpis["calculation_time"]}  
        **Data points:** {self.kpis["data_points"]:,}  
        **Resolution:** {self.resolution_slider.value}x{self.resolution_slider.value}
        """)
        
        # Create control panel
        controls = pn.Card(
            pn.Column(
                self.cache_toggle,
                self.regenerate_btn,
                self.resolution_slider,
                self.stress_range,
                info_pane
            ),
            title="Controls",
            margin=10
        )
        
        # Create dashboard layout
        dashboard = pn.Column(
            pn.Row(
                pn.Column(max_stress_card, width=300),
                pn.Column(avg_stress_card, width=300),
                controls
            ),
            self.plot if hasattr(self, 'plot') else pn.pane.Markdown("Generating plot..."),
            sizing_mode='stretch_width'
        )
        
        return dashboard

# Create and serve the dashboard
dashboard = ThermalStressDashboard()

# For Jupyter notebook
# dashboard.view()

# To serve as a standalone app
app = pn.template.FastListTemplate(
    title="Thermal Stress Analysis Dashboard",
    header_background="#1f77b4",
    main=[dashboard.view()],
).servable()

# To run from command line: panel serve thermal_stress_dashboard.py --show