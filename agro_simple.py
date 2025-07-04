import os

import cartopy.crs as ccrs
import hvplot.xarray
import numpy as np
import panel as pn
import xarray as xr
import xyzservices.providers as xyz

# Initialize Panel
pn.extension()

# --- Calculation functions from agro_indicator_app.py ---
# Simplified and adapted for xarray

def calculate_heat_stress_max(temp_data, threshold=30):
    """Calcule le stress thermique maximal."""
    stress_factor = (temp_data - threshold) / threshold * 100
    stress_classes = (
        (stress_factor > 0).astype(np.float32) +
        (stress_factor > 10).astype(np.float32) +
        (stress_factor > 25).astype(np.float32) +
        (stress_factor > 50).astype(np.float32)
    )
    return stress_classes

def calculate_laying_loss(temp_data, humidity_data):
    """Calcule la perte de ponte."""
    total_stress = (np.abs(temp_data - 20) / 20 + np.abs(humidity_data - 60) / 60) * 50
    stress_classes = (
        (total_stress > 5).astype(np.float32) +
        (total_stress > 15).astype(np.float32) +
        (total_stress > 30).astype(np.float32) +
        (total_stress > 50).astype(np.float32)
    )
    return stress_classes

# --- Data Loading and KPI Calculation ---

# Load the rasm dataset
if os.path.exists('rasm.nc'):
    rasm = xr.open_dataset('rasm.nc')
else:
    rasm = xr.tutorial.open_dataset('rasm').load().isel(time=0)
    rasm.to_netcdf('rasm.nc')

# Check for KPIs and calculate if missing
needs_update = False
if 'stress' not in rasm.data_vars:
    print("Calculating Stress KPI...")
    rasm['stress'] = calculate_heat_stress_max(rasm['Tair'])
    rasm['stress'].attrs['long_name'] = 'Heat Stress Level'
    needs_update = True

if 'laying_loss' not in rasm.data_vars:
    print("Calculating Laying Loss KPI...")
    # rasm doesn't have humidity, creating dummy data
    humidity_data = xr.DataArray(
        np.random.uniform(40, 80, size=rasm['Tair'].shape),
        dims=rasm['Tair'].dims,
        coords=rasm['Tair'].coords
    )
    rasm['laying_loss'] = calculate_laying_loss(rasm['Tair'], humidity_data)
    rasm['laying_loss'].attrs['long_name'] = 'Laying Loss Level'
    needs_update = True

if needs_update:
    print("Saving updated dataset with KPIs to rasm.nc...")
    rasm.to_netcdf('rasm.nc')


# --- UI and Plotting ---

kpi_selector = pn.widgets.Select(
    name='Select KPI',
    options=['Tair', 'stress', 'laying_loss'],
    value='Tair'
)

@pn.depends(kpi_selector.param.value)
def get_plot(kpi):
    # Plot using hvplot
    plot = rasm.hvplot.quadmesh(
        x='xc', y='yc', z=kpi,
        crs=ccrs.PlateCarree(),
        projection=ccrs.GOOGLE_MERCATOR,
        tiles=xyz.Esri.WorldImagery,
        project=True,
        rasterize=True,
        xlim=(-20, 40),
        ylim=(30, 70),
        cmap='viridis',
        frame_width=400,
        title=f'Map of {kpi}'
    )
    return plot

# Display the plot using Panel
panel = pn.Column(kpi_selector, get_plot)
panel.servable()