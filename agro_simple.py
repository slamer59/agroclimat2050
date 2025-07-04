import os

import cartopy.crs as ccrs
import hvplot.xarray
import panel as pn
import xarray as xr
import xyzservices.providers as xyz

# Initialize Panel
pn.extension()

# Load the rasm dataset
if os.path.exists('rasm.nc'):
    rasm = xr.open_dataset('rasm.nc')
else:

    rasm = xr.tutorial.open_dataset('rasm').load().isel(time=0)
    rasm.to_netcdf('rasm.nc')

# Plot using hvplot
plot = rasm.hvplot.quadmesh(
    'xc', 'yc',
    crs=ccrs.PlateCarree(),
    projection=ccrs.GOOGLE_MERCATOR,
    tiles=xyz.Esri.WorldImagery,
    project=True,
    rasterize=True,
    xlim=(-20, 40),
    ylim=(30, 70),
    cmap='viridis',
    frame_width=400,
)



# Display the plot using Panel
panel = pn.Row(plot)
panel.servable()
