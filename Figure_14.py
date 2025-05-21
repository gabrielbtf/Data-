import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm

data_dir = "C:/Users/gabri/Downloads/All_Data_nc"

def open_o3(exp):
    files = glob.glob(f"{data_dir}/o3_Amon_UKESM1-0-LL_{exp}*.nc")
    return xr.open_mfdataset(files, combine="by_coords", parallel=False)["o3"]

o3_sulf = open_o3("G6sulfur")
o3_solar = open_o3("G6solar")

# Barometric formula for standard atmosphere
H = 7.5  # scale height in km
P0 = 101325.0  # reference pressure in Pa
altitude = -H * np.log(o3_sulf.plev / P0)

alt_min, alt_max = 17, 33  # km
lat_min, lat_max = -90, -60  # degrees, Southern polar region (60°S-90°S)

alt_mask = (altitude >= alt_min) & (altitude <= alt_max)
lat_mask = (o3_sulf.lat >= lat_min) & (o3_sulf.lat <= lat_max)

# 360-day calendar has 30 days per month
oct_mask = (o3_sulf.time.dt.month == 10)

o3_sulf_oct = o3_sulf.sel(time=oct_mask).sel(time=slice("2015-01-01", "2100-12-30"))
o3_solar_oct = o3_solar.sel(time=oct_mask).sel(time=slice("2015-01-01", "2100-12-30"))

o3_sulf_alts = o3_sulf_oct.sel(plev=o3_sulf_oct.plev[alt_mask])
o3_solar_alts = o3_solar_oct.sel(plev=o3_solar_oct.plev[alt_mask])

# Then average over the altitude range and the latitude band
o3_sulf_mean = o3_sulf_alts.sel(lat=lat_mask).mean(dim=["plev", "lat"])
o3_solar_mean = o3_solar_alts.sel(lat=lat_mask).mean(dim=["plev", "lat"])

o3_diff = o3_sulf_mean - o3_solar_mean

baseline_mask = (o3_sulf.time.dt.month == 10) & (o3_sulf.time.dt.year >= 2000) & (o3_sulf.time.dt.year <= 2014)
o3_sulf_base = o3_sulf.sel(time=baseline_mask)
o3_solar_base = o3_solar.sel(time=baseline_mask)

o3_sulf_base_alts = o3_sulf_base.sel(plev=o3_sulf_base.plev[alt_mask])
o3_solar_base_alts = o3_solar_base.sel(plev=o3_solar_base.plev[alt_mask])
o3_sulf_base_mean = o3_sulf_base_alts.sel(lat=lat_mask).mean(dim=["plev", "lat"])
o3_solar_base_mean = o3_solar_base_alts.sel(lat=lat_mask).mean(dim=["plev", "lat"])

sigma_natural = (o3_sulf_base_mean - o3_solar_base_mean).std(dim="time")

stipple_mask = abs(o3_diff) < sigma_natural

# Extract years for the x-axis
years = o3_diff.time.dt.year.values

# Convert ozone values to ppm (× 10⁶) for easier interpretation
o3_diff_ppm = o3_diff * 1e6  # convert from mol mol⁻¹ to μmol mol⁻¹

fig, ax = plt.subplots(figsize=(12, 6))

vmax = np.ceil(
    max(
        abs(o3_diff_ppm.min().compute().item()),
        abs(o3_diff_ppm.max().compute().item())
    )
)

# Create Hovmöller diagram
pcm = ax.contourf(years, o3_diff.lon, o3_diff_ppm.T, 
                  levels=np.linspace(-vmax, vmax, 21),
                  cmap="RdBu_r", extend="both")

# Add stippling for areas below natural variability threshold
# Create grid for stippling
X, Y = np.meshgrid(years, o3_diff.lon)
ax.scatter(X[stipple_mask.T], Y[stipple_mask.T], s=1, c='k', alpha=0.3)

# Add colorbar
cbar = plt.colorbar(pcm, ax=ax, orientation="vertical", pad=0.02)
cbar.set_label("Δ O₃ (μmol mol⁻¹)", fontsize=12)

# Configure axes
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Longitude (°E)", fontsize=12)
ax.set_title("Antarctic Spring (October) Ozone Recovery Pulse\n"
             "G6sulfur - G6solar, 60°S-90°S, 17-33 km", fontsize=14)

# Add gridlines
ax.grid(linestyle='--', alpha=0.3)

# Show x-axis tick marks every 5 years
ax.set_xticks(np.arange(2015, 2105, 5))
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
