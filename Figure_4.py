import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
from matplotlib.gridspec import GridSpec

data_dir = "C:/Users/gabri/Downloads/All_Data_nc"

def open_ta(exp):
    files = glob.glob(f"{data_dir}/ta_Amon_UKESM1-0-LL_{exp}*.nc")
    return xr.open_mfdataset(files, combine="by_coords", parallel=False)["ta"]

print("Loading G6sulfur data...")
ta_sulf = open_ta("G6sulfur")   # K
print("Loading G6solar data...")
ta_solar = open_ta("G6solar")

print(f"Available coordinates: {list(ta_sulf.coords)}")
print(f"Time encoding: {ta_sulf.time.encoding}")
print(f"First few time values: {ta_sulf.time.values[:5]}")
print(f"plev values: {ta_sulf.plev.values}")
print(f"Pressure level units: {ta_sulf.plev.attrs.get('units', 'unknown')}")

# ---------- altitude mask ----------
# Determine if pressure levels are in Pa or hPa based on the range of values
if ta_sulf.plev.max() > 1000:  # Likely in Pa
    # 24-26 km corresponds to approximately 30-25 hPa or 3000-2500 Pa
    target_plev_min, target_plev_max = 2500, 3000
    print(f"Using pressure range {target_plev_min}-{target_plev_max} Pa")
else:  # Likely in hPa
    # 24-26 km corresponds to approximately 30-25 hPa
    target_plev_min, target_plev_max = 25, 30
    print(f"Using pressure range {target_plev_min}-{target_plev_max} hPa")

# Create  mask
alt_mask = (ta_sulf.plev >= target_plev_min) & (ta_sulf.plev <= target_plev_max)
selected_plevs = ta_sulf.plev.values[alt_mask]
print(f"Selected pressure levels: {selected_plevs}")
if len(selected_plevs) == 0:
    print("WARNING: No pressure levels selected! Choosing closest available level instead.")
    # Find the closest pressure level
    closest_plev_idx = np.argmin(np.abs(ta_sulf.plev.values - np.mean([target_plev_min, target_plev_max])))
    closest_plev = ta_sulf.plev.values[closest_plev_idx]
    print(f"Using closest pressure level: {closest_plev}")
    alt_mask = ta_sulf.plev == closest_plev

decades = [
    ("2020s", slice("2020-01-01", "2029-12-30")),
    ("2030s", slice("2030-01-01", "2039-12-30")),
    ("2040s", slice("2040-01-01", "2049-12-30")),
    ("2050s", slice("2050-01-01", "2059-12-30")),
    ("2060s", slice("2060-01-01", "2069-12-30")),
    ("2070s", slice("2070-01-01", "2079-12-30")),
    ("2080s", slice("2080-01-01", "2089-12-30")),
    ("2090s", slice("2090-01-01", "2099-12-30"))
]

# ---------- baseline σ (2000-14) for stipple - using 30-day months calendar ----------
base = slice("2000-01-01", "2014-12-30")
print(f"Calculating baseline standard deviation for {base}...")
σ_nat = (ta_sulf.sel(time=base, plev=alt_mask).mean("plev") -
         ta_solar.sel(time=base, plev=alt_mask).mean("plev")).std("time")

# ---------- Calculate temperature differences for all decades ----------
print("Calculating temperature differences for all decades...")
all_dtas = []
for decade_name, period in decades:
    print(f"Processing {decade_name}...")
    ta_sulf_dec = ta_sulf.sel(time=period, plev=alt_mask).mean(dim=["time", "plev"])
    ta_solar_dec = ta_solar.sel(time=period, plev=alt_mask).mean(dim=["time", "plev"])
    dta = ta_sulf_dec - ta_solar_dec
    all_dtas.append(dta)
    print(f"  Shape: {dta.shape}, Range: {dta.min().values:.2f} to {dta.max().values:.2f} K")

# ---------- Find global min/max for consistent color scale ----------
all_vals = np.concatenate([dta.values.flatten() for dta in all_dtas])
vmax = np.ceil(np.nanmax(np.abs(all_vals)))
print(f"Using symmetric color scale with vmax={vmax}")

# ---------- Create a panel of maps using Cartopy ----------
print("Creating figure with maps...")
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 4, figure=fig, wspace=0.3, hspace=0.4)

# Create subplot for each decade
for i, (decade_name, period) in enumerate(decades):
    row, col = divmod(i, 4)
    
    projection = ccrs.Robinson()
    ax = fig.add_subplot(gs[row, col], projection=projection)
    
    dta = all_dtas[i]
    
    # Apply stippling where difference is less than natural variability
    stipple = np.abs(dta) < σ_nat
    
    # Plot the data on the map
    pcm = ax.pcolormesh(dta.lon, dta.lat, dta,
                      cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      transform=ccrs.PlateCarree())
    
    # Add stippling where differences are not significant
    # We need to convert stipple to a masked array with True values masked
    mask = np.ma.masked_where(~stipple, np.ones_like(stipple))
    ax.pcolor(dta.lon, dta.lat, mask, 
              hatch='...', alpha=0, transform=ccrs.PlateCarree())
    
    # Add coastlines and other map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    
    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.2, color='gray', 
                     alpha=0.5, linestyle='--', draw_labels=False)
    
    # Set extent to global
    ax.set_global()
    
    # Add title for this decade
    ax.set_title(f"{decade_name}", fontsize=12)

# Add a colorbar at the bottom spanning all subplots
cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
cb = fig.colorbar(pcm, cax=cbar_ax, orientation="horizontal")
cb.set_label("Δ Temperature (K)", fontsize=12)

# Add overall title
title_text = (f"Temperature anomaly Δta, 24–26 km (≈ 30 hPa)\n"
              f"G6sulfur – G6solar, Decadal Means\n"
              f"Stipple = |Δ| < 1 σ (2000-14)")
fig.suptitle(title_text, fontsize=16, y=0.98)

# Save figure
output_file = "temperature_anomaly_decades_map.png"
print(f"Saving figure to {output_file}...")
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()
print("Done!")
