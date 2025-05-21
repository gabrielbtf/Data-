import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter

data_dir = "C:/Users/gabri/Downloads/All_Data_nc"

# ---------- helper to open rtmt (net TOA radiation) data ----------
def open_rtmt(exp):
    files = glob.glob(f"{data_dir}/rtmt_Amon_UKESM1-0-LL_{exp}*.nc")
    return xr.open_mfdataset(files, combine="by_coords", parallel=False)["rtmt"]

print("Loading G6sulfur data...")
rtmt_sulf = open_rtmt("G6sulfur")   # W m-2
print("Loading G6solar data...")
rtmt_solar = open_rtmt("G6solar")

# ---------- Print diagnostic information ----------
print(f"Available coordinates: {list(rtmt_sulf.coords)}")
print(f"Time encoding: {rtmt_sulf.time.encoding}")
print(f"First few time values: {rtmt_sulf.time.values[:5]}")
print(f"Data dimensions: {rtmt_sulf.dims}")

# ---------- Calculate global mean time series ----------
# Area-weighted average, accounting for differences in grid cell size by latitude
# Calculate grid cell areas (proportional to cosine of latitude in radians)
coslat = np.cos(np.deg2rad(rtmt_sulf.lat))
weights = coslat / coslat.sum()

# Function to calculate global mean with latitude weights
def global_mean(da):
    # First, take zonal mean (average over longitudes)
    zonal_mean = da.mean(dim='lon')
    # Then do weighted average over latitudes
    return (zonal_mean * weights).sum(dim='lat')

# Calculate time series of global means
print("Calculating global mean time series...")
global_rtmt_sulf = global_mean(rtmt_sulf)
global_rtmt_solar = global_mean(rtmt_solar)

# ---------- Calculate annual means and standard deviations ----------
# First, create a new time coordinate with just the year
rtmt_sulf_year = rtmt_sulf.assign_coords(year=rtmt_sulf.time.dt.year)
rtmt_solar_year = rtmt_solar.assign_coords(year=rtmt_solar.time.dt.year)

# Group by year to calculate annual means
annual_rtmt_sulf = global_mean(rtmt_sulf_year).groupby('time.year').mean()
annual_rtmt_solar = global_mean(rtmt_solar_year).groupby('time.year').mean()

# Group by year to calculate inter-annual standard deviations
annual_rtmt_sulf_std = global_mean(rtmt_sulf_year).groupby('time.year').std()
annual_rtmt_solar_std = global_mean(rtmt_solar_year).groupby('time.year').std()

# ---------- Create figure ----------
fig, ax = plt.figure(figsize=(12, 6)), plt.axes()

# Extract years as numeric values for plotting
years_sulf = annual_rtmt_sulf.year.values
years_solar = annual_rtmt_solar.year.values

# Extract the values from the data arrays
values_sulf = annual_rtmt_sulf.values
values_solar = annual_rtmt_solar.values
std_sulf = annual_rtmt_sulf_std.values
std_solar = annual_rtmt_solar_std.values

# Apply Savitzky-Golay filter to smooth the time series (optional)
window_size = 11  # Must be odd number
poly_order = 3    # Polynomial order for fitting
if len(years_sulf) > window_size:
    values_sulf_smooth = savgol_filter(values_sulf, window_size, poly_order)
    values_solar_smooth = savgol_filter(values_solar, window_size, poly_order)
else:
    values_sulf_smooth = values_sulf
    values_solar_smooth = values_solar

# Plot time series with standard deviation shading
ax.plot(years_sulf, values_sulf_smooth, color='darkorange', lw=2, label='G6sulfur')
ax.fill_between(years_sulf, values_sulf_smooth-std_sulf, values_sulf_smooth+std_sulf, 
                color='darkorange', alpha=0.2)

ax.plot(years_solar, values_solar_smooth, color='skyblue', lw=2, label='G6solar')
ax.fill_between(years_solar, values_solar_smooth-std_solar, values_solar_smooth+std_solar, 
                color='skyblue', alpha=0.2)

# Calculate and plot difference between scenarios
diff = values_sulf_smooth - values_solar_smooth
ax.plot(years_sulf, diff, color='darkgreen', lw=1.5, linestyle='--', 
        label='Difference (G6sulfur - G6solar)')

# Add horizontal line at y=0 for reference
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

# Add vertical line at 2020 (approximate start of intervention)
ax.axvline(x=2020, color='gray', linestyle='--', alpha=0.5)
ax.text(2021, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
        'Intervention start', rotation=90, verticalalignment='bottom')

# Enhance the plot with grid, labels, title, and legend
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Net TOA Radiative Flux (W m$^{-2}$)', fontsize=12)
ax.set_title('Global Mean Net TOA Radiative Forcing (rtmt)\nG6sulfur vs G6solar', fontsize=14)

# Set x-axis limits to focus on the relevant period
ax.set_xlim(2000, 2100)

# Add legend and adjust layout
ax.legend(loc='upper right', frameon=True, framealpha=0.8)
plt.tight_layout()

# Save figure
output_file = "Figure6_TOA_forcing_comparison.png"
print(f"Saving figure to {output_file}...")
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()
print("Done!")