import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.gridspec import GridSpec

data_dir = "C:/Users/gabri/Downloads/All_Data_nc"

def open_tntr(exp):
    files = glob.glob(f"{data_dir}/tntr_CFmon_UKESM1-0-LL_{exp}*.nc")
    ds = xr.open_mfdataset(files, combine="by_coords", parallel=False)
    return ds["tntr"]            # (time, lev, lat, lon)

tntr_sulf  = open_tntr("G6sulfur")
tntr_solar = open_tntr("G6solar")

# -------- convert K s⁻1 → K day⁻1 for readability --------
tntr_sulf  = tntr_sulf * 86400
tntr_solar = tntr_solar * 86400

# -------- baseline σ for stippling (2000-14) --------
base = slice("2000-01-01", "2014-12-30")
σ_nat = (tntr_sulf.sel(time=base) - tntr_solar.sel(time=base)).std("time").mean("lon")

decades = [
    ("2020-2029", slice("2020-01-01", "2029-12-30")),
    ("2030-2039", slice("2030-01-01", "2039-12-30")),
    ("2040-2049", slice("2040-01-01", "2049-12-30")),
    ("2050-2059", slice("2050-01-01", "2059-12-30")),
    ("2060-2069", slice("2060-01-01", "2069-12-30")),
    ("2070-2079", slice("2070-01-01", "2079-12-30")),
    ("2080-2089", slice("2080-01-01", "2089-12-30")),
    ("2090-2099", slice("2090-01-01", "2099-12-30"))
]

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.2)

levels = np.arange(-0.4, 0.45, 0.05)
cmap = "RdBu_r"

for i, (decade_label, period) in enumerate(decades):
    row, col = divmod(i, 2)
    ax = fig.add_subplot(gs[row, col])
    
    tntr_sulf_dec = tntr_sulf.sel(time=period).mean("time")
    tntr_solar_dec = tntr_solar.sel(time=period).mean("time")
    Δtntr = (tntr_sulf_dec - tntr_solar_dec).mean("lon")  # (lev, lat)
    
    stipple = np.abs(Δtntr) < σ_nat  # boolean mask (lev, lat)
    
    # Plot data
    lev_km = Δtntr.lev / 1000    # convert m → km for y-axis
    pcm = ax.contourf(Δtntr.lat, lev_km, Δtntr, levels=levels,
                      cmap=cmap, extend="both")
    
    # Add stippling where signal < 1σ
    ax.contourf(Δtntr.lat, lev_km, stipple, levels=[0.5, 1], 
                hatches=["..."], colors="none")
    
    ax.set_ylim(33, 15)  # show 15–33 km only
    ax.set_title(f"{decade_label}")
    
    # Only add y-label for left column
    if col == 0:
        ax.set_ylabel("Altitude (km)")
    
    # Only add x-label for bottom row
    if row == 3:
        ax.set_xlabel("Latitude")

cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  
cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Δ tntr (K day⁻¹)", labelpad=5)

fig.suptitle("Stratospheric radiative-heating anomaly (G6sulfur – G6solar)", 
             fontsize=16, y=0.95)

plt.tight_layout(rect=[0, 0.08, 1, 0.93])  # Adjust for the colorbar and title
plt.show()
