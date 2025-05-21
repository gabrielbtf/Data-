import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.gridspec import GridSpec

data_dir = "C:/Users/gabri/Downloads/All_Data_nc"   # <-- adjust if needed

# ---------- quick open helpers ----------
def open_var(exp, var, table="Amon"):                # Amon for wap and o3
    files = glob.glob(f"{data_dir}/{var}_{table}_UKESM1-0-LL_{exp}*.nc")
    return xr.open_mfdataset(files, combine="by_coords", parallel=False)[var]

wap_sulf  = open_var("G6sulfur", "wap")              # Pa s⁻1
wap_solar = open_var("G6solar" , "wap")
o3_sulf   = open_var("G6sulfur", "o3")               # mol mol-1
o3_solar  = open_var("G6solar" , "o3")
# ----------- EP-flux proxy: vertical shear of zonal wind, 2020s–2090s panel plot -----------

# --- load ua ---
ua_sulf  = open_var("G6sulfur", "ua")
ua_solar = open_var("G6solar" , "ua")

# --- Set up decades ---
decades = {
    "2020s": slice("2020-01-01", "2029-12-30"),
    "2030s": slice("2030-01-01", "2039-12-30"),
    "2040s": slice("2040-01-01", "2049-12-30"),
    "2050s": slice("2050-01-01", "2059-12-30"),
    "2060s": slice("2060-01-01", "2069-12-30"),
    "2070s": slice("2070-01-01", "2079-12-30"),
    "2080s": slice("2080-01-01", "2089-12-30"),
    "2090s": slice("2090-01-01", "2099-12-30"),
}

g  = 9.80665       # m s-2
H  = 7.5           # km (scale height used in your scripts)

fig, axs = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
levels_shear = np.arange(-0.001, 0.0011, 0.0002)

for ax, (decade, period) in zip(axs.flat, decades.items()):
    # --- Decadal zonal-mean anomaly ---
    ua_Δ = (ua_sulf.sel(time=period).mean("time") -
            ua_solar.sel(time=period).mean("time")).mean("lon")   # (plev, lat)
    
    # --- Convert ∂u/∂p → ∂u/∂z (m s-1 km-1) ---
    dua_dp = ua_Δ.differentiate("plev")           # m s-1 Pa-1
    dua_dz = -dua_dp * ua_Δ.plev / (H * 1000.0)
    
    # --- Altitude for axis (same as your other figs) ---
    z_km = -H * np.log(ua_Δ.plev / 101325.0)      # km

    pcm = ax.contourf(ua_Δ.lat, z_km, dua_dz,
                      levels=levels_shear, cmap="RdBu_r", extend="both")
    ax.contour(ua_Δ.lat, z_km, dua_dz, levels=[0], colors="k", linewidths=1.2)
    ax.set_title(decade)
    ax.set_xlim(-90, 90)
    ax.set_ylim(33, 15)

    # Fix for y/x labels
    row, col = divmod(list(axs.flat).index(ax), 4)
    if col == 0:
        ax.set_ylabel("Altitude (km)")
    if row == 1:
        ax.set_xlabel("Latitude")

# Colorbar
fig.subplots_adjust(right=0.92, bottom=0.11)
cbar_ax = fig.add_axes([0.94, 0.13, 0.015, 0.75])
fig.colorbar(pcm, cax=cbar_ax, label="∂u/∂z  (m s⁻¹ km⁻¹)")

fig.suptitle("Vertical shear of zonal wind anomaly (G6sulfur – G6solar)\n"
             "EP-flux proxy, by decade", fontsize=17)
plt.tight_layout(rect=[0, 0, 0.92, 0.95])
# Optional: fig.savefig("EPflux_proxy_all_decades.png", dpi=300)
plt.show()


# ---------- convert units ----------
wap_sulf  = wap_sulf  * 86400 / 100.0   #  Pa s⁻1 → hPa day⁻1  (1 hPa = 100 Pa)
wap_solar = wap_solar * 86400 / 100.0

# ---------- baseline σ for stippling (2000-14) ----------
# Fix: Change end date from 31st to 30th for 360-day calendar
base = slice("2000-01-01", "2014-12-30")
σ_nat = (wap_sulf.sel(time=base) - wap_solar.sel(time=base)).std("time").mean("lon")

# ---------- decadal slices ----------
# Using explicit start and end dates with day 30 instead of day 31 for end dates
decades = [
    ("2020-29", slice("2020-01-01", "2029-12-30")),
    ("2030-39", slice("2030-01-01", "2039-12-30")),
    ("2040-49", slice("2040-01-01", "2049-12-30")),
    ("2050-59", slice("2050-01-01", "2059-12-30")),
    ("2060-69", slice("2060-01-01", "2069-12-30")),
    ("2070-79", slice("2070-01-01", "2079-12-30")),
    ("2080-89", slice("2080-01-01", "2089-12-30")),
    ("2090-99", slice("2090-01-01", "2099-12-30")),
]

# ---------- plotting set-up ----------
fig = plt.figure(figsize=(16,12))
gs  = GridSpec(4,2, hspace=0.3, wspace=0.2, figure=fig)

levels_wap = np.arange(-0.6, 0.65, 0.05)            # hPa day-1
levels_o3  = np.arange(-4, 44, 4) * 1e-6            # mol mol-1 (ppm ≈ μmol mol-1)

cmap_wap = "BrBG_r"

for i, (label, period) in enumerate(decades):
    row, col = divmod(i,2)
    ax = fig.add_subplot(gs[row, col])

    # --- decadal means ---
    wap_Δ = (wap_sulf.sel(time=period).mean("time") -
             wap_solar.sel(time=period).mean("time")).mean("lon")     # (lev, lat)

    o3_Δ  = (o3_sulf .sel(time=period).mean("time") -
             o3_solar.sel(time=period).mean("time")).mean("lon")

    # --- stipple mask ---
    stipple = np.abs(wap_Δ) < σ_nat

    # --- altitude axis (km) ---
    # Barometric formula for standard atmosphere: h = -H * log(P/P0)
    # where H is scale height (~7.5 km), P is pressure, P0 is surface pressure (~101325 Pa)
    H = 7.5  # scale height in km
    P0 = 101325.0  # reference pressure in Pa
    z_km = -H * np.log(wap_Δ.plev / P0)

    pcm = ax.contourf(wap_Δ.lat, z_km, wap_Δ,
                      levels=levels_wap, cmap=cmap_wap, extend="both")
    # overlay ozone contours every 8 ppm (~8 × 10⁻⁶)
    cs  = ax.contour(o3_Δ.lat, z_km, o3_Δ,
                     levels=levels_o3, colors="k", linewidths=0.7)
    # Fix: Change lambda format to be compatible with matplotlib's expectations
    ax.clabel(cs, fmt=lambda v: f"{v*1e6:.0f}", fontsize=6)

    # stippling
    ax.contourf(wap_Δ.lat, z_km, stipple,
                levels=[0.5,1], hatches=["..."], colors="none")

    ax.set_ylim(33,15)
    ax.set_title(label)
    if col == 0:
        ax.set_ylabel("Altitude (km)")
    if row == 3:
        ax.set_xlabel("Latitude")

# ---------- shared colour-bar ----------
cax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
cb  = fig.colorbar(pcm, cax=cax, orientation="horizontal")
cb.set_label("Δ wap  (hPa day$^{-1}$)  —  negative = stronger upwelling")

fig.suptitle("Residual vertical velocity anomaly & ozone response\n(G6sulfur – G6solar)", fontsize=16, y=0.95)
plt.tight_layout(rect=[0, 0.08, 1, 0.93])
plt.show()