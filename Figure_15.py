import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress

DATA_DIR = Path(r"C:\Users\gabri\Downloads\All_Data_nc")
EXP_SULFUR = "G6sulfur"
EXP_SOLAR  = "G6solar"

def open_o3_both(exp):
    files = sorted(DATA_DIR.glob(f"o3_*_{exp}_*.nc"))
    if not files:
        raise FileNotFoundError(f"No o3 files found for {exp}")
    ds = xr.open_mfdataset(files, combine="by_coords")
    return ds["o3"]

def open_n2o_both(exp):
    files = sorted(DATA_DIR.glob(f"n2o_*_{exp}_*.nc"))
    if not files:
        raise FileNotFoundError(f"No n2o files found for {exp}")
    ds = xr.open_mfdataset(files, combine="by_coords")
    return ds["n2o"]

o3_sulf = open_o3_both(EXP_SULFUR)
o3_solar = open_o3_both(EXP_SOLAR)
n2o_sulf = open_n2o_both(EXP_SULFUR)
n2o_solar = open_n2o_both(EXP_SOLAR)

decades = [(y, slice(f"{y}-01-01", f"{y+9}-12-30")) for y in range(2020, 2100, 10)]

fig, axs = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
slopes = []
rvals = []
pvals = []

for ax, (y, s) in zip(axs.flat, decades):
    # Select southern high latitudes
    o3_diff = (o3_sulf - o3_solar).sel(time=s)
    n2o_diff = (n2o_sulf - n2o_solar).sel(time=s)
    # Use only latitudes <= -60
    o3_diff = o3_diff.where(o3_diff.lat <= -60, drop=True)
    n2o_diff = n2o_diff.where(n2o_diff.lat <= -60, drop=True)
    # Mean over time, lat, lon to get profile vs lev
    dO3 = o3_diff.mean(["time", "lat", "lon"])
    dN2O = n2o_diff.mean(["time", "lat", "lon"])
    x = dN2O.values.ravel()
    yv = dO3.values.ravel()
    mask = ~np.isnan(x) & ~np.isnan(yv)
    x, yv = x[mask], yv[mask]

    # For leftmost panels
    row, col = divmod(list(axs.flat).index(ax), 4)
    if col == 0:
        ax.set_ylabel("Δ O₃ (mol mol⁻¹)")
    ax.set_xlabel("Δ N₂O (mol mol⁻¹)")
    ax.set_title(f"{y}s")

    if len(x) >= 3 and np.ptp(x) > 0:
        result = linregress(x, yv)
        slope = result.slope
        r = result.rvalue
        p = result.pvalue
        slopes.append(slope)
        rvals.append(r)
        pvals.append(p)
        ax.scatter(x, yv, s=25, alpha=0.7, edgecolor="k")
        ax.plot(x, slope*x + result.intercept, c="k", lw=1.2, label=f"slope={slope:.2f}")
        ax.legend()
        ax.annotate(f"$r$={r:.2f}\n$p$={p:.2g}", (0.05, 0.85), xycoords="axes fraction")
    else:
        ax.set_title(f"{y}s\nNO FIT")
        slopes.append(np.nan)
        rvals.append(np.nan)
        pvals.append(np.nan)


fig.suptitle("Δ O₃ vs Δ N₂O (Antarctic, 60–90°S), by decade", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("O3_vs_N2O_by_decade_stats.png", dpi=300)
plt.show()

import matplotlib.pyplot as plt
valid_decades = [y for (y, _), slope in zip(decades, slopes) if slope is not None]
plt.figure()
plt.plot([y for (y, _) in decades[:len(slopes)]], slopes, marker="o", label="slope")
plt.xlabel("Decade Start")
plt.ylabel("Slope of Δ O₃ vs Δ N₂O")
plt.title("Transport-chemistry diagnostic slope by decade\n(Antarctic, 60–90°S)")
plt.savefig("O3_vs_N2O_slope_by_decade_stats.png", dpi=300)
plt.show()

print("Decade | slope | r | p")
for (y, _), slope, r, p in zip(decades, slopes, rvals, pvals):
    print(f"{y}s: {slope:.2g} (r={r:.2f}, p={p:.2g})")
