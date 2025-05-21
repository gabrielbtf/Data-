import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(r"C:\Users\gabri\Downloads\All_Data_nc")
EXP_SULFUR = "G6sulfur"
EXP_SOLAR  = "G6solar"
H_SCALE_KM = 7.5

def open_tntr_both(exp):
    files = sorted(DATA_DIR.glob(f"tntr_*_{exp}_*.nc"))
    if not files:
        raise FileNotFoundError(f"No files found for {exp}")
    ds = xr.open_mfdataset(files, combine="by_coords")
    return ds["tntr"]

# Load both experiments and concatenate along time
tntr_sulf = open_tntr_both(EXP_SULFUR)
tntr_solar = open_tntr_both(EXP_SOLAR)

# Define decades from 2020s through 2090s (CMIP6 360-day calendar: use day 30)
decades = [(y, slice(f"{y}-01-01", f"{y+9}-12-30")) for y in range(2020, 2100, 10)]

fig, ax = plt.subplots(figsize=(5, 7))

colors = plt.cm.viridis(np.linspace(0, 1, len(decades)))

for (y, s), c in zip(decades, colors):
    dtntr = (
        (tntr_sulf.sel(time=s).mean("time") - tntr_solar.sel(time=s).mean("time"))
        * 86400  # K s⁻¹ → K day⁻¹
    ).mean(["lat", "lon"])
    lev = dtntr.lev
    if lev.max() > 2000:
        z_km = -H_SCALE_KM * np.log(lev / 101325.0)
    else:
        z_km = lev / 1000.0
    ax.plot(dtntr, z_km, label=f"{y}s", color=c)

ax.axvline(0, ls="--", c="k", lw=0.8)
ax.set_xlabel("Δ tntr  (K day$^{-1}$)")
ax.set_ylabel("Altitude (km)")
ax.set_ylim(33, 20)             # Focus only on 20–33 km (as before)
ax.set_xlim(-0.2, 0.2)          # X-axis shrunk to ±0.2 K/day (fills out the graph)
ax.set_title("Global-mean radiative-heating anomaly by decade\n(G6sulfur – G6solar)")
ax.legend(title="Decade")
fig.tight_layout()
fig.savefig("heating_profile_by_decade.png", dpi=300)
plt.show()
print("Saved heating_profile_by_decade.png")
