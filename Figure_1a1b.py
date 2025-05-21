import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

data_dir = "C:/Users/gabri/Downloads/All_Data_nc"

so2_sulf_files = f"{data_dir}/so2_AERmon_UKESM1-0-LL_G6sulfur*.nc"
aod_sulf_files = f"{data_dir}/od550aer_AERmon_UKESM1-0-LL_G6sulfur*.nc"

def global_mean_2D(field):
    weights = np.cos(np.deg2rad(field.lat))
    return field.weighted(weights).mean(("lat", "lon"))

so2_sulf = xr.open_mfdataset(so2_sulf_files, parallel=False, combine="by_coords")
aod_sulf = xr.open_mfdataset(aod_sulf_files, parallel=False, combine="by_coords")

alt_mask = (so2_sulf.lev >= 17_000) & (so2_sulf.lev <= 33_000)
so2_strat = so2_sulf["so2"].sel(lev=alt_mask).mean("lev")  # shape: (time, lat, lon)

# ---------- Global means ----------
so2_gm = global_mean_2D(so2_strat)  # (time,)
aod_gm = global_mean_2D(aod_sulf["od550aer"])  # Make sure it's also a global mean

# ---------- Annual means with 'YE' ----------
so2_ann = so2_gm.resample(time="1YE").mean()
aod_ann = aod_gm.resample(time="1YE").mean()

# ---------debug----------
print(f"so2_ann dimensions: {so2_ann.dims}")
print(f"aod_ann dimensions: {aod_ann.dims}")

so2_ann, aod_ann = xr.align(so2_ann, aod_ann, join='inner')

so2_df = so2_ann.to_pandas()
aod_df = aod_ann.to_pandas()

# ---------- Check if so2_df and aod_df are Series (1D) or DataFrame (2D+) ----------
if hasattr(so2_df, 'name'):  # It's a Series
    so2_values = so2_df.values
else:  # It's a DataFrame or higher
    so2_values = so2_df.values.flatten()

if hasattr(aod_df, 'name'):  # It's a Series
    aod_values = aod_df.values
else:  # It's a DataFrame or higher
    aod_values = aod_df.values.flatten()

import pandas as pd
df = pd.DataFrame({'so2': so2_values, 'aod': aod_values})
df = df.dropna()
x = df['so2'].values
y = df['aod'].values

# ---------- Linear fit ----------
slope, intercept, r, p, stderr = linregress(x, y)

# ---------- Decadal stats ----------
def decadal_stats(series):
    # Get years from the time index
    years = np.array([t.year for t in series.index])
    decades = np.arange((years[0]//10)*10, years[-1]+1, 10)
    means, sigs, mids = [], [], []
    
    for d in decades[:-1]:
        sel = (years >= d) & (years < d+10)
        vals = series.values[sel]
        if np.any(~np.isnan(vals)):
            means.append(np.nanmean(vals))
            sigs.append(np.nanstd(vals))
            mids.append(d+5)
    return np.array(mids), np.array(means), np.array(sigs)

so2_series = pd.Series(so2_values, index=so2_ann.time.values)
aod_series = pd.Series(aod_values, index=aod_ann.time.values)

mid, so2m, so2sig = decadal_stats(so2_series)
_, aodm, aodsig = decadal_stats(aod_series)

fig, axs = plt.subplots(2, 1, figsize=(7, 9), gridspec_kw={"height_ratios": [2, 3]})

# (A) Scatter with fit
axs[0].scatter(x, y, alpha=0.6, label="Annual means", color="tab:blue")
axs[0].plot(x, intercept + slope*x, color="k",
            label=f"Fit: slope={slope:.3e}, $R^2$={r**2:.2f}")
axs[0].set_xlabel("Global-mean SO$_2$ (17–33 km) [kg kg$^{-1}$]")
axs[0].set_ylabel("Global-mean AOD$_{550}$")
axs[0].set_title("SO$_2$ vs AOD$_{550}$ – G6sulfur")
axs[0].legend()

# (B) Time series with ±1 σ
axs[1].plot(mid, so2m, marker="o", color="tab:orange", label="SO$_2$")
axs[1].fill_between(mid, so2m-so2sig, so2m+so2sig, alpha=0.2, color="tab:orange")
axs[1].set_ylabel("SO$_2$ (kg kg$^{-1}$)", color="tab:orange")
axs[1].tick_params(axis="y", labelcolor="tab:orange")

ax2 = axs[1].twinx()
ax2.plot(mid, aodm, marker="s", color="tab:blue", label="AOD$_{550}$")
ax2.fill_between(mid, aodm-aodsig, aodm+aodsig, alpha=0.2, color="tab:blue")
ax2.set_ylabel("AOD$_{550}$", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")

axs[1].set_xlabel("Year (mid-decade)")
axs[1].set_title("Decadal means ±1 σ (inter-annual)")

# Combine legends for both y-axes
lines, labels = axs[1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axs[1].legend(lines + lines2, labels + labels2, loc="upper left")

plt.tight_layout()
plt.show()
