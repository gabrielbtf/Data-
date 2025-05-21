import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sulfur_paths = [
    "/Users/gabrielfitzpatrick/Downloads/o3/o3_Amon_UKESM1-0-LL_G6sulfur_r1i1p1f2_gn_202001-204912.nc",
    "/Users/gabrielfitzpatrick/Downloads/o3/o3_Amon_UKESM1-0-LL_G6sulfur_r1i1p1f2_gn_205001-210012.nc"
]
solar_paths = [
    "/Users/gabrielfitzpatrick/Downloads/o3/o3_Amon_UKESM1-0-LL_G6solar_r1i1p1f2_gn_202001-204912.nc",
    "/Users/gabrielfitzpatrick/Downloads/o3/o3_Amon_UKESM1-0-LL_G6solar_r1i1p1f2_gn_205001-210012.nc"
]

pressure_levels_pa = {
    "20hpa": 2000,
    "50hpa": 5000,
    "70hpa": 7000,
    "100hpa": 10000,
}

decades = [(2030, 2039), (2040, 2049), (2050, 2059),
           (2060, 2069), (2070, 2079), (2080, 2089),
           (2090, 2099), (2100, 2109)]

def slice_at_plev(o3, target_pa):
    plevs = o3['plev'].values
    nearest = plevs[np.argmin(np.abs(plevs - target_pa))]
    return o3.sel(plev=nearest)

# === Load datasets ===
print("📦 Loading ozone datasets...")
ds_sulfur = xr.open_mfdataset(sulfur_paths, combine='by_coords')
ds_solar = xr.open_mfdataset(solar_paths, combine='by_coords')
o3_sulfur = ds_sulfur['o3']
o3_solar = ds_solar['o3']

# === Loop over pressure levels ===
for label, pa in pressure_levels_pa.items():
    print(f"\n🔬 Processing pressure: {label.upper()} ({pa} Pa)")

    # Step 1: Determine consistent colorbar scale
    print("📊 Scanning for global vmin/vmax across all decades...")
    diffs = []
    for start, end in decades:
        s_dec = o3_sulfur.sel(time=slice(f"{start}-01-01", f"{end}-12-30")).mean("time")
        g_dec = o3_solar.sel(time=slice(f"{start}-01-01", f"{end}-12-30")).mean("time")
        diff = slice_at_plev(s_dec, pa) - slice_at_plev(g_dec, pa)
        diffs.append(diff)
    all_diffs = xr.concat(diffs, dim="time")  # stack them for global min/max
    vmin = float(all_diffs.min().values)
    vmax = float(all_diffs.max().values)
    # Round and symmetrize the scale
    abs_limit = max(abs(vmin), abs(vmax))
    limit = np.ceil(abs_limit * 1e8) / 1e8
    vmin, vmax = -limit, limit
    print(f"✅ Colorbar scale fixed: vmin={vmin:.1e}, vmax={vmax:.1e}")

    fig, axs = plt.subplots(2, 4, figsize=(24, 7.5), subplot_kw={'projection': ccrs.Robinson()})
    plt.subplots_adjust(
        top=0.90,
        bottom=0.14,
        left=0.04,
        right=0.98,
        wspace=0.13,
        hspace=0.28
    )

    pcm = None
    for idx, (start, end) in enumerate(decades):
        decade_label = f"{start}s"
        ax = axs.flat[idx]
        s_dec = o3_sulfur.sel(time=slice(f"{start}-01-01", f"{end}-12-30")).mean("time")
        g_dec = o3_solar.sel(time=slice(f"{start}-01-01", f"{end}-12-30")).mean("time")
        diff = slice_at_plev(s_dec, pa) - slice_at_plev(g_dec, pa)

        pcm = ax.contourf(
            diff['lon'], diff['lat'], diff,
            transform=ccrs.PlateCarree(),
            cmap='RdBu_r',
            levels=np.linspace(vmin, vmax, 21),
            extend='both'
        )
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.set_global()
        ax.set_title(decade_label, fontsize=13)
        ax.gridlines(draw_labels=False, linewidth=0.2, linestyle='--')

    fig.suptitle(f"Ozone Difference (G6sulfur − G6solar) at {label.upper()} Across Decades", fontsize=18, y=0.965)

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.035])  # [left, bottom, width, height]
    cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal', label="O₃ anomaly (kg/kg)")
    cbar.ax.tick_params(labelsize=12)

    # Save as PDF
    pdf_path = f"/Users/gabrielfitzpatrick/Downloads/o3/o3_G6diff_panel_{label}.pdf"
    plt.savefig(pdf_path, format="pdf")
    plt.close()
    print(f" Saved panel plot: {pdf_path}")

print("\n✅ All ozone PDF panels with fixed colorbars saved.")
