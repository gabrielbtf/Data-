import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# === SETTINGS ===
folder = "/Users/gabrielfitzpatrick/Downloads/o3"
solar_paths = [
    f"{folder}/o3_Amon_UKESM1-0-LL_G6solar_r1i1p1f2_gn_202001-204912.nc",
    f"{folder}/o3_Amon_UKESM1-0-LL_G6solar_r1i1p1f2_gn_205001-210012.nc"
]
sulfur_paths = [
    f"{folder}/o3_Amon_UKESM1-0-LL_G6sulfur_r1i1p1f2_gn_202001-204912.nc",
    f"{folder}/o3_Amon_UKESM1-0-LL_G6sulfur_r1i1p1f2_gn_205001-210012.nc"
]

decades = {
    "2030s": ("2030-01", "2039-12"),
    "2040s": ("2040-01", "2049-12"),
    "2050s": ("2050-01", "2059-12"),
    "2060s": ("2060-01", "2069-12"),
    "2070s": ("2070-01", "2079-12"),
    "2080s": ("2080-01", "2089-12"),
    "2090s": ("2090-01", "2099-12"),
    "2100s": ("2100-01", "2109-12"),
}

def load_and_prep(paths):
    ds = xr.open_mfdataset(paths, combine='by_coords')
    if not np.issubdtype(ds.time.dtype, np.datetime64):
        ds['time'] = xr.cftime_range(start="2020-01", periods=ds.sizes["time"], freq="MS")
    o3 = ds["o3"]
    lat = ds["lat"]
    plev = ds["plev"] / 100  # Convert Pa to hPa
    plev.attrs["units"] = "hPa"
    return o3, lat, plev

def get_anomalies(o3, baseline, decades):
    results = []
    for decade, (start, end) in decades.items():
        o3_decade = o3.sel(time=slice(start, end)).mean(dim=["time", "lon"])
        anomaly = o3_decade - baseline
        results.append(anomaly)
    return results

# === Function to make one panel plot per scenario ===
def make_panel_pdf(o3, lat, plev, baseline, scenario_name, outfolder, levels=np.linspace(-1.2e-6, 1.2e-6, 21)):
    anomalies = get_anomalies(o3, baseline, decades)
    fig, axs = plt.subplots(2, 4, figsize=(22, 11), sharex=True, sharey=True)  # More vertical room!

    plt.subplots_adjust(
        top=0.93,    # More space for main title
        bottom=0.10, # More room below for colorbar
        left=0.06,
        right=0.98,
        wspace=0.13,
        hspace=0.38   # Extra vertical space between subplot rows
    )

    for idx, anomaly in enumerate(anomalies):
        row, col = divmod(idx, 4)
        ax = axs[row, col]
        pcm = ax.contourf(
            lat, plev, anomaly.transpose("plev", "lat"),
            levels=levels, cmap="RdBu_r", extend="both"
        )
        ax.set_yscale("log")
        ax.set_ylim(1000, 1)
        ax.set_title(f"{scenario_name} {list(decades.keys())[idx]}", fontsize=14, pad=12)  # pad pushes it down
        if col == 0:
            ax.set_ylabel("Pressure (hPa)")
        else:
            ax.set_ylabel("")
        if row == 1:
            ax.set_xlabel("Latitude")
        else:
            ax.set_xlabel("")

    # Main title: as high as possible, but not cut off
    fig.suptitle(f"Ozone Zonal Mean Anomalies by Decade ({scenario_name})", fontsize=21, y=0.99)

    # Colorbar: way lower, below "Latitude"
    cbar_ax = fig.add_axes([0.18, 0.025, 0.65, 0.03])  # Lower (bottom=0.025)
    cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal', label="O₃ anomaly (kg/kg)")
    cbar.ax.tick_params(labelsize=13)

    pdfname = os.path.join(outfolder, f"zonmean_o3_anomaly_panel_{scenario_name}.pdf")
    plt.savefig(pdfname, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"✅ Saved {pdfname}")
# === Make both PDFs ===
print("📦 Loading data and generating panel plots...")

# G6solar
o3_solar, lat, plev = load_and_prep(solar_paths)
o3_baseline_solar = o3_solar.sel(time=slice("2020-01", "2029-12")).mean(dim=["time", "lon"])
make_panel_pdf(o3_solar, lat, plev, o3_baseline_solar, "G6solar", folder)

# G6sulfur
o3_sulfur, lat, plev = load_and_prep(sulfur_paths)
o3_baseline_sulfur = o3_sulfur.sel(time=slice("2020-01", "2029-12")).mean(dim=["time", "lon"])
make_panel_pdf(o3_sulfur, lat, plev, o3_baseline_sulfur, "G6sulfur", folder)

print("✅ All PDFs saved.")
