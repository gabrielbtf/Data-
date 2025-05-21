import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def print_info(title, arr):
    arr_np = arr.values if hasattr(arr, 'values') else arr
    print(f"{title}: shape={arr.shape}, dtype={arr.dtype}, NaNs={np.isnan(arr_np).sum()}, min={np.nanmin(arr_np):.3e}, max={np.nanmax(arr_np):.3e}")

def zonal_mean_lat_plev(da):
    # Mean over lon, collapse to (time, lev/plev, lat)
    if 'lon' in da.dims:
        return da.mean(dim='lon')
    return da

def delta_field(sulf, solar):
    # G6sulfur minus G6solar
    return sulf - solar

def main():
    data_dir = Path(r"C:\Users\gabri\Downloads\All_Data_nc")
    output_dir = data_dir / "output_plots"
    output_dir.mkdir(exist_ok=True)

    # ---- Load tntr ----
    print("Loading tntr...")
    tntr_sulf = xr.open_mfdataset(sorted(data_dir.glob("tntr_CFmon_UKESM1-0-LL_G6sulfur*.nc")), combine="by_coords")['tntr']
    tntr_sol  = xr.open_mfdataset(sorted(data_dir.glob("tntr_CFmon_UKESM1-0-LL_G6solar*.nc")),  combine="by_coords")['tntr']

    # ---- Load od550aer ----
    print("Loading od550aer...")
    od_sulf = xr.open_mfdataset(sorted(data_dir.glob("od550aer_AERmon_UKESM1-0-LL_G6sulfur*.nc")), combine="by_coords")['od550aer']
    od_sol  = xr.open_mfdataset(sorted(data_dir.glob("od550aer_AERmon_UKESM1-0-LL_G6solar*.nc")),  combine="by_coords")['od550aer']

    # ---- Select wavelength=550nm ----
    if 'wavelength' in od_sulf.dims:
        od_sulf = od_sulf.sel(wavelength=550, method="nearest")
        od_sol  = od_sol.sel(wavelength=550, method="nearest")

    # ---- tntr: use lev (pressure, hPa), od550aer: has no vertical, so broadcast ----
    print(f"tntr_sulf dims: {tntr_sulf.dims}")
    print(f"od550aer_sulf dims: {od_sulf.dims}")
    # Try to find 'lev' or 'plev'
    vert_name = 'lev' if 'lev' in tntr_sulf.dims else 'plev'
    tntr_sulf_zm = zonal_mean_lat_plev(tntr_sulf)
    tntr_sol_zm  = zonal_mean_lat_plev(tntr_sol)
    # For od550aer, no vertical, so need to interpolate or broadcast to lev
    od_sulf_zm = od_sulf.mean(dim='lon')
    od_sol_zm  = od_sol.mean(dim='lon')
    # Now average over time for simplicity (or use seasonal mean as you wish)
    tntr_sulf_zm = tntr_sulf_zm.mean(dim='time')
    tntr_sol_zm  = tntr_sol_zm.mean(dim='time')
    od_sulf_zm   = od_sulf_zm.mean(dim='time')
    od_sol_zm    = od_sol_zm.mean(dim='time')

    print_info("tntr_sulf_zm", tntr_sulf_zm)
    print_info("od_sulf_zm", od_sulf_zm)

    # Δtntr and Δod550aer
    delta_tntr = delta_field(tntr_sulf_zm, tntr_sol_zm)
    delta_od = delta_field(od_sulf_zm, od_sol_zm)
    # For od550aer: if no vertical, repeat for each level
    if vert_name not in od_sulf_zm.dims:
        print("Broadcasting od550aer to match vertical levels of tntr...")
        od_lat = od_sulf_zm['lat']
        od_vals = delta_od.values  # shape (lat,)
        lev = tntr_sulf_zm[vert_name]
        od_broadcast = np.tile(od_vals, (len(lev), 1))  # (lev, lat)
        delta_od = xr.DataArray(od_broadcast, dims=(vert_name, 'lat'), coords={vert_name: lev, 'lat': od_lat})
    else:
        # Interpolate od550aer onto tntr vertical grid if needed
        delta_od = delta_od.interp({vert_name: tntr_sulf_zm[vert_name]})

    # --- Diagnostics ---
    print_info("delta_tntr", delta_tntr)
    print_info("delta_od", delta_od)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    LAT, LEV = np.meshgrid(delta_tntr['lat'], delta_tntr[vert_name])
    # Δtntr
    c1 = axes[0].contourf(
        delta_tntr['lat'], delta_tntr[vert_name], delta_tntr.T, levels=21, cmap='RdBu_r', extend='both'
    )
    axes[0].set_title(r'$\Delta$ Heating Rate (tntr) [K/day]')
    axes[0].set_xlabel('Latitude')
    axes[0].set_ylabel('Pressure (hPa)')
    axes[0].invert_yaxis()
    plt.colorbar(c1, ax=axes[0])
    # Δod550aer
    c2 = axes[1].contourf(
        delta_od['lat'], delta_od[vert_name], delta_od.T, levels=21, cmap='YlOrBr', extend='both'
    )
    axes[1].set_title(r'$\Delta$ AOD$_{550}$')
    axes[1].set_xlabel('Latitude')
    axes[1].invert_yaxis()
    plt.colorbar(c2, ax=axes[1])
    plt.tight_layout()
    figname = output_dir / "Figure8_Heating_AOD.png"
    plt.savefig(figname, dpi=200)
    print(f"Figure saved: {figname}")

if __name__ == "__main__":
    main()
