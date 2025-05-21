import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Helper: Integrate O₃ column in Dobson Units (DU) ---
def integrate_column(o3):
    """
    Integrate O3 [kg/kg] over pressure to get total column O3 in Dobson Units.
    Args:
        o3: xarray DataArray with dims including pressure ('plev' or 'lev')
    Returns:
        o3col_DU: 2D DataArray [lat, lon] or [lat] (Dobson Units)
    """
    # Find which axis is pressure
    if 'plev' in o3.dims:
        pcoord = 'plev'
        plev = o3['plev']
    elif 'lev' in o3.dims:
        pcoord = 'lev'
        plev = o3['lev']
    else:
        raise ValueError("No pressure dimension 'plev' or 'lev' found")

    # Constants
    M_air = 28.97e-3  # kg/mol
    NA = 6.02214076e23
    g = 9.80665  # m/s2

    # Compute dp in [Pa] as DataArray for broadcasting
    dp = np.abs(np.gradient(plev))
    dp = xr.DataArray(dp, coords={pcoord: plev}, dims=(pcoord,))

    # Broadcast dp to o3
    # (xarray will automatically align dims when you do o3 * dp)
    # Integrate Σ o3 * dp / g * (NA/M_air) over pressure
    o3_col_molec_per_m2 = (o3 * dp / g) * (NA / M_air)
    o3_col_DU = o3_col_molec_per_m2.sum(dim=pcoord) / 2.69e20  # DU

    # If time is present, return time-mean
    if 'time' in o3_col_DU.dims:
        o3_col_DU = o3_col_DU.mean(dim='time')

    return o3_col_DU

    # Constants
    M_air = 28.97e-3  # kg/mol
    M_O3 = 48.00e-3   # kg/mol
    NA = 6.02214076e23
    DU_per_molec_cm2 = 2.69e16  # molecules/cm^2 per DU

    # Use the pressure thickness between levels for integration (assume midpoints)
    dp = np.abs(np.gradient(plev))
    # o3: [time, plev, lat, lon] or [plev, lat, lon]
    # Need to broadcast dp
    while len(dp.shape) < len(o3.shape):
        dp = dp[:, None]

    # Integrate: Σ o3 * dp
    # O3 mass mixing ratio to number of molecules
    g = 9.80665
    o3_col_molec_per_m2 = (o3 * dp / g) * (NA / M_air)
    # Convert to DU: 1 DU = 2.69e20 molec/m²
    o3_col_DU = (o3_col_molec_per_m2.sum(dim=pcoord)) / 2.69e20

    # If time is present, return time-mean
    if 'time' in o3_col_DU.dims:
        o3_col_DU = o3_col_DU.mean(dim='time')

    return o3_col_DU

# --- Main script ---
def main():
    data_dir = Path(r"C:\Users\gabri\Downloads\All_Data_nc")
    out_dir = data_dir / "output_figs"
    out_dir.mkdir(exist_ok=True)

    # --- Load O3 for both experiments (edit variable/experiment if needed) ---
    print("Loading O3 (G6sulfur)...")
    o3_sulf = xr.open_mfdataset(
        str(data_dir / "o3_Amon_UKESM1-0-LL_G6sulfur*.nc"),
        combine='by_coords'
    )['o3']

    print("Loading O3 (G6solar)...")
    o3_sol = xr.open_mfdataset(
        str(data_dir / "o3_Amon_UKESM1-0-LL_G6solar*.nc"),
        combine='by_coords'
    )['o3']

    # --- Integrate to total column O3 (DU) ---
    print("Integrating O3 columns...")
    o3col_sulf = integrate_column(o3_sulf)
    o3col_sol  = integrate_column(o3_sol)
    delta_o3col = o3col_sulf - o3col_sol

    print("Shapes (sulf/sol/delta):", o3col_sulf.shape, o3col_sol.shape, delta_o3col.shape)

    # --- Plot the map (ΔO₃, Dobson Units) ---
    print("Plotting ΔO3 column (DU)...")
    plt.figure(figsize=(12,5))
    if 'lat' in delta_o3col.dims and 'lon' in delta_o3col.dims:
        pcm = plt.pcolormesh(delta_o3col['lon'], delta_o3col['lat'], delta_o3col, cmap='RdBu_r')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
    elif 'lat' in delta_o3col.dims:
        plt.plot(delta_o3col['lat'], delta_o3col, label='ΔO3col (DU)')
        plt.xlabel('Latitude')
    plt.colorbar(pcm, label='ΔO₃ (DU)')
    plt.title('ΔO₃ Column (G6sulfur - G6solar) [Dobson Units]')
    plt.tight_layout()
    plt.savefig(out_dir / "figure9_delta_o3col_map.png", dpi=200)
    plt.show()
    print("Saved:", out_dir / "figure9_delta_o3col_map.png")

if __name__ == "__main__":
    main()
