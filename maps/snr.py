import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
import os
from glob import glob
import re
import pickle
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point, MultiPolygon
from shapely.prepared import prep
import gc

base_folder = './model_outputs/NewVersion/CorrectedFormula'
processed_data_file = 'processed_snr_data_Final_CorrectedFormula.pkl'
TEMP_DIR = './temp_snr_storage'

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

window_sizes = ['WS1', 'WS3', 'WS5']
warming_levels = [1.5, 2.0, 3.0]
grid_resolution = 1.0

metrics_config = {
    'diff_duration_mean': 'ΔDuration', 'diff_frequency': 'ΔFrequency',
    'diff_intensity_mean': 'ΔPeak', 'diff_drought_days': 'ΔDrought Days',
    'ratio_duration_mean': 'ΔDuration %', 'ratio_frequency': 'ΔFrequency %',
    'ratio_intensity_mean': 'ΔPeak %', 'ratio_drought_days': 'ΔDrought Days %'
}

lon_grid = np.arange(-180, 180 + grid_resolution, grid_resolution)
lat_grid = np.arange(-90, 90 + grid_resolution, grid_resolution)
grid_lon, grid_lat = np.meshgrid(lon_grid, lat_grid)

def get_final_mask():
    print("Pre-calculating land mask...")
    land_shp = shpreader.natural_earth(resolution='110m', category='physical', name='land')
    try:
        land_union = MultiPolygon([g for g in shpreader.Reader(land_shp).geometries() if g.geom_type in ['Polygon', 'MultiPolygon']])
        prep_land = prep(land_union)
        mask = np.array([prep_land.contains(Point(lo, la)) for lo, la in zip(grid_lon.ravel(), grid_lat.ravel())])
        mask = mask.reshape(grid_lon.shape)
    except:
        mask = np.full(grid_lon.shape, True)
    return ~(mask & (grid_lat >= -60))

final_mask = get_final_mask()

def stream_model_data(files, wl):
    lons, lats = [], []
    vals = {m: [] for m in metrics_config.keys()}

    for f in files:
        with open(f, 'rb') as fb:
            df = pickle.load(fb)
            if not isinstance(df, pd.DataFrame):
                continue

            df = df[df['warming_level'] == wl]
            if df.empty:
                continue

            df['lon'] = np.where(df['lon'] > 180, df['lon'] - 360, df['lon'])

            for c in ['ratio_duration_mean', 'ratio_frequency', 'ratio_intensity_mean', 'ratio_drought_days']:
                if c in df.columns:
                    df[c] = (df[c] - 1) * 100

            lons.append(df['lon'].values)
            lats.append(df['lat'].values)
            for m in metrics_config.keys():
                if m in df.columns:
                    vals[m].append(df[m].values)

    if not lons:
        return None, None

    final_pts = np.column_stack((np.concatenate(lons), np.concatenate(lats))).astype(np.float64)
    final_vals = {m: np.concatenate(v).astype(np.float64) for m, v in vals.items() if v}
    return final_pts, final_vals

def process_snr_max_efficiency():
    grid_registry = []

    for ws in window_sizes:
        ws_path = os.path.join(base_folder, ws)
        if not os.path.isdir(ws_path):
            continue

        file_list = glob(os.path.join(ws_path, "*.pkl"))
        grouped = {}
        for fp in file_list:
            match = re.match(r"IDF_metrics_(.+?)_WS\d+", os.path.basename(fp))
            if match:
                grouped.setdefault(match.group(1), []).append(fp)

        for model_name in grouped:
            for wl in warming_levels:
                print(f"Working: {ws} | WL {wl} | {model_name}")

                pts, val_dict = stream_model_data(grouped[model_name], wl)
                if pts is None or pts.shape[0] < 4:
                    continue

                try:
                    tri = Delaunay(pts)

                    for m_name, values in val_dict.items():
                        interpolator = LinearNDInterpolator(tri, values)
                        grid = interpolator(grid_lon, grid_lat)

                        grid = grid.astype(np.float32)
                        grid[final_mask] = np.nan

                        safe_model = model_name.replace(" ", "_")
                        t_path = os.path.join(TEMP_DIR, f"{ws}_{wl}_{m_name}_{safe_model}.npy")
                        np.save(t_path, grid)
                        grid_registry.append({'key': (ws, wl, m_name), 'mod': model_name, 'p': t_path})

                        del grid, interpolator

                except Exception as e:
                    print(f"Error for {model_name}: {e}")

                finally:
                    if 'tri' in locals():
                        del tri
                    del pts, val_dict
                    gc.collect()

    print("\nMerging all binary chunks into final pickle...")
    all_model_grids = {}

    for entry in grid_registry:
        key = entry['key']
        if key not in all_model_grids:
            all_model_grids[key] = {}

        all_model_grids[key][entry['mod']] = np.load(entry['p'])
        os.remove(entry['p'])

    with open(processed_data_file, 'wb') as f:
        pickle.dump({
            "all_model_grids": all_model_grids,
            "grid_lon": grid_lon,
            "grid_lat": grid_lat,
            "final_mask": final_mask,
            "metrics": metrics_config
        }, f)

    if os.path.exists(TEMP_DIR):
        os.rmdir(TEMP_DIR)

    print(f"Success! Processed data saved to: {processed_data_file}")

if __name__ == "__main__":
    process_snr_max_efficiency()