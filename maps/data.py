import pandas as pd
import numpy as np
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point, MultiPolygon
from shapely.prepared import prep
from scipy.interpolate import griddata
import os
from glob import glob
import re
import pickle
import gc

base_folder = './model_outputs/NewVersion/CorrectedFormula'
window_sizes = ['WS1', 'WS3', 'WS5']
warming_levels = [1.5, 2.0, 3.0]
grid_resolution = 1.0
OUTPUT_FILE = 'averaged_gridsCorrectedAllData_Normal_Final_CorrectedFormula.pkl'
TEMP_DIR = './temp_binary_chunks'

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

metrics = {
    'diff_duration_mean': 'Δduration (days)',
    'diff_frequency': 'Δfrequency (count)',
    'diff_intensity_mean': 'Δintensity (units)',
    'diff_drought_days': 'Δdrought days (days)',
    'ratio_duration_mean': 'Δduration (%)',
    'ratio_frequency': 'Δfrequency (%)',
    'ratio_intensity_mean': 'Δintensity (%)',
    'ratio_drought_days': 'Δdrought days (%)'
}

lon_grid = np.arange(-180, 180 + grid_resolution, grid_resolution)
lat_grid = np.arange(-90, 90 + grid_resolution, grid_resolution)
grid_lon, grid_lat = np.meshgrid(lon_grid, lat_grid)

def get_final_mask():
    land_shp = shpreader.natural_earth(resolution='110m', category='physical', name='land')
    try:
        land_union = MultiPolygon([g for g in shpreader.Reader(land_shp).geometries() if g.geom_type in ['Polygon', 'MultiPolygon']])
        prepared_land = prep(land_union)
        flat_lon, flat_lat = grid_lon.flatten(), grid_lat.flatten()
        l_mask = np.array([prepared_land.contains(Point(lo, la)) for lo, la in zip(flat_lon, flat_lat)]).reshape(grid_lon.shape)
    except:
        l_mask = np.full(grid_lon.shape, True)

    show_mask = l_mask & (grid_lat >= -60)
    return ~show_mask

final_mask = get_final_mask()

def process_data_optimized():
    temp_registry = []

    for ws in window_sizes:
        ws_path = os.path.join(base_folder, ws)
        if not os.path.isdir(ws_path):
            continue

        file_list = glob(os.path.join(ws_path, "*.pkl"))
        grouped_files = {}
        for fp in file_list:
            match = re.match(r"IDF_metrics_(.+?)_WS\d+(_WL\d\.\d_chunk\d+)?\.pkl", os.path.basename(fp))
            if match:
                model_name = match.group(1)
                grouped_files.setdefault(model_name, []).append(fp)

        for wl in warming_levels:
            print(f"\n--- Processing {ws} | WL {wl} ---")

            accumulators = {m: np.full_like(grid_lon, np.nan, dtype=np.float32) for m in metrics}
            counts = {m: np.zeros_like(grid_lon, dtype=np.int32) for m in metrics}

            for model_name, files in grouped_files.items():
                model_df = pd.DataFrame()
                for f in files:
                    with open(f, 'rb') as fb:
                        chunk = pickle.load(fb)
                        if isinstance(chunk, pd.DataFrame):
                            chunk = chunk[chunk['warming_level'] == wl]
                            model_df = pd.concat([model_df, chunk], ignore_index=True)
                    del chunk

                if model_df.empty:
                    continue

                model_df['lon'] = model_df['lon'].apply(lambda x: x - 360 if x > 180 else x)
                ratio_cols = [c for c in metrics.keys() if 'ratio' in c and c in model_df.columns]
                for c in ratio_cols:
                    model_df[c] = (model_df[c] - 1) * 100

                for m_name in metrics.keys():
                    if m_name not in model_df.columns:
                        continue

                    sub = model_df[['lon', 'lat', m_name]].dropna()
                    if len(sub) < 4:
                        continue

                    interp = griddata(
                        sub[['lon', 'lat']].values,
                        sub[m_name].values,
                        (grid_lon, grid_lat),
                        method='linear',
                        fill_value=np.nan
                    ).astype(np.float32)

                    interp[final_mask] = np.nan

                    v_mask = ~np.isnan(interp)
                    curr_acc = accumulators[m_name]
                    curr_cnt = counts[m_name]

                    upd = v_mask & ~np.isnan(curr_acc)
                    new = v_mask & np.isnan(curr_acc)

                    curr_acc[upd] = (curr_acc[upd] * curr_cnt[upd] + interp[upd]) / (curr_cnt[upd] + 1)
                    curr_cnt[upd] += 1
                    curr_acc[new] = interp[new]
                    curr_cnt[new] = 1

                del model_df
                gc.collect()

            for m_name in metrics.keys():
                chunk_key = (ws, wl, m_name)
                temp_fn = os.path.join(TEMP_DIR, f"{ws}_{wl}_{m_name}.npy")
                np.save(temp_fn, accumulators[m_name])
                temp_registry.append((chunk_key, temp_fn))

            del accumulators, counts
            gc.collect()

    print("\nFinalizing output file...")

    final_data = {
        'grid_lon': grid_lon,
        'grid_lat': grid_lat,
        'metrics': metrics,
        'window_sizes': window_sizes,
        'warming_levels': warming_levels,
        'averaged_grids': {}
    }

    for key, path in temp_registry:
        final_data['averaged_grids'][key] = np.load(path)
        os.remove(path)

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(final_data, f)

    os.rmdir(TEMP_DIR)
    print("Done.")

if __name__ == "__main__":
    process_data_optimized()