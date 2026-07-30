import pandas as pd
import numpy as np
import os
import glob
import re
import pickle
import gc
from scipy.interpolate import griddata
from pathlib import Path

era5_folder = './ERA5_outputs/CorrectedFormula'
cmip6_folder = "../historical_outputs/CorrectedFormula"
output_plot_folder = './plots/CorrectedFormula'
os.makedirs(output_plot_folder, exist_ok=True)

metrics = ['frequency', 'duration_mean', 'intensity_mean']
grid_res = 1.0

lat_steps = np.arange(-90, 90 + 10, 10)

def get_model_groups(folder, ws=1):
    files = glob.glob(os.path.join(folder, f"IDF_metrics_*.pkl"))
    groups = {}
    for f in files:
        m = re.search(r"IDF_metrics_(.+?)_WS", os.path.basename(f))
        if m:
            groups.setdefault(m.group(1), []).append(f)
    return groups

def process_and_stream_csv(name, model_groups, is_era5=False):
    output_path = os.path.join(output_plot_folder, f"{name}_gridded_metrics.csv")

    pd.DataFrame(columns=['lon', 'lat'] + metrics).to_csv(output_path, index=False)

    for b in range(len(lat_steps) - 1):
        lat_min, lat_max = lat_steps[b], lat_steps[b + 1]
        print(f"Processing Band: {lat_min}° to {lat_max}°")

        lons = np.arange(-180, 180 + grid_res, grid_res)
        lats = np.arange(lat_min, lat_max + (grid_res if b == len(lat_steps) - 2 else 0), grid_res)
        gv_lon, gv_lat = np.meshgrid(lons, lats)

        band_sums = {m: np.zeros(gv_lon.shape, dtype=np.float32) for m in metrics}
        band_counts = {m: np.zeros(gv_lon.shape, dtype=np.int32) for m in metrics}

        for model_name, paths in model_groups.items():
            temp_dfs = []
            for p in paths:
                with open(p, 'rb') as f:
                    chunk = pickle.load(f)
                    mask = (chunk['lat'] >= lat_min - 2) & (chunk['lat'] <= lat_max + 2)
                    temp_dfs.append(chunk[mask])

            if not temp_dfs:
                continue

            df = pd.concat(temp_dfs, ignore_index=True)
            df['lon'] = df['lon'].apply(lambda x: x - 360 if x > 180 else x)

            pts = df[['lon', 'lat']].values

            for m in metrics:
                interp = griddata(pts, df[m].values, (gv_lon, gv_lat), method='linear')
                valid = ~np.isnan(interp)
                band_sums[m][valid] += interp[valid].astype(np.float32)
                band_counts[m][valid] += 1

            del df, temp_dfs
            gc.collect()

        band_df = pd.DataFrame({
            'lon': gv_lon.ravel(),
            'lat': gv_lat.ravel()
        })

        for m in metrics:
            res = np.divide(
                band_sums[m],
                band_counts[m],
                where=band_counts[m] > 0,
                out=np.full(gv_lon.shape, np.nan)
            )
            band_df[m] = res.ravel()

        band_df.to_csv(output_path, mode='a', header=False, index=False)

        del band_df, band_sums, band_counts
        gc.collect()

era5_files = glob.glob(os.path.join(era5_folder, "IDF_metrics_ERA5_data_WS1*.pkl"))

if era5_files:
    print("Streaming ERA5 to CSV...")
    process_and_stream_csv("ERA5", {"ERA5": era5_files}, is_era5=True)

print(cmip6_folder)

cmip6_groups = get_model_groups(cmip6_folder)

if cmip6_groups:
    print("\nStreaming CMIP6 Ensemble to CSV...")
    process_and_stream_csv("CMIP6", cmip6_groups)

print("\n Done! All data streamed directly to disk.")