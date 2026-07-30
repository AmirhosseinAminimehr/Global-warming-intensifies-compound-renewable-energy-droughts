import os
import glob
import pandas as pd
import numpy as np
import xarray as xr
import pickle
from tqdm import tqdm

def get_continent(lat, lon):
    def wrap_lon(l):
        return (l + 360) % 360

    lon = wrap_lon(lon)
    if 10 <= lat <= 85 and 180 <= lon <= 300:
        return "North America"
    elif -60 <= lat <= 15 and 270 <= lon <= 330:
        return "South America"
    elif -40 <= lat <= 40 and (340 <= lon <= 360 or 0 <= lon <= 50):
        return "Africa"
    elif 35 <= lat <= 70 and (lon >= 350 or lon <= 60):
        return "Europe"
    elif -10 <= lat <= 80 and 40 <= lon <= 180:
        return "Asia"
    elif -50 <= lat <= -10 and 110 <= lon <= 155:
        return "Australia"
    else:
        return "Other"

def find_common_coords(dfs):
    all_lats = set()
    all_lons = set()
    for df in dfs:
        all_lats.update(df['lat'].unique())
    all_lons.update(df['lon'].unique())
    return sorted(list(all_lats)), sorted(list(all_lons))

def interpolate_to_common_grid(df, common_lats, common_lons, warming_level):
    ds = df.set_index(['lat', 'lon'])[df.columns.difference(['lat', 'lon', 'model', 'scenario', 'warming_level', 'period_start', 'period_end'])].to_xarray()

    new_coords = {'lat': common_lats, 'lon': common_lons}
    ds_new = xr.DataArray(
        np.full((len(common_lats), len(common_lons)), np.nan),
        coords=new_coords,
        dims=['lat', 'lon']
    )

    interpolated = ds.interp(lat=common_lats, lon=common_lons, method='linear', kwargs={'fill_value': None})

    interpolated_df = interpolated.to_dataframe().reset_index()
    interpolated_df['warming_level'] = warming_level

    return interpolated_df

output_data_file = "prepared_data_CorrectedFormula.pkl"

print("--- Data Preparation ---")
print("Loading and combining data from CSV and PKL files...")

csv_files = glob.glob(os.path.join("model_outputs/GridCellLevel_NoChunk/Corrected/", 'metrics_*.csv'))
if not csv_files:
    print("No CSV files found.")
    exit()

csv_dfs = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
csv_dfs['model'] = csv_dfs['model'].str.replace('-', '_')

pkl_files = glob.glob(os.path.join("model_outputs/NewVersion/CorrectedFormula/WS1", 'IDF_metrics_*.pkl'))

if not pkl_files:
    print("No pickle files found.")
    exit()

print("Processing pickle files...")
final_df_list = []

for pkl_file in tqdm(pkl_files, desc="Processing pickle files"):
    with open(pkl_file, 'rb') as f:
        df = pickle.load(f)

    df.drop(columns=['event_series', 'hist_event_series', 'hist_pv_series', 'hist_wind_series', 'pv_series', 'wind_series'], inplace=True, errors='ignore')

    if df['warming_level'].isnull().any():
        df['warming_level'] = 0.0

    df_ws1 = df[df['window_size'] == 1].copy()

    current_merged_df = pd.merge(csv_dfs, df_ws1, on=['model', 'scenario', 'warming_level', 'period_start', 'period_end', 'lat', 'lon'], how='inner')
    final_df_list.append(current_merged_df)

if not final_df_list:
    print("No data to process after merging.")
    exit()

merged_df = pd.concat(final_df_list, ignore_index=True)

models = merged_df['model'].unique()
warming_levels = merged_df['warming_level'].unique()

all_lats, all_lons = find_common_coords([merged_df[merged_df['model'] == m] for m in models])
print(f"Found common grid with {len(all_lats)} latitudes and {len(all_lons)} longitudes.")

interpolated_dfs = []
for wl in tqdm(warming_levels, desc="Interpolating data"):
    wl_dfs = [merged_df[(merged_df['warming_level'] == wl) & (merged_df['model'] == m)] for m in models]

    interpolated_wl_dfs = []
    for m, df in zip(models, wl_dfs):
        if not df.empty:
            interp_df = interpolate_to_common_grid(df, all_lats, all_lons, wl)
            interp_df['model'] = m
            interpolated_wl_dfs.append(interp_df)

    if interpolated_wl_dfs:
        interpolated_dfs.append(pd.concat(interpolated_wl_dfs, ignore_index=True))

if not interpolated_dfs:
    print("No data to process after interpolation.")
    exit()

final_df = pd.concat(interpolated_dfs, ignore_index=True)

group_cols = ['lat', 'lon', 'warming_level']

numeric_cols = final_df.select_dtypes(include=np.number).columns.tolist()

cols_to_avg = [col for col in numeric_cols if col not in group_cols]

mean_df = final_df.groupby(group_cols)[cols_to_avg].mean().reset_index()

mean_df['continent'] = mean_df.apply(lambda row: get_continent(row['lat'], row['lon']), axis=1)
mean_df = mean_df[mean_df['continent'] != 'Other']
print(f"Final dataset has {len(mean_df)} rows after cleaning.")

print(f"Saving prepared data to '{output_data_file}'...")
mean_df.to_pickle(output_data_file)