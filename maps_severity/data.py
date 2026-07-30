import pandas as pd
import numpy as np
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point, MultiPolygon
from shapely.prepared import prep
from scipy.interpolate import griddata
import os
from glob import glob
import pickle

base_folder = './model_outputs/NewVersion/CorrectedFormulaDiffDroughts/'
output_data_folder = './prepared_data_new'
window_sizes = [1]
warming_levels = [1.5, 2.0, 3.0]
categories = ['moderate', 'severe', 'extreme']
antarctica_lat_threshold = -60
grid_resolution = 1.0

plotting_metrics = {
    'duration_mean_percent': {
        'label': 'ΔDuration (%)',
        'categories': {
            'moderate': {'vmin': -3, 'vmax': 3},
            'severe': {'vmin': -3, 'vmax': 3},
            'extreme': {'vmin': -3, 'vmax': 3}
        }
    },
    'frequency_percent': {
        'label': 'ΔFrequency (%)',
        'categories': {
            'moderate': {'vmin': -3, 'vmax': 3},
            'severe': {'vmin': -3, 'vmax': 3},
            'extreme': {'vmin': -3, 'vmax': 3}
        }
    },
    'intensity_mean_percent': {
        'label': 'ΔIntensity (%)',
        'categories': {
            'moderate': {'vmin': -1, 'vmax': 1},
            'severe': {'vmin': -1, 'vmax': 1},
            'extreme': {'vmin': -1, 'vmax': 1}
        }
    }
}

os.makedirs(output_data_folder, exist_ok=True)

land_shp = shpreader.natural_earth(resolution='110m', category='physical', name='land')
try:
    land_union = MultiPolygon([
        g for g in shpreader.Reader(land_shp).geometries()
        if g.geom_type in ['Polygon', 'MultiPolygon']
    ])
    prepared_land = prep(land_union)
except Exception as e:
    print(f"Error loading land shapefile: {e}")
    prepared_land = None

def is_on_land(lon, lat):
    if prepared_land is None:
        return True
    if lon > 180:
        lon -= 360
    return prepared_land.contains(Point(lon, lat))

lon_grid = np.arange(-180, 180 + grid_resolution, grid_resolution)
lat_grid = np.arange(-90, 90 + grid_resolution, grid_resolution)
grid_lon, grid_lat = np.meshgrid(lon_grid, lat_grid)

flat_lon = grid_lon.flatten()
flat_lat = grid_lat.flatten()
land_mask = np.array([
    is_on_land(lon, lat)
    for lon, lat in zip(flat_lon, flat_lat)
]).reshape(grid_lon.shape)

ant_mask = grid_lat >= antarctica_lat_threshold

def recategorize_events_with_cumulative_logic(events):
    moderate_events = []
    severe_events = []
    extreme_events = []

    for event in events:
        intensity = event[3]
        abs_intensity = -intensity

        if abs_intensity > 1.28:
            moderate_events.append(event)
        if abs_intensity > 1.64:
            severe_events.append(event)
        if abs_intensity >= 1.96:
            extreme_events.append(event)

    return {
        'moderate': moderate_events,
        'severe': severe_events,
        'extreme': extreme_events
    }

def calculate_ratios_only(future_events, historical_events):
    fut_cat = recategorize_events_with_cumulative_logic(future_events)
    hist_cat = recategorize_events_with_cumulative_logic(historical_events)

    metrics = {}

    for cat in ["moderate", "severe", "extreme"]:
        fut_events = fut_cat[cat]
        hist_events = hist_cat[cat]

        if len(hist_events) > 0:
            freq_ratio = len(fut_events) / len(hist_events)
        else:
            freq_ratio = np.nan

        fut_duration_mean = np.mean([e[2] for e in fut_events]) if fut_events else np.nan
        hist_duration_mean = np.mean([e[2] for e in hist_events]) if hist_events else np.nan

        if hist_duration_mean > 0 and not np.isnan(fut_duration_mean) and not np.isnan(hist_duration_mean):
            duration_ratio = fut_duration_mean / hist_duration_mean
        else:
            duration_ratio = np.nan

        fut_intensity_mean = np.mean([(-1.0 * e[3]) for e in fut_events]) if fut_events else np.nan
        hist_intensity_mean = np.mean([(-1.0 * e[3]) for e in hist_events]) if hist_events else np.nan

        if hist_intensity_mean > 0 and not np.isnan(fut_intensity_mean) and not np.isnan(hist_intensity_mean):
            intensity_ratio = fut_intensity_mean / hist_intensity_mean
        else:
            intensity_ratio = np.nan

        metrics[f'ratio_frequency_{cat}'] = freq_ratio
        metrics[f'ratio_duration_mean_{cat}'] = duration_ratio
        metrics[f'ratio_intensity_mean_{cat}'] = intensity_ratio

    return metrics

ws = window_sizes[0]

metric_types = {
    'percentage': {
        'duration_mean_percent': 'ratio_duration_mean',
        'frequency_percent': 'ratio_frequency',
        'intensity_mean_percent': 'ratio_intensity_mean'
    }
}

for metric_type, metrics in metric_types.items():
    for primary_metric, data_column in metrics.items():

        metric_data = {
            'grid_lon': grid_lon,
            'grid_lat': grid_lat,
            'land_mask': land_mask,
            'ant_mask': ant_mask,
            'warming_levels': warming_levels,
            'categories': categories,
            'p_props': plotting_metrics[primary_metric],
            'gridded_diffs': {},
            'metric_type': metric_type
        }

        print(f"\n--- Processing Metric: {primary_metric} (WS {ws}, Type: {metric_type}) ---")

        for category in categories:
            for warming_level in warming_levels:

                metric_to_extract = f'{data_column}_{category}'

                accumulated = np.zeros_like(grid_lon, dtype=float)
                counts = np.zeros_like(grid_lon, dtype=int)

                pkl_pattern = os.path.join(
                    base_folder,
                    f'*_WS{ws}_WL{warming_level:.1f}_chunk*.pkl'
                )
                pkl_files = glob(pkl_pattern)

                if not pkl_files:
                    print(f"No files found for WL={warming_level:.1f} | Category={category}.")
                    metric_data['gridded_diffs'][(category, warming_level)] = np.full_like(grid_lon, np.nan)
                    continue

                print(f"Aggregating WL={warming_level:.1f}, Category={category} from {len(pkl_files)} files...")

                for f in pkl_files:
                    try:
                        with open(f, 'rb') as f_in:
                            df_chunk = pickle.load(f_in)

                        df_chunk['lon'] = df_chunk['lon'].apply(
                            lambda x: x - 360 if x > 180 else x
                        )

                        df_chunk['on_land'] = df_chunk.apply(
                            lambda row: is_on_land(row['lon'], row['lat']),
                            axis=1
                        )

                        df_chunk['new_ratios'] = df_chunk.apply(
                            lambda row: calculate_ratios_only(
                                row.get('event_series', []),
                                row.get('hist_event_series', [])
                            ),
                            axis=1
                        )

                        df_chunk[metric_to_extract] = df_chunk['new_ratios'].apply(
                            lambda x: x.get(metric_to_extract, np.nan)
                        )

                        df_chunk = df_chunk[
                            df_chunk['on_land'] &
                            df_chunk[metric_to_extract].notnull()
                        ]

                        if df_chunk.empty:
                            continue

                        pts = df_chunk[['lon', 'lat']].values
                        vals = df_chunk[metric_to_extract].values
                        vals = (vals - 1) * 100

                        interp = griddata(
                            pts,
                            vals,
                            (grid_lon, grid_lat),
                            method='linear'
                        )

                        mask = ~np.isnan(interp)
                        accumulated[mask] += interp[mask]
                        counts[mask] += 1

                    except Exception as e:
                        print(f"Error processing file {f}: {e}")

                mean_diff = np.where(
                    counts > 0,
                    accumulated / counts,
                    np.nan
                )

                mean_diff[~(land_mask & ant_mask)] = np.nan

                metric_data['gridded_diffs'][
                    (category, warming_level)
                ] = mean_diff

                if np.isnan(mean_diff).all():
                    print(
                        f"No valid data for WL={warming_level:.1f} | "
                        f"Category={category} after aggregation."
                    )

        output_filename = os.path.join(
            output_data_folder,
            f'{primary_metric}_WS{ws}_data_Final.pkl'
        )

        with open(output_filename, 'wb') as f_out:
            pickle.dump(metric_data, f_out)

        print(f"Data Saved: {output_filename}")

print("\nData Preparation Complete.")