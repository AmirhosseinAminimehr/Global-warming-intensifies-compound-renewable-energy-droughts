import pandas as pd
import numpy as np
import os
from glob import glob
from scipy.interpolate import griddata
import pickle

base_folder = 'model_outputs/GridCellLevel_NoChunk/Corrected'
output_dir = 'processed_data'
warming_levels = [1.5, 2.0, 3.0]
antarctica_lat_threshold = -60
grid_resolution = 1.0

metrics = {
    'diff_rsds_mean': (-6, 6, 'ΔShortwave Radiation (W/m²)'),
    'diff_uas_mean': (-0.2, 0.2, 'ΔEastward Wind Speed (m/s)'),
    'diff_vas_mean': (-0.2, 0.2, 'ΔNorthward Wind Speed (m/s)')
}
metric_names = list(metrics.keys())


def get_land_mask(grid_lon, grid_lat, antarctica_lat_threshold):
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import Point, MultiPolygon
        from shapely.prepared import prep
        
        land_shp = shpreader.natural_earth(resolution='110m', category='physical', name='land')
        land_union = MultiPolygon([g for g in shpreader.Reader(land_shp).geometries() if g.geom_type in ['Polygon', 'MultiPolygon']])
        prepared_land = prep(land_union)

        print("Pre-calculating land mask...")
        flat_lon = grid_lon.flatten()
        flat_lat = grid_lat.flatten()
        
        land_mask = np.array([prepared_land.contains(Point(lon, lat)) for lon, lat in zip(flat_lon, flat_lat)]).reshape(grid_lon.shape)
        
        ant_mask = grid_lat >= antarctica_lat_threshold
        
        ocean_mask = ~land_mask
        final_mask = ocean_mask & ant_mask
        
        print("Land mask pre-calculation complete.")
        return final_mask
        
    except Exception as e:
        print(f"Error creating land mask: {e}.")
        return np.zeros_like(grid_lon, dtype=bool)


def process_data_for_plotting(base_folder, warming_levels, metrics, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    
    lon_grid = np.arange(-180, 180 + grid_resolution, grid_resolution)
    lat_grid = np.arange(-90, 90 + grid_resolution, grid_resolution)
    grid_lon, grid_lat = np.meshgrid(lon_grid, lat_grid)
    
    final_mask = get_land_mask(grid_lon, grid_lat, antarctica_lat_threshold)
    
    np.save(os.path.join(output_dir, 'grid_lon.npy'), grid_lon)
    np.save(os.path.join(output_dir, 'grid_lat.npy'), grid_lat)
    np.save(os.path.join(output_dir, 'final_mask.npy'), final_mask)

    metric_names = list(metrics.keys())
    accumulators = {
        (metric, wl): np.zeros_like(grid_lon, dtype=np.float32)
        for metric in metric_names
        for wl in warming_levels
    }
    model_counts = {wl: 0 for wl in warming_levels}

    all_files = glob(os.path.join(base_folder, "metrics_*.csv"))
    grouped_files = {}
    for filepath in all_files:
        filename = os.path.basename(filepath)
        model_name = filename.split('_')[1]
        if model_name not in grouped_files:
            grouped_files[model_name] = []
        grouped_files[model_name].append(filepath)

    for model_name, files in grouped_files.items():
        print(f"Processing model: {model_name}")
        try:
            model_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        except Exception as e:
            print(f"Error reading files for model {model_name}: {e}.")
            continue
        
        for wl in warming_levels:
            df_wl = model_df[model_df['warming_level'] == wl].copy()
            if not df_wl.empty:
                df_wl['lon'] = df_wl['lon'].apply(lambda x: x - 360 if x > 180 else x)
                model_counts[wl] += 1
                
                for metric in metric_names:
                    pts = df_wl[['lon', 'lat']].values
                    vals = df_wl[metric].values
                    
                    interp = griddata(pts, vals, (grid_lon, grid_lat), method='linear')
                    
                    interp[final_mask] = np.nan
                    
                    accumulators[(metric, wl)] = np.nansum([accumulators[(metric, wl)], interp], axis=0)

    averaged_grids = {}
    for r, metric in enumerate(metric_names):
        for c, wl in enumerate(warming_levels):
            key = (metric, wl)
            if model_counts[wl] > 0:
                avg_grid = accumulators[key] / model_counts[wl]
            else:
                avg_grid = np.full_like(grid_lon, np.nan)
            
            filename = f'avg_{metric}_{wl}C.npy'
            np.save(os.path.join(output_dir, filename), avg_grid)
            averaged_grids[key] = os.path.join(output_dir, filename)
    
    metadata = {
        'warming_levels': warming_levels,
        'metrics': metrics,
        'antarctica_lat_threshold': antarctica_lat_threshold,
        'model_counts': model_counts,
        'averaged_grids_paths': averaged_grids
    }
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\n Data processing complete. {len(averaged_grids)} arrays and metadata saved to '{output_dir}'.")


if __name__ == '__main__':
    process_data_for_plotting(base_folder, warming_levels, metrics, output_dir)