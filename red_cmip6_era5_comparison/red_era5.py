import os
import re
import time
import xarray as xr
import numpy as np
import pandas as pd
from pycop import estimation, archimedean
import scipy
from tqdm import tqdm
import glob
import gc
import pickle

def load_data(file_pattern, var_name, start_year, end_year, base_dir="./data/"):
    file_path = os.path.join(base_dir, file_pattern)
    files = sorted(glob.glob(file_path))
    
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {file_path}")

    tqdm.write(f"Found {len(files)} files.")

    try:
        ds_all = xr.open_mfdataset(files, combine='by_coords', decode_times=True, engine="netcdf4")[var_name]
        if var_name == "t2m":
            ds_all = ds_all - 273.15
    except Exception as e:
        raise IOError(f"Error opening mfdataset for {var_name} from pattern {file_pattern}: {e}")

    ds_all = ds_all.rename({'valid_time': 'time', 'latitude': 'lat', 'longitude': 'lon'})

    if not ds_all.time.to_index().is_monotonic_increasing:
        tqdm.write("Detected non-monotonic time index.")
        ds_all = ds_all.sortby('time')

    start_date_str = f"{start_year}-01-01"
    end_date_str = f"{end_year}-12-31"

    ds_sliced = ds_all.sel(time=slice(start_date_str, end_date_str))
    
    if ds_sliced.size == 0:
        raise ValueError(f"No data found for {var_name} within the specified time slice {start_year}-{end_year}.")

    del ds_all
    gc.collect()
        
    return ds_sliced

def compute_plotting_position(ts: xr.DataArray, window_size=5) -> xr.DataArray:
    if ts.size == 0 or 'time' not in ts.dims:
        tqdm.write(f"Empty or malformed DataArray passed to compute_plotting_position.")
        if ts.dims:
            return xr.DataArray(np.full(ts.shape, np.nan, dtype=np.float32), dims=ts.dims, coords=ts.coords)
        else:
            return xr.DataArray(np.array([]))
            
    ts_smoothed = ts.rolling(time=window_size, center=False).mean()
    output_dims = ts_smoothed.dims
    output_coords = ts_smoothed.coords

    if 'time' not in ts_smoothed.dims:
        raise ValueError("Smoothed DataArray lost its 'time' dimension.")

    months = ts_smoothed['time'].dt.month.values
    time_len, lat_len, lon_len = ts_smoothed.shape
    result = np.full(ts_smoothed.shape, np.nan, dtype=np.float32)

    for m in range(1, 13):
        idx = np.where(months == m)[0]
        if len(idx) == 0:
            continue
            
        month_data = ts_smoothed.isel(time=idx).values
        flat = month_data.reshape(len(idx), -1)
        mask = ~np.isnan(flat)

        for col in range(flat.shape[1]):
            valid = mask[:, col]
            if valid.sum() < 2:
                continue
                
            ranks = scipy.stats.rankdata(flat[valid, col], method='average')
            pp = (ranks - 0.44) / (valid.sum() + 0.12)
            flat[valid, col] = pp

        result_month = flat.reshape(len(idx), lat_len, lon_len)
        result[idx, :, :] = result_month
    
    del ts_smoothed
    del months, time_len, lat_len, lon_len, idx, month_data, flat, mask, ranks, pp, result_month
    gc.collect()

    return xr.DataArray(result, dims=output_dims, coords=output_coords)

def compute_copula_index(u, v):

    u = np.asarray(u)
    v = np.asarray(v)
    
    if len(u) != len(v):
        raise ValueError("u and v must have same length.")
    
    mask = ~np.isnan(u) & ~np.isnan(v)
    if mask.sum() < 2:
        return np.full_like(u, np.nan, dtype=float)
    
    u_valid = u[mask].astype(float)
    v_valid = v[mask].astype(float)
    
    u_uni = u_valid
    v_uni = v_valid
    
    u_uni = np.clip(u_uni, 1e-10, 1 - 1e-10)
    v_uni = np.clip(v_uni, 1e-10, 1 - 1e-10)
    
    copula = archimedean("clayton")
    try:
        param, _ = estimation.fit_cmle(copula, np.vstack([u_uni, v_uni]))
    except Exception as e:
        tqdm.write(f"Copula parameter estimation failed: {e}")
        return np.full_like(u_valid, np.nan)
    
    try:
        cdf_vals = copula.get_cdf(u_uni, v_uni, param)
    except Exception as e:
        tqdm.write(f"Copula get_cdf failed: {e}")
        return np.full_like(u_valid, np.nan)
    
    cdf_vals = np.clip(cdf_vals, 1e-10, 1 - 1e-10)
      
    std_index_valid = scipy.stats.norm.ppf(cdf_vals)

    mean_sci = np.nanmean(std_index_valid)
    std_sci = np.nanstd(std_index_valid)

    if std_sci > 1e-6:
        std_index_final = (std_index_valid - mean_sci) / std_sci
    else:
        std_index_final = np.full_like(std_index_valid, np.nan)
    
    std_index = np.full_like(u, np.nan, dtype=float)
    std_index[mask] = std_index_final
    
    return std_index

def extract_events(index, times, threshold=-1.28):
    droughts = []
    in_event = False
    start_time_idx = -1

    index_np = np.asarray(index)
    times_np = np.asarray(times)

    for i in range(len(index_np)):
        val = index_np[i]
        
        if np.isnan(val):
            if in_event:
                end_time_idx = i
                segment = index_np[start_time_idx:end_time_idx]
                if len(segment) > 0:
                    droughts.append([
                        times_np[start_time_idx],
                        times_np[end_time_idx - 1],
                        end_time_idx - start_time_idx,
                        segment.mean()
                    ])
                in_event = False
            continue

        if val < threshold and not in_event:
            start_time_idx = i
            in_event = True
        elif val >= threshold and in_event:
            end_time_idx = i
            segment = index_np[start_time_idx:end_time_idx]
            if len(segment) > 0:
                droughts.append([
                    times_np[start_time_idx],
                    times_np[end_time_idx - 1],
                    end_time_idx - start_time_idx,
                    segment.mean()
                ])
            in_event = False
    
    if in_event:
        end_time_idx = len(index_np)
        segment = index_np[start_time_idx:end_time_idx]
        if len(segment) > 0:
            droughts.append([
                times_np[start_time_idx],
                times_np[end_time_idx - 1],
                end_time_idx - start_time_idx,
                segment.mean()
            ])
            
    del index_np, times_np
    return droughts

def save_buffered_results(buffer, output_dir, model_name, WS, chunk_num):
    if not buffer:
        return
    
    df_chunk = pd.concat(buffer, ignore_index=True)
    safe_model = model_name.replace("/", "_").replace(" ", "_")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = f"{output_dir}/IDF_metrics_{safe_model}_WS{WS}_1985-2014_chunk{chunk_num}.pkl"
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(df_chunk, f)
        tqdm.write(f"[{model_name} | WS={WS}] Saved chunk {chunk_num}: {output_path} ({df_chunk.shape[0]} rows)")
    except Exception as e:
        tqdm.write(f"[{model_name} | WS={WS}] Error saving pickle file {output_path}: {e}")
    
    del df_chunk
    gc.collect()

WS = 1
start_year, end_year = 1985, 2014
model = "ERA5_data"

tqdm.write(f"Processing data for the period {start_year}-{end_year} with WS={WS}")

MAX_BUFFER_SIZE = 20000
current_grid_results_buffer = []
chunk_counter = 0

try:
    t0_load = time.time()

    ssrd = load_data("era5_solar_*.nc", "ssrd", start_year, end_year)
    u10 = load_data("era5_u_*.nc", "u10", start_year, end_year)
    v10 = load_data("era5_v_*.nc", "v10", start_year, end_year)
    t2m = load_data("era5_t_*.nc", "t2m", start_year, end_year)
    tqdm.write(f"[{model}] Loaded data for all variables in {time.time() - t0_load:.2f}s")

    ssrd_Wm2 = ssrd / 3600.0
    
    wind_speed = np.sqrt(u10**2 + v10**2)
    Tcell = 4.3 + 0.943 * t2m + 0.028 * ssrd_Wm2 - 1.528 * wind_speed
    pr = 1 - 0.005 * (Tcell - 25)
    PV_pot = pr * (ssrd_Wm2 / 1000)
    wind_speed_100 = wind_speed * ((80 / 10) ** (0.143))
    W_pot = xr.where(
        wind_speed_100 < 3.5, 0,
        xr.where(
            wind_speed_100 < 13,
            (wind_speed_100**3 - 3.5**3) / (13**3 - 3.5**3),
            xr.where(wind_speed_100 < 25, 1, 0)
        )
    )

    del u10, v10, t2m, ssrd, ssrd_Wm2, wind_speed, Tcell, pr, wind_speed_100
    gc.collect()

    PV_smoothed = compute_plotting_position(PV_pot, window_size=WS)
    W_smoothed = compute_plotting_position(W_pot, window_size=WS)

    del PV_pot, W_pot
    gc.collect()

    for i in tqdm(range(PV_smoothed.lat.size), desc=f"Processing grid cells for {model}"):
        for j in range(PV_smoothed.lon.size):
            u_full = PV_smoothed[:, i, j].values
            v_full = W_smoothed[:, i, j].values
            t = PV_smoothed['time'].values

            valid_mask = ~np.isnan(u_full) & ~np.isnan(v_full)
            u_valid = u_full[valid_mask]
            v_valid = v_full[valid_mask]

            if len(u_valid) < 2:
                del u_full, v_full, t
                continue

            try:
                std_idx_valid = compute_copula_index(u_valid, v_valid)

                if np.all(np.isnan(std_idx_valid)):
                    del u_full, v_full, t
                    continue

                std_idx_full_length = np.full_like(u_full, np.nan)
                std_idx_full_length[valid_mask] = std_idx_valid

                events = extract_events(std_idx_full_length, t)

                result_entry = {
                    'model': model,
                    'scenario': "reanalysis",
                    'window_size': WS,
                    'lat': PV_smoothed.lat[i].item(),
                    'lon': PV_smoothed.lon[j].item(),
                    'period_start': start_year,
                    'period_end': end_year,
                    'frequency': len(events),
                    'pv_series': u_full,
                    'wind_series': v_full,
                    'event_series': events,
                    'intensity_mean': np.mean([(-1 * e[3]) for e in events]) if events else np.nan,
                    'duration_mean': np.mean([e[2] for e in events]) if events else np.nan,
                }

                current_grid_results_buffer.append(pd.DataFrame([result_entry]))

                del u_full, v_full, t, std_idx_valid, std_idx_full_length, events, result_entry

                if len(current_grid_results_buffer) >= MAX_BUFFER_SIZE:
                    chunk_counter += 1
                    save_buffered_results(current_grid_results_buffer, "ERA5_outputs/CorrectedFormula", model, WS, chunk_counter)
                    current_grid_results_buffer = []
                    gc.collect()

            except Exception as e:
                tqdm.write(f"[{model}] Event extraction failure at lat={i}, lon={j}: {e}")
                if 'u_full' in locals():
                    del u_full
                if 'v_full' in locals():
                    del v_full
                if 't' in locals():
                    del t
                if 'std_idx_valid' in locals():
                    del std_idx_valid
                if 'std_idx_full_length' in locals():
                    del std_idx_full_length

    gc.collect()

    if current_grid_results_buffer:
        chunk_counter += 1
        save_buffered_results(current_grid_results_buffer, "ERA5_outputs/CorrectedFormula", model, WS, chunk_counter)
        current_grid_results_buffer = []

    del PV_smoothed, W_smoothed
    gc.collect()

    tqdm.write(f"[{model}] Finished processing in {time.time() - t0_load:.2f}s")

except Exception as e:
    tqdm.write(f"[{model}] Error in processing: {e}.")
    for var_name in ['ssrd', 'u10', 'v10', 't2m', 'PV_pot', 'W_pot', 'PV_smoothed', 'W_smoothed']:
        if var_name in locals():
            del locals()[var_name]
    gc.collect()

print("Processing complete.")