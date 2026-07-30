import os
import re
import time
import xarray as xr
import numpy as np
import pandas as pd
from pycop import estimation, archimedean
import scipy
from tqdm import tqdm
import cftime
import gc
import pickle
import scipy.stats
from scipy.stats import  norm, rankdata

warming_df = pd.read_csv("./Yr_gw_ssp585_30.csv")
warming_df = warming_df.melt(id_vars=["Row"], var_name="model", value_name="year")
warming_df = warming_df.dropna(subset=["year"])
warming_df["year"] = warming_df["year"].astype(int)
warming_df["warming_level"] = warming_df["Row"].astype(float)
warming_df["start_year"] = warming_df["year"] - 14
warming_df["end_year"] = warming_df["year"] + 15

def get_model_names_from_folder(var_folder):

    files = os.listdir(var_folder)
    pattern = re.compile(rf"uas_day_(.+?)_historical")
    models = set()
    for f in files:
        match = pattern.search(f)
        if match:
            models.add(match.group(1))
    return sorted(list(models)) 

def load_all_years_for_model(var, model, scenario, base_dir="./CMIP6/"):

    folder = os.path.join(base_dir, var)
        
    hist_files = [os.path.join(folder, f) for f in os.listdir(folder)
                  if f.startswith(f"{var}_day_{model}_historical")]
    fut_files = [os.path.join(folder, f) for f in os.listdir(folder)
                  if f.startswith(f"{var}_day_{model}_{scenario}")]
        
    if not hist_files and not fut_files: 
        raise FileNotFoundError(f"No files found for {var} {model} {scenario} in {folder}")
    
    try:
        if hist_files:
            ds_hist = xr.open_mfdataset(hist_files, combine='by_coords', engine="netcdf4")[var]
            if var == "tas":
                ds_hist = ds_hist - 273.15
        else:
            ds_hist = xr.DataArray() 
            tqdm.write(f"No historical files for {var} {model}.")

        if fut_files:
            ds_fut = xr.open_mfdataset(fut_files, combine='by_coords', engine="netcdf4")[var]
            if var == "tas":
                ds_fut = ds_fut - 273.15
        else:
            ds_fut = xr.DataArray() 
            tqdm.write(f"No future files for {var} {model} {scenario}.")
    except Exception as e:
        raise IOError(f"Error opening mfdataset for {var} {model} {scenario}: {e}")

    if ds_hist.size > 0 and ds_fut.size > 0:
        ds_all = xr.concat([ds_hist, ds_fut], dim='time')
    elif ds_hist.size > 0:
        ds_all = ds_hist
    elif ds_fut.size > 0:
        ds_all = ds_fut
    else:
        raise ValueError(f"No valid data loaded for {var} {model} {scenario}.")

    del ds_hist, ds_fut
        
    return ds_all

def extract_period_data(full_data, start_year, end_year):

    buffer_days = 30
    start_date = pd.Timestamp(f"{start_year}-01-01") - pd.Timedelta(days=buffer_days)
    end_date = pd.Timestamp(f"{end_year}-12-31") + pd.Timedelta(days=buffer_days)

    if isinstance(full_data['time'].values[0], cftime.datetime): 
        cftime_class = full_data['time'].values[0].__class__ 
        start_date_cftime = cftime_class(start_date.year, start_date.month, start_date.day)
        end_date_cftime = cftime_class(end_date.year, end_date.month, end_date.day)
        ds_sliced = full_data.sel(time=slice(start_date_cftime, end_date_cftime))
    else:
        ds_sliced = full_data.sel(time=slice(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
    
    if ds_sliced.size == 0:
        raise ValueError(f"No data found within the specified time slice {start_year}-{end_year}.")

    return ds_sliced

def compute_plotting_position(ts: xr.DataArray, window_size=5) -> xr.DataArray:

    if ts.size == 0 or 'time' not in ts.dims:
        tqdm.write(f"Empty or malformed DataArray passed to compute_plotting_position.")
        if ts.dims:
            return xr.DataArray(np.full(ts.shape, np.nan, dtype=np.float32), dims=ts.dims, coords=ts.coords)
        else:
            return xr.DataArray(np.array([]))

    ts = ts.load()      
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
        raise RuntimeError(f"Copula parameter estimation failed: {e}")
    
    try:
        cdf_vals = copula.get_cdf(u_uni, v_uni, param)
    except Exception as e:
        raise RuntimeError(f"Copula get_cdf failed: {e}")
    
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

def save_buffered_results(buffer, output_dir, model, WS, warming_level, chunk_num):

    if not buffer:
        return
    
    df_chunk = pd.concat(buffer, ignore_index=True)
    safe_model = model.replace("/", "_").replace(" ", "_")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = f"{output_dir}/IDF_metrics_{safe_model}_WS{WS}_WL{warming_level:.1f}_chunk{chunk_num}.pkl"
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(df_chunk, f)
        tqdm.write(f"[{model} | WS={WS} | WL={warming_level}] Saved chunk {chunk_num}: {output_path} ({df_chunk.shape[0]} rows)")
    except Exception as e:
        tqdm.write(f"[{model} | WS={WS} | WL={warming_level}] Error saving pickle file {output_path}: {e}")
    
    del df_chunk
    gc.collect()

window_sizes = [1, 3, 5]
scenarios = ["ssp585"] 
all_models = ["CanESM5", "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR", "MIROC6", "MPI-ESM1-2-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0", "CMCC-CM2-SR5", "CMCC-ESM2", "IITM-ESM"] 


tqdm.write(f"Processing models: {all_models}") 

MAX_BUFFER_SIZE = 10000 

for model in tqdm(all_models, desc="Models"): 
    for scenario in scenarios: 

        t0_load_all = time.time() 
        uas_all = None 
        vas_all = None 
        rsds_all = None 
        tas_all = None 
        
        try: 
            uas_all = load_all_years_for_model("uas", model, scenario) 
            vas_all = load_all_years_for_model("vas", model, scenario) 
            rsds_all = load_all_years_for_model("rsds", model, scenario) 
            tas_all = load_all_years_for_model("tas", model, scenario) 
            tqdm.write(f"[{model}] Loaded ALL data for all WS in {time.time() - t0_load_all:.2f}s") 
        except Exception as e: 
            tqdm.write(f"[{model}] Error loading ALL data for all WS: {e}.") 
            for var in [uas_all, vas_all, rsds_all, tas_all]: 
                if var is not None: 
                    del var 
            gc.collect() 
            continue  

        for WS in window_sizes: 
            t0_ws_all = time.time() 
                 
            wind_speed_all = np.sqrt(uas_all**2 + vas_all**2) 
            Tcell_all = 4.3 + 0.943 * tas_all + 0.028 * rsds_all - 1.528 * wind_speed_all 
            pr_all = 1 - 0.005 * (Tcell_all - 25) 
            PV_pot_all = pr_all * (rsds_all / 1000) 
            wind_speed_100_all = wind_speed_all * ((80 / 10) ** (0.143)) 
            W_pot_all = xr.where( 
                wind_speed_100_all < 3.5, 0, 
                xr.where( 
                    wind_speed_100_all < 13, 
                    (wind_speed_100_all**3 - 3.5**3) / (13**3 - 3.5**3), 
                    xr.where(wind_speed_100_all < 25, 1, 0) 
                ) 
            ) 

              
            
            del wind_speed_all, Tcell_all, pr_all, wind_speed_100_all 
            gc.collect() 

            PV_all_smoothed = compute_plotting_position(PV_pot_all, window_size = WS) 
            W_all_smoothed = compute_plotting_position(W_pot_all, window_size = WS) 
            del PV_pot_all, W_pot_all  
            gc.collect() 

            std_idx_all_data = xr.full_like(PV_all_smoothed, np.nan) 
                
            all_series_per_grid = {} 

            for i in range(PV_all_smoothed.lat.size): 
                for j in range(PV_all_smoothed.lon.size): 
                    u_current_full = PV_all_smoothed[:, i, j].values 
                    v_current_full = W_all_smoothed[:, i, j].values 
                        
                    valid_mask = ~np.isnan(u_current_full) & ~np.isnan(v_current_full)
                    u_current_valid = u_current_full[valid_mask]
                    v_current_valid = v_current_full[valid_mask]

                    if len(u_current_valid) < 2: 
                        continue 
                        
                    try: 
                        std_idx = compute_copula_index(u_current_valid, v_current_valid) 
                        
                        std_idx_full_length = np.full_like(u_current_full, np.nan) 
                        std_idx_full_length[valid_mask] = std_idx
                        std_idx_all_data[:, i, j] = std_idx_full_length
                        
                        all_series_per_grid[(i, j)] = {'u_series': u_current_full.copy(), 'v_series': v_current_full.copy()} 
                            
                        del std_idx 
                        del std_idx_full_length 
                    except Exception as e: 
                        tqdm.write(f"[{model}] Copula failed at lat={i}, lon={j} for ALL data (WS={WS}): {e}") 
            gc.collect()  
                
            tqdm.write(f"[{model} | WS={WS}] Processed ALL data and copula indices in {time.time() - t0_ws_all:.2f}s") 

            t0_hist = time.time()
            hist_start, hist_end = 1850, 1879
            hist_events_grid = {} 
            
            hist_std_idx = extract_period_data(std_idx_all_data, hist_start, hist_end)
            hist_PV_smoothed = extract_period_data(PV_all_smoothed, hist_start, hist_end)
            hist_W_smoothed = extract_period_data(W_all_smoothed, hist_start, hist_end)
            
            for i in range(hist_PV_smoothed.lat.size): 
                for j in range(hist_PV_smoothed.lon.size): 
                    idx_vals = hist_std_idx[:, i, j].values 
                    t_vals = hist_PV_smoothed['time'].values  

                    if np.isnan(idx_vals).all(): 
                        del idx_vals, t_vals  
                        continue  

                    if isinstance(t_vals[0], cftime.datetime): 
                        cftime_class = t_vals[0].__class__
                        hist_start_dt = cftime_class(hist_start, 1, 1) 
                        hist_end_dt = cftime_class(hist_end, 12, 31) 
                    else: 
                        hist_start_dt = np.datetime64(f"{hist_start}-01-01") 
                        hist_end_dt = np.datetime64(f"{hist_end}-12-31") 
                        
                    hist_mask = (t_vals >= hist_start_dt) & (t_vals <= hist_end_dt) 
                        
                    events = extract_events(idx_vals[hist_mask], t_vals[hist_mask]) 
                        
                    hist_events_grid[(i, j)] = { 
                        'events': events, 
                        'pv_series': all_series_per_grid.get((i, j), {}).get('u_series', np.array([])), 
                        'wind_series': all_series_per_grid.get((i, j), {}).get('v_series', np.array([])) 
                    } 

                    del idx_vals, t_vals, hist_start_dt, hist_end_dt, hist_mask, events  
                
            del hist_std_idx, hist_PV_smoothed, hist_W_smoothed
            gc.collect()  
            tqdm.write(f"[{model} | WS={WS}] Processed historical events in {time.time() - t0_hist:.2f}s") 

            relevant_windows = warming_df[warming_df["model"] == model.replace("-", "_")] 
                
            output_dir = "model_outputs/NewVersion/Temp" 
            os.makedirs(output_dir, exist_ok=True) 

            for _, row in relevant_windows.iterrows(): 
                t0_wl_start = time.time() 
                current_warming_level = row['warming_level'] 
                start, end = row['start_year'], row['end_year'] 
                        
                current_wl_grid_results_buffer = [] 
                chunk_counter = 0 

                try: 
                    fut_std_idx = extract_period_data(std_idx_all_data, start, end)
                    fut_PV_smoothed = extract_period_data(PV_all_smoothed, start, end)
                    fut_W_smoothed = extract_period_data(W_all_smoothed, start, end)

                    tqdm.write(f"[{model} | WS={WS} | WL={current_warming_level}] Extracted future window data in {time.time() - t0_wl_start:.2f}s") 

                    for i in range(fut_PV_smoothed.lat.size): 
                        for j in range(fut_PV_smoothed.lon.size): 
                            try:
                                idx_vals = fut_std_idx[:, i, j].values  
                                t_vals = fut_PV_smoothed['time'].values  

                                if isinstance(t_vals[0], cftime.datetime): 
                                    cftime_class = t_vals[0].__class__
                                    start_dt = cftime_class(start, 1, 1) 
                                    end_dt = cftime_class(end, 12, 31) 
                                else: 
                                    start_dt = np.datetime64(f"{start}-01-01") 
                                    end_dt = np.datetime64(f"{end}-12-31") 
                                    
                                mask = (t_vals >= start_dt) & (t_vals <= end_dt) 
                                    
                                events = extract_events(idx_vals[mask], t_vals[mask]) 
                                    
                                hist_data_for_grid = hist_events_grid.get( 
                                    (i, j),  
                                    {'events': [], 'pv_series': np.array([]), 'wind_series': np.array([])} 
                                ) 
                                hist_events = hist_data_for_grid['events'] 
                                hist_pv_series_stored = hist_data_for_grid['pv_series'] 
                                hist_wind_series_stored = hist_data_for_grid['wind_series'] 

                                full_u_series = all_series_per_grid.get((i, j), {}).get('u_series', np.array([]))
                                full_v_series = all_series_per_grid.get((i, j), {}).get('v_series', np.array([]))


                                result_entry = { 
                                    'model': model, 
                                    'scenario': scenario, 
                                    'window_size': WS, 
                                    'lat': fut_PV_smoothed.lat[i].item(), 
                                    'lon': fut_PV_smoothed.lon[j].item(), 
                                    'warming_level': current_warming_level, 
                                    'period_start': start, 
                                    'period_end': end, 
                                    'frequency': len(events), 
                                    'hist_frequency': len(hist_events), 
                                    'ratio_duration_mean': (np.mean([e[2] for e in events]) / np.mean([e[2] for e in hist_events])) if events and hist_events else np.nan, 
                                    'ratio_frequency': len(events) / len(hist_events) if hist_events else np.nan, 
                                    'hist_pv_series': hist_pv_series_stored, 
                                    'hist_wind_series': hist_wind_series_stored, 
                                    'pv_series': full_u_series, 
                                    'wind_series': full_v_series,  
                                    'event_series': events, 
                                    'hist_event_series': hist_events, 
                                    'intensity_mean': np.mean([(-1 * e[3]) for e in events]) if events else np.nan, 
                                    'hist_intensity_mean': np.mean([(-1 * e[3]) for e in hist_events]) if hist_events else np.nan, 
                                    'ratio_intensity_mean': (np.mean([(-1 * e[3]) for e in events]) / np.mean([(-1 * e[3]) for e in hist_events])) if events and hist_events else np.nan, 
                                    'duration_mean': np.mean([e[2] for e in events]) if events else np.nan, 
                                    'hist_duration_mean': np.mean([e[2] for e in hist_events]) if hist_events else np.nan, 
                                    'hist_drought_days': sum(e[2] for e in hist_events) if hist_events else 0,
                                    'fut_drought_days': sum(e[2] for e in events) if events else 0,
                                    'ratio_drought_days': (sum(e[2] for e in events) / sum(e[2] for e in hist_events)) if events and hist_events and sum(e[2] for e in hist_events) > 0 else np.nan,
                                    'diff_duration_mean': (np.mean([e[2] for e in events]) - np.mean([e[2] for e in hist_events])) if events and hist_events else np.nan, 
                                    'diff_frequency': len(events) - len(hist_events) if hist_events else np.nan, 
                                    'diff_intensity_mean': (np.mean([(-1 * e[3]) for e in events]) - np.mean([(-1 * e[3]) for e in hist_events])) if events and hist_events else np.nan, 
                                    'diff_drought_days': (sum(e[2] for e in events) - sum(e[2] for e in hist_events)) if events and hist_events and sum(e[2] for e in hist_events) > 0 else np.nan

                                } 
                                current_wl_grid_results_buffer.append(pd.DataFrame([result_entry])) 
                                    
                                del idx_vals, t_vals, start_dt, end_dt, mask, events, hist_events, hist_data_for_grid, hist_pv_series_stored, hist_wind_series_stored, result_entry 
                                    
                                if len(current_wl_grid_results_buffer) >= MAX_BUFFER_SIZE: 
                                    chunk_counter += 1 
                                    save_buffered_results(current_wl_grid_results_buffer, output_dir, model, WS, current_warming_level, chunk_counter) 
                                    current_wl_grid_results_buffer = [] 
                                    gc.collect() 

                            except Exception as e: 
                                tqdm.write(f"[{model} | WS={WS} | WL={current_warming_level}] Event extraction failure at lat={i}, lon={j}: {e}") 
                    gc.collect()  
                        
                    if current_wl_grid_results_buffer: 
                        chunk_counter += 1 
                        save_buffered_results(current_wl_grid_results_buffer, output_dir, model, WS, current_warming_level, chunk_counter) 
                        current_wl_grid_results_buffer = [] 
                            
                    del fut_std_idx, fut_PV_smoothed, fut_W_smoothed 
                    gc.collect() 

                    tqdm.write(f"[{model} | WS={WS} | WL={current_warming_level}] Finished processing warming level in {time.time() - t0_wl_start:.2f}s") 
                except Exception as e: 
                    tqdm.write(f"[{model} | WS={WS} | WL={current_warming_level}] Error in future window processing: {e}.") 
                    gc.collect() 
                        
            if 'PV_all_smoothed' in locals() and PV_all_smoothed is not None: del PV_all_smoothed 
            if 'W_all_smoothed' in locals() and W_all_smoothed is not None: del W_all_smoothed 
            if 'std_idx_all_data' in locals() and std_idx_all_data is not None: del std_idx_all_data 
            if 'all_series_per_grid' in locals() and all_series_per_grid is not None: del all_series_per_grid 
            if 'hist_events_grid' in locals() and hist_events_grid is not None: del hist_events_grid 
            gc.collect() 

        if uas_all is not None: del uas_all 
        if vas_all is not None: del vas_all 
        if rsds_all is not None: del rsds_all 
        if tas_all is not None: del tas_all 
        gc.collect() 

print("Processing complete.")