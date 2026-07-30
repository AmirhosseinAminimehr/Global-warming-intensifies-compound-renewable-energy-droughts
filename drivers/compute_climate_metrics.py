import os
import time
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm
import cftime
import gc

print("Loading warming windows CSV...")
warming_df = pd.read_csv("./Yr_gw_ssp585_30.csv")
warming_df = warming_df.melt(id_vars=["Row"], var_name="model", value_name="year")
warming_df = warming_df.dropna(subset=["year"])
warming_df["year"] = warming_df["year"].astype(int)
warming_df["warming_level"] = warming_df["Row"].astype(float)
warming_df["start_year"] = warming_df["year"] - 14
warming_df["end_year"] = warming_df["year"] + 15
print(f"Warming windows loaded: {len(warming_df)} rows")

def load_var_for_model(var, model, scenario, start_year, end_year, base_dir="./CMIP6/"):
    print(f"Loading {var} for {model} ({scenario}) {start_year}-{end_year}")

    folder = os.path.join(base_dir, var)
    try:
        files = os.listdir(folder)
    except Exception as e:
        raise FileNotFoundError(f"Cannot access folder {folder}: {e}")

    hist_files = [os.path.join(folder, f) for f in files
                  if f.startswith(f"{var}_day_{model}_historical")]
    fut_files = [os.path.join(folder, f) for f in files
                 if f.startswith(f"{var}_day_{model}_{scenario}")]

    print(f"Found {len(hist_files)} historical files, {len(fut_files)} future files")

    if not hist_files and not fut_files:
        raise FileNotFoundError(f"No files found for {var} {model} {scenario} in {folder}")

    try:
        if hist_files:
            print(f"Opening historical dataset ({len(hist_files)} files)...")
            ds_hist = xr.open_mfdataset(hist_files, combine='by_coords')[var]
        else:
            ds_hist = xr.DataArray()

        if fut_files:
            print(f"Opening future dataset ({len(fut_files)} files)...")
            ds_fut = xr.open_mfdataset(fut_files, combine='by_coords')[var]
        else:
            ds_fut = xr.DataArray()
    except Exception as e:
        raise IOError(f"Error opening mfdataset for {var} {model} {scenario}: {e}")

    if ds_hist.size > 0 and ds_fut.size > 0:
        ds_all = xr.concat([ds_hist, ds_fut], dim='time')
        print("Concatenated historical + future data")
    elif ds_hist.size > 0:
        ds_all = ds_hist
        print("Using historical data only")
    elif ds_fut.size > 0:
        ds_all = ds_fut
        print("Using future data only")
    else:
        raise ValueError(f"No valid data loaded for {var} {model} {scenario}.")

    del ds_hist, ds_fut

    buffer_days = 30
    start_date = pd.Timestamp(f"{start_year}-01-01") - pd.Timedelta(days=buffer_days)
    end_date = pd.Timestamp(f"{end_year}-12-31") + pd.Timedelta(days=buffer_days)

    print(f"Selecting time slice: {start_date} -> {end_date}")
    if isinstance(ds_all['time'].values[0], cftime.DatetimeNoLeap):
        start_date_cftime = cftime.DatetimeNoLeap(start_date.year, start_date.month, start_date.day)
        end_date_cftime = cftime.DatetimeNoLeap(end_date.year, end_date.month, end_date.day)
        ds_sliced = ds_all.sel(time=slice(start_date_cftime, end_date_cftime))
    else:
        ds_sliced = ds_all.sel(time=slice(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))

    print(f"Final dataset shape: {ds_sliced.shape}")

    if ds_sliced.size == 0:
        raise ValueError(f"No data found for {var} {model} {scenario} within {start_year}-{end_year}.")

    del ds_all
    return ds_sliced

scenarios = ["ssp585"]
all_models = [
    "CanESM5", "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR", "MIROC6",
    "MPI-ESM1-2-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0",
    "CMCC-CM2-SR5", "CMCC-ESM2", "IITM-ESM"
]
print(f"Processing models: {all_models}")

output_dir = "model_outputs/GridCellLevel_NoChunk/Corrected"
os.makedirs(output_dir, exist_ok=True)

for model in tqdm(all_models, desc="Models"):
    for scenario in scenarios:
        hist_start, hist_end = 1850, 1879
        print(f"\n--- Processing model {model} ({scenario}) ---")

        try:
            t0_load_hist = time.time()
            uas_hist = load_var_for_model("uas", model, scenario, hist_start, hist_end)
            vas_hist = load_var_for_model("vas", model, scenario, hist_start, hist_end)
            rsds_hist = load_var_for_model("rsds", model, scenario, hist_start, hist_end)
            tas_hist = load_var_for_model("tas", model, scenario, hist_start, hist_end)
            print(f"Historical data loaded in {time.time() - t0_load_hist:.2f}s")
        except Exception as e:
            tqdm.write(f"[{model}] Error loading historical data: {e}.")
            gc.collect()
            continue

        print("Computing wind_hist...")
        wind_hist = np.sqrt(uas_hist**2 + vas_hist**2)

        uas_hist_np = uas_hist.values
        vas_hist_np = vas_hist.values
        rsds_hist_np = rsds_hist.values
        tas_hist_np = tas_hist.values
        wind_hist_np = wind_hist.values

        relevant_windows = warming_df[warming_df["model"] == model.replace("-", "_")]
        print(f"Found {len(relevant_windows)} warming windows for {model}")

        for _, row in relevant_windows.iterrows():
            t0_wl_start = time.time()
            wl = row['warming_level']
            start, end = row['start_year'], row['end_year']
            print(f"Processing warming level {wl} ({start}-{end})")

            results = []

            try:
                uas = load_var_for_model("uas", model, scenario, start, end)
                vas = load_var_for_model("vas", model, scenario, start, end)
                rsds = load_var_for_model("rsds", model, scenario, start, end)
                tas = load_var_for_model("tas", model, scenario, start, end)

                wind = np.sqrt(uas**2 + vas**2)

                uas_np = uas.values
                vas_np = vas.values
                rsds_np = rsds.values
                tas_np = tas.values
                wind_np = wind.values
                lat_np = rsds.lat.values
                lon_np = rsds.lon.values

                print("Looping over grid cells...")
                for i in range(rsds_np.shape[1]):
                    for j in range(rsds_np.shape[2]):
                        results.append({
                            'model': model,
                            'scenario': scenario,
                            'warming_level': wl,
                            'period_start': start,
                            'period_end': end,
                            'lat': float(lat_np[i]),
                            'lon': float(lon_np[j]),
                            'uas_hist_mean': float(uas_hist_np[:, i, j].mean()),
                            'vas_hist_mean': float(vas_hist_np[:, i, j].mean()),
                            'rsds_hist_mean': float(rsds_hist_np[:, i, j].mean()),
                            'tas_hist_mean': float(tas_hist_np[:, i, j].mean()),
                            'wind_hist_mean': float(wind_hist_np[:, i, j].mean()),
                            'uas_mean': float(uas_np[:, i, j].mean()),
                            'vas_mean': float(vas_np[:, i, j].mean()),
                            'rsds_mean': float(rsds_np[:, i, j].mean()),
                            'tas_mean': float(tas_np[:, i, j].mean()),
                            'wind_mean': float(wind_np[:, i, j].mean()),
                            'diff_uas_mean': float(uas_np[:, i, j].mean() / uas_hist_np[:, i, j].mean()),
                            'diff_vas_mean': float(vas_np[:, i, j].mean() / vas_hist_np[:, i, j].mean()),
                            'diff_rsds_mean': float(rsds_np[:, i, j].mean() / rsds_hist_np[:, i, j].mean()),
                            'diff_tas_mean': float(tas_np[:, i, j].mean() / tas_hist_np[:, i, j].mean()),
                            'diff_wind_mean': float(wind_np[:, i, j].mean() / wind_hist_np[:, i, j].mean()),
                        })

                print(f"Finished looping over grid cells: {len(results)} results")

                df = pd.DataFrame(results)
                out_path = f"{output_dir}/metrics_{model}_WL{wl:.1f}.csv"
                df.to_csv(out_path, index=False)

                tqdm.write(f"[{model} | WL={wl}] Finished in {time.time() - t0_wl_start:.2f}s, saved {len(df)} rows")

                del uas, vas, rsds, tas, wind, results, df, uas_np, vas_np, rsds_np, tas_np, wind_np
                gc.collect()
            except Exception as e:
                tqdm.write(f"[{model} | WL={wl}] Error: {e}.")
                gc.collect()

        del uas_hist, vas_hist, rsds_hist, tas_hist, wind_hist, uas_hist_np, vas_hist_np
        gc.collect()

print("Processing complete.")