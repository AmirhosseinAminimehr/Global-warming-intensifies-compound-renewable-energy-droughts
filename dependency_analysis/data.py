import os
import re
import time
import xarray as xr
import numpy as np
import pandas as pd
import cftime
import gc
import pickle
import scipy.stats

target_locations = {
    "Central_Africa": {"lat": 0.0, "lon": 20.0},
    "Northern_North_America": {"lat": 55.0, "lon": -100.0},
    "East_Asia": {"lat": 35.0, "lon": 115.0},
    "South_Asia": {"lat": 22.0, "lon": 77.0}
}

warming_levels_to_extract = [1.5, 2.0, 3.0]
historical_start, historical_end = 1850, 1879

warming_df = pd.read_csv("./Yr_gw_ssp585_30.csv")
warming_df = warming_df.melt(id_vars=["Row"], var_name="model", value_name="year").dropna()
warming_df["year"] = warming_df["year"].astype(int)
warming_df["warming_level"] = warming_df["Row"].astype(float)

all_models = [
    "CanESM5", "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR", "MIROC6",
    "MPI-ESM1-2-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0",
    "CMCC-CM2-SR5", "CMCC-ESM2", "IITM-ESM"
]

def load_all_years_for_model(var, model, scenario, base_dir="./CMIP6/"):
    folder = os.path.join(base_dir, var)
    hist_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith(f"{var}_day_{model}_historical")]
    fut_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith(f"{var}_day_{model}_{scenario}")]

    if hist_files:
        ds_hist = xr.open_mfdataset(hist_files, combine='by_coords', engine="netcdf4")[var]
        if var == "tas":
            ds_hist = ds_hist - 273.15
    else:
        ds_hist = xr.DataArray()

    if fut_files:
        ds_fut = xr.open_mfdataset(fut_files, combine='by_coords', engine="netcdf4")[var]
        if var == "tas":
            ds_fut = ds_fut - 273.15
    else:
        ds_fut = xr.DataArray()

    if ds_hist.size > 0 and ds_fut.size > 0:
        return xr.concat([ds_hist, ds_fut], dim='time')
    return ds_hist if ds_hist.size > 0 else ds_fut

def extract_period_data(full_data, start_year, end_year):
    start_date = pd.Timestamp(f"{start_year}-01-01")
    end_date = pd.Timestamp(f"{end_year}-12-31")

    if isinstance(full_data['time'].values[0], cftime.datetime):
        cftime_class = full_data['time'].values[0].__class__
        start_date_cftime = cftime_class(start_date.year, start_date.month, start_date.day)
        end_date_cftime = cftime_class(end_date.year, end_date.month, end_date.day)
        return full_data.sel(time=slice(start_date_cftime, end_date_cftime))
    else:
        return full_data.sel(time=slice(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))

output_dir = "location_outputs"
os.makedirs(output_dir, exist_ok=True)

for model in all_models:
    print(f"Processing extraction for model: {model}")
    try:
        uas = load_all_years_for_model("uas", model, "ssp585").astype("float32")
        vas = load_all_years_for_model("vas", model, "ssp585").astype("float32")
        rsds = load_all_years_for_model("rsds", model, "ssp585").astype("float32")
        tas = load_all_years_for_model("tas", model, "ssp585").astype("float32")
    except Exception as e:
        print(f"Skipping {model} due to loading error: {e}")
        continue

    wind_speed = np.sqrt(uas**2 + vas**2)
    Tcell = 4.3 + 0.943 * tas + 0.028 * rsds - 1.528 * wind_speed
    pr = 1 - 0.005 * (Tcell - 25)
    PV_pot = pr * (rsds / 1000)

    wind_speed_100 = wind_speed * ((80 / 10) ** 0.143)
    W_pot = xr.where(
        wind_speed_100 < 3.5, 0,
        xr.where(
            wind_speed_100 < 13,
            (wind_speed_100**3 - 3.5**3) / (13**3 - 3.5**3),
            xr.where(wind_speed_100 < 25, 1, 0)
        )
    )

    model_extracted_data = {}

    for loc_name, coords in target_locations.items():
        lon_mapped = coords["lon"] if coords["lon"] >= 0 or "lon" not in PV_pot.coords else coords["lon"]
        if "lon" in PV_pot.coords and PV_pot.lon.max() > 180 and coords["lon"] < 0:
            lon_mapped = coords["lon"] + 360

        pv_point = PV_pot.sel(lat=coords["lat"], lon=lon_mapped, method="nearest")
        w_point = W_pot.sel(lat=coords["lat"], lon=lon_mapped, method="nearest")

        loc_data = {}

        pv_hist = extract_period_data(pv_point, historical_start, historical_end).values
        w_hist = extract_period_data(w_point, historical_start, historical_end).values
        loc_data["Historical"] = {"PV": pv_hist, "Wind": w_hist}

        model_key_in_csv = model.replace("-", "_")
        relevant_windows = warming_df[warming_df["model"] == model_key_in_csv]

        for gwl in warming_levels_to_extract:
            row = relevant_windows[relevant_windows["warming_level"] == gwl]
            if not row.empty:
                center_year = int(row.iloc[0]["year"])
                start_yr, end_yr = center_year - 14, center_year + 15

                try:
                    pv_gwl = extract_period_data(pv_point, start_yr, end_yr).values
                    w_gwl = extract_period_data(w_point, start_yr, end_yr).values
                    loc_data[f"GWL_{gwl}"] = {"PV": pv_gwl, "Wind": w_gwl}
                except Exception as e:
                    print(f"Could not extract GWL {gwl} for {model} at {loc_name}: {e}")
            else:
                print(f"GWL {gwl} window not found for model {model} in CSV.")

        model_extracted_data[loc_name] = loc_data

    safe_model_name = model.replace("/", "_").replace(" ", "_")
    with open(f"{output_dir}/Extracted_Power_{safe_model_name}.pkl", "wb") as f:
        pickle.dump(model_extracted_data, f)

    del uas, vas, rsds, tas, wind_speed, Tcell, pr, PV_pot, wind_speed_100, W_pot
    gc.collect()

print("Extraction completely finished.")