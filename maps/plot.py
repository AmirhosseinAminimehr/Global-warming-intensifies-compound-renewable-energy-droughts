import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
import pickle
import os
from scipy import stats

INPUT_FILE = 'averaged_gridsCorrectedAllData_Normal_Final_CorrectedFormula.pkl'
SIGNIFICANCE_DATA_FILE = 'processed_snr_data_Final_CorrectedFormula.pkl'

ALPHA = 0.05

metrics = {
    'diff_duration_mean': (-0.25, 0, 0.50, 'Δduration (days)'),
    'diff_frequency': (-150, 0, 300, 'Δfrequency (count)'),
    'diff_intensity_mean': (-0.05, 0, 0.15, 'Δintensity (units)'),
    'diff_drought_days': (-250, 0, 600, 'Δdrought days (days)'),
    'ratio_duration_mean': (-20, 0, 20, 'Δduration (%)'),
    'ratio_frequency': (-60, 0, 60, 'Δfrequency (%)'),
    'ratio_intensity_mean': (-6, 0, 6, 'Δintensity (%)'),
    'ratio_drought_days': (-30, 0, 60, 'Δdrought days (%)')
}

try:
    with open(INPUT_FILE, 'rb') as f:
        data = pickle.load(f)

    averaged_grids = data['averaged_grids']
    grid_lon = data['grid_lon']
    grid_lat = data['grid_lat']
    window_sizes = data['window_sizes']
    warming_levels = data['warming_levels']
    antarctica_lat_threshold = -60

    print(f"Successfully loaded data from {INPUT_FILE}")

except FileNotFoundError:
    print(f"Data file '{INPUT_FILE}' not found. Please run data.py first.")
    exit()
except Exception as e:
    print(f"Error loading data from file: {e}")
    exit()

try:
    with open(SIGNIFICANCE_DATA_FILE, 'rb') as f:
        significance_data = pickle.load(f)

    all_model_grids = significance_data["all_model_grids"]
    final_mask = significance_data["final_mask"]

    print(f"Successfully loaded significance data from {SIGNIFICANCE_DATA_FILE}")

except FileNotFoundError:
    print(f"Significance data file '{SIGNIFICANCE_DATA_FILE}' not found.")
    all_model_grids = None
    final_mask = None
except Exception as e:
    print(f"Error loading significance data: {e}.")
    all_model_grids = None
    final_mask = None

def calculate_significance_mask(model_grids_list, metric, final_mask):
    if model_grids_list is None or len(model_grids_list) == 0:
        return None

    stacked_grids = np.stack(model_grids_list, axis=2)

    if metric.startswith('diff_'):
        delta_grids = stacked_grids
    else:
        delta_grids = stacked_grids

    ensemble_mean = np.nanmean(delta_grids, axis=2)
    inter_model_std = np.nanstd(delta_grids, axis=2)

    N = delta_grids.shape[2]
    significance_mask = np.full(ensemble_mean.shape, False, dtype=bool)

    if N > 1:
        degrees_of_freedom = N - 1

        with np.errstate(divide='ignore', invalid='ignore'):
            t_statistic = ensemble_mean / (inter_model_std / np.sqrt(N))

        p_value = stats.t.sf(np.abs(t_statistic), degrees_of_freedom) * 2

        significance_mask = p_value < ALPHA

        if final_mask is not None:
            significance_mask[final_mask] = False

        significance_mask[np.isnan(ensemble_mean)] = False

    return significance_mask

def add_crosshatch(ax, significance_mask, grid_lon, grid_lat):
    if significance_mask is None or not np.any(significance_mask):
        return

    lon_flat = grid_lon.flatten()
    lat_flat = grid_lat.flatten()
    mask_flat = significance_mask.flatten()

    sig_lons = lon_flat[mask_flat]
    sig_lats = lat_flat[mask_flat]

    if len(sig_lons) == 0:
        return

    spacing = 3

    for i in range(0, len(sig_lons), spacing):
        if i < len(sig_lons):
            lon, lat = sig_lons[i], sig_lats[i]

            ax.plot(
                [lon - 0.5, lon + 0.5],
                [lat, lat],
                color='#000000',
                linewidth=0.4,
                alpha=0.8,
                transform=ccrs.PlateCarree()
            )

            ax.plot(
                [lon, lon],
                [lat - 0.5, lat + 0.5],
                color='#000000',
                linewidth=0.4,
                alpha=0.8,
                transform=ccrs.PlateCarree()
            )

print("Starting plotting...")

for metric, (vmin, vcenter, vmax, label) in metrics.items():
    nrows = len(window_sizes)
    ncols = len(warming_levels)

    fig = plt.figure(figsize=(4.5 * ncols, 2.6 * nrows))

    dpi = fig.dpi
    pixel_inches = 2 / dpi

    gs = GridSpec(
        nrows=nrows,
        ncols=ncols,
        hspace=0,
        wspace=pixel_inches,
        figure=fig
    )

    subplot_idx = 0
    im = None
    axes_grid = np.empty((nrows, ncols), dtype=object)

    for r, ws in enumerate(window_sizes):
        for c, warming_level in enumerate(warming_levels):
            ax = fig.add_subplot(gs[r, c], projection=ccrs.PlateCarree())
            axes_grid[r, c] = ax

            grid_key = (ws, warming_level, metric)

            if grid_key not in averaged_grids:
                print(f"No processed data found for {grid_key}.")
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center')
            else:
                data_array = averaged_grids[grid_key]

                valid_data = np.sum(~np.isnan(data_array))
                total_cells = data_array.size
                coverage = (valid_data / total_cells) * 100
                print(f"{grid_key}: {valid_data}/{total_cells} cells ({coverage:.1f}% coverage)")

                norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

                im = ax.pcolormesh(
                    grid_lon,
                    grid_lat,
                    data_array,
                    cmap='coolwarm',
                    norm=norm,
                    transform=ccrs.PlateCarree()
                )

                if all_model_grids is not None:
                    significance_grid_key = (ws, warming_level, metric)

                    if significance_grid_key in all_model_grids:
                        model_grids_list = list(all_model_grids[significance_grid_key].values())

                        significance_mask = calculate_significance_mask(
                            model_grids_list,
                            metric,
                            final_mask
                        )

                        add_crosshatch(
                            ax,
                            significance_mask,
                            grid_lon,
                            grid_lat
                        )

                        if significance_mask is not None:
                            total_pixels = np.sum(~np.isnan(data_array))
                            significant_pixels = np.sum(significance_mask)

                            if total_pixels > 0:
                                significance_percentage = (significant_pixels / total_pixels) * 100

                                print(
                                    f"  Significance: {significance_percentage:.1f}% "
                                    f"({significant_pixels}/{total_pixels} pixels)"
                                )

            ax.set_extent(
                [-180, 180, antarctica_lat_threshold, 90],
                crs=ccrs.PlateCarree()
            )

            ax.add_feature(cfeature.OCEAN, color='white', zorder=1)

            ax.coastlines(linewidth=0.6, zorder=2)

            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(
                    f'Warming level: {warming_level}°C',
                    fontsize=12,
                    pad=2
                )

            subplot_label = chr(97 + subplot_idx)

            ax.text(
                0.01,
                0.98,
                f'({subplot_label})',
                transform=ax.transAxes,
                fontsize=10,
                ha='left',
                va='top'
            )

            subplot_idx += 1

    fig.canvas.draw()

    for r, ws in enumerate(window_sizes):
        row_axes = axes_grid[r, :]
        row_positions = [ax.get_position() for ax in row_axes if ax is not None]

        if row_positions:
            y_min = min(pos.y0 for pos in row_positions)
            y_max = max(pos.y1 for pos in row_positions)
            y_center = (y_min + y_max) / 2

            if r == 0:
                y_center += 0.10
            elif r == 1:
                y_center += 0.12
            elif r == 2:
                y_center += 0.14

            window_names = {
                'WS1': 'Window size 1',
                'WS3': 'Window size 3',
                'WS5': 'Window size 5'
            }

            window_label = window_names.get(ws, f'Window size {ws}')

            fig.text(
                0.045,
                y_center,
                window_label,
                va='center',
                ha='right',
                rotation='vertical',
                fontsize=12
            )

    if im is not None:
        cbar_ax = fig.add_axes([0.2, 0.23, 0.6, 0.025])

        cbar = fig.colorbar(
            im,
            cax=cbar_ax,
            orientation='horizontal'
        )

        cbar.ax.set_xlabel(label, labelpad=2)
        cbar.ax.tick_params(labelsize=8)

    plt.subplots_adjust(
        left=0.06,
        right=0.97,
        top=0.96,
        bottom=0.27
    )

    out_name = f'CorrectedFormulaPlots/Maps/Combined_{metric}_WS_with_significance_CorrectedFormula.png'

    plt.savefig(
        out_name,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(fig)

    print(f"Saved {out_name}")

print("All plots completed!")