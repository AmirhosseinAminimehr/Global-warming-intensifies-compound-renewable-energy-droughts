import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
import pickle

INPUT_FILE = 'averaged_gridsCorrectedAllData_Normal_Final_CorrectedFormula.pkl'
SIGNIFICANCE_DATA_FILE = 'processed_snr_data_Final_CorrectedFormula.pkl'

metrics = {
    'ratio_duration_mean': (-10, 0, 20, 'Δduration (%)'),
    'ratio_frequency': (-30, 0, 60, 'Δfrequency (%)'),
    'ratio_intensity_mean': (-3, 0, 6, 'Δintensity (%)'),
    'ratio_drought_days': (-30, 0, 60, 'Δdrought days (%)')
}

COLOR_SCALE = {
    'ratio_duration_mean': (0, 20),
    'ratio_frequency': (0, 40),
    'ratio_intensity_mean': (0, 3),
    'ratio_drought_days': (0, 100)
}

with open(INPUT_FILE, 'rb') as f:
    data = pickle.load(f)

grid_lon = data['grid_lon']
grid_lat = data['grid_lat']
window_sizes = data['window_sizes']
warming_levels = data['warming_levels']
antarctica_lat_threshold = -60

with open(SIGNIFICANCE_DATA_FILE, 'rb') as f:
    significance_data = pickle.load(f)

all_model_grids = significance_data["all_model_grids"]

print("Data loaded successfully.")

print("Starting STD ratio maps...")

for metric, (_, _, _, label) in metrics.items():

    vmin, vmax = COLOR_SCALE[metric]

    nrows = len(window_sizes)
    ncols = len(warming_levels)

    fig = plt.figure(figsize=(4.5 * ncols, 2.6 * nrows))

    gs = GridSpec(
        nrows=nrows,
        ncols=ncols,
        hspace=0,
        wspace=0.02,
        figure=fig
    )

    subplot_idx = 0
    axes_grid = np.empty((nrows, ncols), dtype=object)
    im = None

    for r, ws in enumerate(window_sizes):
        for c, warming_level in enumerate(warming_levels):

            ax = fig.add_subplot(gs[r, c], projection=ccrs.PlateCarree())
            axes_grid[r, c] = ax

            grid_key = (ws, warming_level, metric)

            if grid_key not in all_model_grids:
                ax.text(0.5, 0.5, "No Data",
                        transform=ax.transAxes,
                        ha='center', va='center')
                continue

            model_dict = all_model_grids[grid_key]

            model_stack = np.stack(
                [arr for arr in model_dict.values()],
                axis=0
            )

            std_map = np.nanstd(model_stack, axis=0)

            print(f"{grid_key}: STD computed")

            im = ax.pcolormesh(
                grid_lon,
                grid_lat,
                std_map,
                cmap='viridis',
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree()
            )

            ax.set_extent(
                [-180, 180, antarctica_lat_threshold, 90],
                crs=ccrs.PlateCarree()
            )

            ax.add_feature(cfeature.OCEAN, color='white', zorder=1)
            ax.coastlines(linewidth=0.6, zorder=2)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.4, zorder=2)

            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(f'Warming level: {warming_level}°C', fontsize=12, pad=2)

            ax.text(
                0.01, 0.98,
                f'({chr(97 + subplot_idx)})',
                transform=ax.transAxes,
                fontsize=10,
                ha='left',
                va='top'
            )

            subplot_idx += 1

    fig.canvas.draw()

    for r, ws in enumerate(window_sizes):
        row_axes = axes_grid[r, :]
        positions = [ax.get_position() for ax in row_axes if ax is not None]

        if positions:
            y_center = (min(p.y0 for p in positions) +
                        max(p.y1 for p in positions)) / 2

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

            fig.text(
                0.045,
                y_center,
                window_names.get(ws, f'Window size {ws}'),
                va='center',
                ha='right',
                rotation='vertical',
                fontsize=12
            )

    if im is not None:
        cbar_ax = fig.add_axes([0.2, 0.23, 0.6, 0.025])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.ax.set_xlabel(f"Inter-model standard deviation of {label}")
        cbar.ax.tick_params(labelsize=8)

    plt.subplots_adjust(left=0.06, right=0.97, top=0.96, bottom=0.27)

    out_name = f'CorrectedFormulaPlots/Maps/STD_RATIO_{metric}_final.png'
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {out_name}")

print("All ratio STD maps completed.")