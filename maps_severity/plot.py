import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
import os
from glob import glob
import pickle
import numpy as np

input_data_folder = './prepared_data_new'
antarctica_lat_threshold = -60

plotting_metrics = {
    'duration_mean_percent': {
        'label': 'Δduration (%)',
        'categories': {
            'moderate': {'vmin': -20, 'vcenter': 0, 'vmax': 20},
            'severe': {'vmin': -20, 'vcenter': 0, 'vmax': 20},
            'extreme': {'vmin': -20, 'vcenter': 0, 'vmax': 20}
        }
    },
    'frequency_percent': {
        'label': 'Δfrequency (%)',
        'categories': {
            'moderate': {'vmin': -60, 'vcenter': 0, 'vmax': 60},
            'severe': {'vmin': -60, 'vcenter': 0, 'vmax': 60},
            'extreme': {'vmin': -60, 'vcenter': 0, 'vmax': 60}
        }
    },
    'intensity_mean_percent': {
        'label': 'Δintensity (%)',
        'categories': {
            'moderate': {'vmin': -6, 'vcenter': 0, 'vmax': 6},
            'severe': {'vmin': -6, 'vcenter': 0, 'vmax': 6},
            'extreme': {'vmin': -6, 'vcenter': 0, 'vmax': 6}
        }
    }
}

data_files = glob(os.path.join(input_data_folder, '*_data_Final.pkl'))

if not data_files:
    print(f"No prepared data files found in {input_data_folder}. Run data.py first.")
    exit()

for data_file in data_files:
    with open(data_file, 'rb') as f_in:
        metric_data = pickle.load(f_in)

    grid_lon = metric_data['grid_lon']
    grid_lat = metric_data['grid_lat']
    warming_levels = metric_data['warming_levels']
    categories = metric_data['categories']
    p_props = metric_data['p_props']
    gridded_diffs = metric_data['gridded_diffs']
    metric_type = metric_data.get('metric_type', 'percentage')

    filename = os.path.basename(data_file)
    primary_metric = filename.split('_WS')[0]
    ws = 1

    print(f"\n--- Plotting Metric: {primary_metric} (WS {ws}, Type: {metric_type}) ---")

    if primary_metric not in plotting_metrics:
        print(f"No plotting configuration found for metric '{primary_metric}'.")
        plotting_config = {
            'label': f'Δ{primary_metric.split("_")[0].capitalize()} (%)',
            'categories': {
                'moderate': {'vmin': -5, 'vcenter': 0, 'vmax': 15},
                'severe': {'vmin': -5, 'vcenter': 0, 'vmax': 15},
                'extreme': {'vmin': -5, 'vcenter': 0, 'vmax': 15}
            }
        }
    else:
        plotting_config = plotting_metrics[primary_metric]

    nrows = len(categories)
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

    im = None
    axes_grid = np.empty((nrows, ncols), dtype=object)

    subplot_idx = 0

    for r, category in enumerate(categories):
        for c, warming_level in enumerate(warming_levels):
            ax = fig.add_subplot(gs[r, c], projection=ccrs.PlateCarree())
            axes_grid[r, c] = ax

            mean_diff = gridded_diffs.get((category, warming_level))

            vmin = plotting_config['categories'][category]['vmin']
            vcenter = plotting_config['categories'][category]['vcenter']
            vmax = plotting_config['categories'][category]['vmax']

            if mean_diff is None or np.isnan(mean_diff).all():
                ax.set_visible(False)
                print(f"Skipping plot for WL={warming_level:.1f} | Category={category} (No data).")
                subplot_idx += 1
                continue

            plot_data = mean_diff

            ax.set_extent(
                [-180, 180, antarctica_lat_threshold, 90],
                crs=ccrs.PlateCarree()
            )

            ax.add_feature(cfeature.OCEAN, color='white', zorder=1)
            ax.coastlines(linewidth=0.6, zorder=2)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.4, zorder=2)

            cmap = 'coolwarm'
            norm = TwoSlopeNorm(
                vmin=vmin,
                vcenter=vcenter,
                vmax=vmax
            )

            im = ax.pcolormesh(
                grid_lon,
                grid_lat,
                plot_data,
                cmap=cmap,
                norm=norm,
                transform=ccrs.PlateCarree()
            )

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

    for r, category in enumerate(categories):
        row_axes = axes_grid[r, :]
        row_positions = [
            ax.get_position()
            for ax in row_axes
            if ax is not None
        ]

        if row_positions:
            y_min = min(pos.y0 for pos in row_positions)
            y_max = max(pos.y1 for pos in row_positions)

            if r == 0:
                y_center = (y_min + y_max) / 2 + 0.1
            elif r == 1:
                y_center = (y_min + y_max) / 2 + 0.12
            else:
                y_center = (y_min + y_max) / 2 + 0.14

            category_label = category.capitalize()

            fig.text(
                0.045,
                y_center,
                category_label,
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

        cbar.ax.set_xlabel(
            plotting_config["label"],
            labelpad=2,
            fontsize=10
        )

        cbar.ax.tick_params(labelsize=8)

    plt.subplots_adjust(
        left=0.06,
        right=0.97,
        top=0.96,
        bottom=0.27
    )

    out_name = (
        f'CorrectedFormulaPlots/Maps/'
        f'Combined_{primary_metric}_WS{ws}_PercentChange_NewLogic.png'
    )

    plt.savefig(
        out_name,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(fig)

    print(f"Saved {out_name}")

print("\nPlotting Complete.")