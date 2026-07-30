import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
import pickle

output_dir = 'processed_data'

plot_filename = 'CorrectedFormulaPlots/WindSolar/PercentChange_CorrectedFormula.png'

metrics = {
    'diff_rsds_mean': (-10, 10, 'Δsolar radiation (%)'),
    'diff_uas_mean': (-50, 50, 'Δzonal wind speed (%)'),
    'diff_vas_mean': (-50, 50, 'Δmeridional wind speed (%)')
}

def create_multi_metric_map_from_saved_data(output_dir, plot_filename):

    print(f"Loading data from '{output_dir}'...")

    try:
        with open(os.path.join(output_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
    except FileNotFoundError:
        print(f"metadata.pkl not found.")
        return

    warming_levels = metadata['warming_levels']
    antarctica_lat_threshold = metadata['antarctica_lat_threshold']
    model_counts = metadata['model_counts']
    averaged_grids_paths = metadata['averaged_grids_paths']
    metric_names = list(metrics.keys())

    try:
        grid_lon = np.load(os.path.join(output_dir, 'grid_lon.npy'))
        grid_lat = np.load(os.path.join(output_dir, 'grid_lat.npy'))
    except FileNotFoundError:
        print("Grid files missing.")
        return

    nrows = len(metrics)
    ncols = len(warming_levels)

    fig = plt.figure(figsize=(5 * ncols, 2.3 * nrows))

    dpi = fig.dpi
    pixel_inches = 2 / dpi

    gs = GridSpec(nrows=nrows, ncols=ncols + 2, figure=fig,
                  width_ratios=[0.08] + [1.0] * ncols + [0.08],
                  height_ratios=[10.0] * nrows,
                  wspace=pixel_inches, hspace=0.1)

    for r, metric in enumerate(metric_names):
        vmin, vmax, label = metrics[metric]
        im = None

        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        label_ax = fig.add_subplot(gs[r, 0])
        label_ax.axis('off')

        for c, warming_level in enumerate(warming_levels):

            ax = fig.add_subplot(gs[r, c + 1], projection=ccrs.PlateCarree())

            key = (metric, warming_level)
            grid_path = averaged_grids_paths.get(key)

            if model_counts[warming_level] > 0 and grid_path:
                try:
                    averaged_grid = np.load(grid_path)
                except FileNotFoundError:
                    ax.text(0.5, 0.5, 'File Missing',
                            transform=ax.transAxes, ha='center', va='center')
                    continue

                averaged_grid_percent = (averaged_grid - 1.0) * 100

                im = ax.pcolormesh(grid_lon, grid_lat,
                                   averaged_grid_percent,
                                   cmap='coolwarm', norm=norm,
                                   transform=ccrs.PlateCarree())
            else:
                ax.text(0.5, 0.5, 'No Data',
                        transform=ax.transAxes, ha='center', va='center')

            ax.set_extent([-180, 180, antarctica_lat_threshold, 90],
                          crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.OCEAN, color='white', zorder=1)
            ax.coastlines(linewidth=0.6, zorder=2)
            ax.add_feature(cfeature.BORDERS,
                           linestyle=':', linewidth=0.4, zorder=2)

            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(f'Warming level: {warming_level}°C',
                             fontsize=13, pad=12)

            subplot_label = chr(97 + r * ncols + c)
            ax.text(0.02, 0.98, f'({subplot_label})',
                    transform=ax.transAxes,
                    fontsize=10, ha='left', va='top')

        if im is not None:
            cbar_ax = fig.add_subplot(gs[r, ncols + 1])

            cbar = fig.colorbar(im, cax=cbar_ax,
                                orientation='vertical', norm=norm)

            cbar.set_label(label, fontsize=11)

            ticks = np.linspace(vmin, vmax, 5)
            cbar.set_ticks(ticks)

            tick_labels = [f'{tick:.0f}' if tick != 0 else '0'
                           for tick in ticks]
            cbar.set_ticklabels(tick_labels)

            cbar.ax.tick_params(labelsize=9)

    fig.canvas.draw()

    plt.subplots_adjust(left=0.07, right=0.93,
                        top=0.94, bottom=0.06)

    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\n Saved {plot_filename}")


if __name__ == '__main__':
    create_multi_metric_map_from_saved_data(output_dir, plot_filename)