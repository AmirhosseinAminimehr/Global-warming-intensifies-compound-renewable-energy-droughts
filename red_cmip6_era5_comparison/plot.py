import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

output_plot_folder = './plots/CorrectedFormula'

metrics_to_plot = {
    'frequency': ('Frequency', 'Events'),
    'duration_mean': ('Duration', 'Days'),
    'intensity_mean': ('Intensity', 'SREI')
}

antarctica_lat_threshold = -60
grid_resolution = 1.0

era5_csv = os.path.join(output_plot_folder, "ERA5_gridded_metrics.csv")
cmip6_csv = os.path.join(output_plot_folder, "CMIP6_gridded_metrics.csv")

era5_df = pd.read_csv(era5_csv)
cmip6_df = pd.read_csv(cmip6_csv)

lon = np.arange(-180, 180 + grid_resolution, grid_resolution)
lat = np.arange(-90, 90 + grid_resolution, grid_resolution)
grid_lon, grid_lat = np.meshgrid(lon, lat)

def df_to_grid(df, metric):
    return df.pivot(index="lat", columns="lon", values=metric).values

era5_gridded = {m: df_to_grid(era5_df, m) for m in metrics_to_plot}
cmip6_gridded = {m: df_to_grid(cmip6_df, m) for m in metrics_to_plot}

for dataset in [era5_gridded, cmip6_gridded]:
    nan_mask = np.isnan(dataset["intensity_mean"])
    dataset["frequency"][nan_mask] = np.nan

metric_ranges = {
    "frequency": (400, 900),
    "duration_mean": (0, 4),
    "intensity_mean": (1.5, 1.9)
}

fig, axes = plt.subplots(
    nrows=len(metrics_to_plot),
    ncols=2,
    figsize=(12, 10),
    subplot_kw={"projection": ccrs.PlateCarree()},
    constrained_layout=True
)

subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
label_idx = 0

for i, (metric, (axis_label, cbar_label)) in enumerate(metrics_to_plot.items()):
    vmin, vmax = metric_ranges[metric]

    ax1 = axes[i, 0]
    im1 = ax1.pcolormesh(
        grid_lon,
        grid_lat,
        cmip6_gridded[metric],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        zorder=1
    )

    ax2 = axes[i, 1]
    im2 = ax2.pcolormesh(
        grid_lon,
        grid_lat,
        era5_gridded[metric],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        zorder=1
    )

    for ax in [ax1, ax2]:
        ax.set_extent([-180, 180, antarctica_lat_threshold, 90], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=3)
        ax.coastlines(linewidth=0.6, zorder=4)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.4, zorder=4)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.text(
            0.02,
            0.96,
            subplot_labels[label_idx],
            transform=ax.transAxes,
            fontsize=12,
            va='top',
            ha='left',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )
        label_idx += 1

    ax1.set_ylabel(axis_label, fontsize=12, rotation=90, labelpad=20)

    cbar = fig.colorbar(
        im1,
        ax=[ax1, ax2],
        orientation="horizontal",
        fraction=0.05,
        pad=0.08
    )
    cbar.set_label(cbar_label, fontsize=11)

axes[0, 0].set_title("CMIP6 (Multi-model mean)", fontsize=12)
axes[0, 1].set_title("ERA5 (Reanalysis)", fontsize=12)

out_name = os.path.join(
    output_plot_folder,
    "CMIP6_ERA5_Drought_Metrics_Comparison_LandOnly.png"
)

plt.savefig(out_name, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f" Saved plot to {out_name}")