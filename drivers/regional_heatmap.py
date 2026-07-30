import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import geopandas as gpd
from shapely.vectorized import contains
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

IPCC_SHP = "IPCC-WGI-reference-regions-v4.shp"
DROUGHT_FILE = 'averaged_gridsCorrectedAllData_Normal_Final_CorrectedFormula.pkl'
WIND_SOLAR_META = 'processed_data/metadata.pkl'
GRID_LON_FILE = 'processed_data/grid_lon.npy'
GRID_LAT_FILE = 'processed_data/grid_lat.npy'

warming_levels = [1.5, 2.0, 3.0]


continent_groups = {
    'North\nAmerica': ['NWN', 'NEN', 'WNA', 'CNA', 'ENA'],
    'Central\nAmerica': ['NCA', 'SCA', 'CAR'],
    'South\nAmerica': ['NWS', 'NSA', 'NES', 'SAM', 'SWS', 'SES', 'SSA'],
    'Europe': ['NEU', 'WCE', 'EEU', 'MED'],
    'Africa': ['SAH', 'WAF', 'CAF', 'NEAF', 'SEAF', 'WSAF', 'ESAF', 'MDG'],
    'Asia': ['RAR', 'WSB', 'ESB', 'RFE', 'WCA', 'ECA', 'TIB', 'EAS', 'ARP', 'SAS', 'SEA'],
    'Australasia': ['NAU', 'CAU', 'EAU', 'SAU', 'NZ']
}

ordered_regions = []
for c in continent_groups.values():
    ordered_regions.extend(c)


print("Loading data...")

with open(DROUGHT_FILE, 'rb') as f:
    drought_data = pickle.load(f)

drought_grids = drought_data['averaged_grids']
grid_lon = drought_data['grid_lon']
grid_lat = drought_data['grid_lat']

with open(WIND_SOLAR_META, 'rb') as f:
    meta = pickle.load(f)

paths = meta['averaged_grids_paths']

grid_lon_ws = np.load(GRID_LON_FILE)
grid_lat_ws = np.load(GRID_LAT_FILE)


print("Loading IPCC regions...")
ipcc = gpd.read_file(IPCC_SHP)


print("Creating region masks...")

region_masks = {}
for reg in ordered_regions:
    geom = ipcc[ipcc['Acronym'] == reg].geometry.values[0]
    region_masks[reg] = contains(geom, grid_lon, grid_lat)


print("Aggregating to regions...")

n_reg = len(ordered_regions)
n_wl = len(warming_levels)

drought_mat = np.zeros((n_reg, n_wl))
solar_mat = np.zeros_like(drought_mat)
u_mat = np.zeros_like(drought_mat)
v_mat = np.zeros_like(drought_mat)

for i, reg in enumerate(ordered_regions):
    mask = region_masks[reg]

    for j, wl in enumerate(warming_levels):

        dgrid = drought_grids[('WS3', wl, 'ratio_frequency')]
        vals = dgrid[mask]
        drought_mat[i, j] = np.nanmean(vals) if np.any(~np.isnan(vals)) else np.nan

        sgrid = np.load(paths[('diff_rsds_mean', wl)])
        sgrid[sgrid == 0] = np.nan
        vals = ((sgrid - 1) * 100)[mask]
        solar_mat[i, j] = np.nanmean(vals) if np.any(~np.isnan(vals)) else np.nan

        ugrid = np.load(paths[('diff_uas_mean', wl)])
        ugrid[ugrid == 0] = np.nan
        vals = ((ugrid - 1) * 100)[mask]
        u_mat[i, j] = np.nanmean(vals) if np.any(~np.isnan(vals)) else np.nan

        vgrid = np.load(paths[('diff_vas_mean', wl)])
        vgrid[vgrid == 0] = np.nan
        vals = ((vgrid - 1) * 100)[mask]
        v_mat[i, j] = np.nanmean(vals) if np.any(~np.isnan(vals)) else np.nan


variables = [
    {'id': 'u', 'label': 'Δzonal wind', 'vmin': -80, 'vmax': 80},
    {'id': 'v', 'label': 'Δmeridional wind', 'vmin': -80, 'vmax': 80},
    {'id': 'solar', 'label': 'Δsolar radiation', 'vmin': -10, 'vmax': 10},
    {'id': 'drought', 'label': 'Δfrequency of RED ', 'vmin': -60, 'vmax': 60}
]

heat_data = {
    'u': u_mat,
    'v': v_mat,
    'solar': solar_mat,
    'drought': drought_mat
}

sig_data = {k: np.zeros_like(v, dtype=bool) for k, v in heat_data.items()}


def draw_continent_brackets(ax):
    idx = 0
    for cont, regs in continent_groups.items():
        start = idx
        end = idx + len(regs) - 1

        ax.plot([-0.08, -0.08], [start-0.4, end+0.4], 'k')
        ax.plot([-0.08, 0], [start-0.4, start-0.4], 'k')
        ax.plot([-0.08, 0], [end+0.4, end+0.4], 'k')

        ax.text(-0.32, (start+end)/2, cont,
                ha='right', va='center', fontsize=14)

        idx += len(regs)


def plot_heatmap():
    fig = plt.figure(figsize=(16, 15))
    gs = GridSpec(1, 5, width_ratios=[0.4, 1, 1, 1, 1])

    ax_left = fig.add_subplot(gs[0, 0])
    ax_left.set_ylim(n_reg - 0.5, -0.5)
    ax_left.set_xlim(-0.15, 0.2)

    ax_left.set_yticks(np.arange(n_reg))
    ax_left.set_yticklabels(ordered_regions, fontsize=11)
    ax_left.yaxis.tick_right()
    ax_left.set_xticks([])

    ax_left.tick_params(axis='y', pad=-3, length=0)

    for spine in ax_left.spines.values():
        spine.set_visible(False)

    draw_continent_brackets(ax_left)

    subplot_labels = ['(a)', '(b)', '(c)', '(d)']

    for i, var in enumerate(variables):
        ax = fig.add_subplot(gs[0, i + 1])
        data = heat_data[var['id']]

        im = ax.imshow(
            data,
            cmap='coolwarm',
            norm=TwoSlopeNorm(vmin=var['vmin'], vcenter=0, vmax=var['vmax']),
            aspect='auto'
        )

        ax.text(
            0.03, 0.995, subplot_labels[i],
            transform=ax.transAxes,
            fontsize=13,
            va='top',
            ha='left',
        )

        for r in range(n_reg):
            for c in range(n_wl):
                if sig_data[var['id']][r, c]:
                    ax.plot(c, r, 'ko', markersize=2)

        ax.set_xticks(range(n_wl))
        ax.set_xticklabels([f"{wl}°C" for wl in warming_levels],
                           rotation=45, ha='right')

        ax.set_title(var['label'], fontsize=15, pad=10)
        ax.set_yticks([])

        ax.set_xticks(np.arange(-0.5, n_wl, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_reg, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="both", length=0)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="8%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax, orientation='vertical')
        if i == 3:
            cbar.set_label('Change (%)', fontsize=12)
        cbar.ax.tick_params(labelsize=8)

    plt.subplots_adjust(left=0.1, right=0.92, top=0.92, bottom=0.08, wspace=0.2)

    plt.savefig("CorrectedFormulaPlots/Drivers/IPCC_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_heatmap()