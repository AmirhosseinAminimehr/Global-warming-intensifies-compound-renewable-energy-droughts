import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

all_models = [
    "CanESM5", "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR", "MIROC6",
    "MPI-ESM1-2-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0",
    "CMCC-CM2-SR5", "CMCC-ESM2", "IITM-ESM"
]

input_dir = "location_outputs"

target_locations = [
    "Central_Africa",
    "Northern_North_America",
    "East_Asia",
    "South_Asia"
]

periods = ["Historical", "GWL_1.5", "GWL_2.0", "GWL_3.0"]

ensemble_data = {
    loc: {p: {"PV": [], "Wind": []} for p in periods}
    for loc in target_locations
}

print("Pooling data across all models...")

for model in all_models:
    safe_model_name = model.replace("/", "_").replace(" ", "_")
    file_path = f"{input_dir}/Extracted_Power_{safe_model_name}.pkl"

    if not os.path.exists(file_path):
        print(f"Data for {model} missing.")
        continue

    with open(file_path, "rb") as f:
        model_data = pickle.load(f)

    for loc in target_locations:
        if loc in model_data:
            for p in periods:
                if p in model_data[loc]:
                    ensemble_data[loc][p]["PV"].append(model_data[loc][p]["PV"])
                    ensemble_data[loc][p]["Wind"].append(model_data[loc][p]["Wind"])

for loc in target_locations:
    for p in periods:
        if ensemble_data[loc][p]["PV"]:
            ensemble_data[loc][p]["PV"] = np.concatenate(
                ensemble_data[loc][p]["PV"]
            )
            ensemble_data[loc][p]["Wind"] = np.concatenate(
                ensemble_data[loc][p]["Wind"]
            )
        else:
            ensemble_data[loc][p]["PV"] = np.array([])
            ensemble_data[loc][p]["Wind"] = np.array([])

FILTER_ZEROS = True

period_styles = {
    "Historical": {
        "color": "#1f77b4",
        "linestyle": "-",
        "label": "Preindustrial period",
        "fill": True,
        "alpha": 0.12
    },
    "GWL_1.5": {
        "color": "#2ca02c",
        "linestyle": "--",
        "label": "Global warming level 1.5°C",
        "fill": False,
        "alpha": 1.0
    },
    "GWL_2.0": {
        "color": "#ff7f0e",
        "linestyle": "-.",
        "label": "Global warming level 2.0°C",
        "fill": False,
        "alpha": 1.0
    },
    "GWL_3.0": {
        "color": "#d62728",
        "linestyle": "-",
        "label": "Global warming level 3.0°C",
        "fill": False,
        "alpha": 1.0
    }
}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, loc_name in enumerate(target_locations):

    ax = axes[idx]
    loc_data = ensemble_data[loc_name]

    all_x, all_y = [], []
    has_valid_2d_data = False
    cleaned_periods = {}

    for period in periods:

        x = np.asarray(loc_data[period]["PV"], dtype=float)
        y = np.asarray(loc_data[period]["Wind"], dtype=float)

        valid_mask = ~np.isnan(x) & ~np.isnan(y)

        if FILTER_ZEROS:
            valid_mask = valid_mask & (x > 0.001) & (y > 0.001)

        x_clean = x[valid_mask]
        y_clean = y[valid_mask]

        if len(x_clean) > 50:
            cleaned_periods[period] = (x_clean, y_clean)

            all_x.extend(x_clean)
            all_y.extend(y_clean)

            if np.var(y_clean) > 1e-5 and np.var(x_clean) > 1e-5:
                has_valid_2d_data = True

    for period, style in period_styles.items():

        if period in cleaned_periods:

            x_plot, y_plot = cleaned_periods[period]

            try:
                if has_valid_2d_data:

                    sns.kdeplot(
                        x=x_plot,
                        y=y_plot,
                        ax=ax,
                        fill=style["fill"],
                        alpha=style["alpha"],
                        color=style["color"],
                        linestyles=style["linestyle"],
                        linewidths=2.4 if not style["fill"] else 0,
                        levels=6,
                        thresh=0.08,
                        bw_adjust=1.6,
                        common_norm=False
                    )

                else:

                    sns.histplot(
                        x=x_plot,
                        ax=ax,
                        element="step",
                        fill=False,
                        color=style["color"],
                        ls=style["linestyle"],
                        lw=2
                    )

            except Exception as e:
                print(f"Skipping plot layer for {loc_name} {period}: {e}")

    clean_title = loc_name.replace("_", " ")

    ax.set_title(
        clean_title,
        fontsize=15,
        pad=12
    )

    ax.text(
        0.02,
        0.98,
        f"({chr(97 + idx)})",
        transform=ax.transAxes,
        fontsize=14,
        va='top',
        ha='left'
    )

    ax.set_xlabel("Solar power potential", fontsize=12)
    ax.set_ylabel("Wind power potential", fontsize=12)

    ax.grid(True, linestyle="--", alpha=0.5)

    if all_x and all_y:

        pad_x = (max(all_x) - min(all_x)) * 0.1 or 0.05
        pad_y = (max(all_y) - min(all_y)) * 0.1 or 0.05

        ax.set_xlim(
            max(0, min(all_x) - pad_x),
            min(1.0, max(all_x) + pad_x)
        )

        if has_valid_2d_data:

            ax.set_ylim(
                max(0, min(all_y) - pad_y),
                min(1.0, max(all_y) + pad_y)
            )

        else:
            ax.set_ylabel("")

handles = [
    plt.Line2D(
        [0], [0],
        color=s["color"],
        linestyle=s["linestyle"],
        lw=2.5
    )
    for s in period_styles.values()
]

labels = [s["label"] for s in period_styles.values()]

fig.legend(
    handles,
    labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.96),
    ncol=4,
    fontsize=12,
    frameon=True
)

plt.suptitle("", fontsize=18, y=0.99)

plt.tight_layout(rect=[0, 0, 1, 0.93])

output_filename = "Solar_Wind_Dependency_Changes1.png"

plt.savefig(
    output_filename,
    dpi=300
)

print(f"Successfully generated plot: {output_filename}")

plt.show()