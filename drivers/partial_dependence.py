import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.inspection import PartialDependenceDisplay
import os
import tqdm
import numpy as np

prepared_data_file = "prepared_data_CorrectedFormula.pkl"
trained_models_file = "trained_models_global_Pos_Neg_CorrectedFormula.pkl"

print("--- Plotting Results --")

try:
    mean_df = pd.read_pickle(prepared_data_file)
    with open(trained_models_file, 'rb') as f:
        trained_models = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

input_vars = ['diff_rsds_mean', 'diff_uas_mean', 'diff_vas_mean', 'warming_level', 'lon', 'lat']
output_vars = ['ratio_intensity_mean', 'ratio_duration_mean', 'ratio_frequency']
features_to_plot = ['diff_rsds_mean', 'diff_uas_mean', 'diff_vas_mean']

variable_labels = {
    'diff_rsds_mean': 'Solar radiation',
    'diff_uas_mean': 'Zonal wind speed',
    'diff_vas_mean': 'Meridional wind speed',
    'ratio_intensity_mean': 'Intensity',
    'ratio_duration_mean': 'Duration',
    'ratio_frequency': 'Frequency'
}

output_plots_dir = "CorrectedFormulaPlots/Drivers"
os.makedirs(output_plots_dir, exist_ok=True)

regime_config = {
    'positive': {'color': 'blue', 'label': 'Positive change'},
    'negative': {'color': 'red', 'label': 'Negative change'}
}

fig, axes = plt.subplots(
    nrows=len(output_vars),
    ncols=len(features_to_plot),
    figsize=(4 * len(features_to_plot), 4.5 * len(output_vars)),
    squeeze=False
)

panel_counter = 0

for row_idx, output_var in enumerate(tqdm.tqdm(output_vars, desc="Generating Combined PDPs")):

    row_y_values = []

    for regime in ['negative', 'positive']:

        if regime not in trained_models.get(output_var, {}):
            continue

        model = trained_models[output_var][regime]['model']

        data_subset = mean_df.dropna(subset=input_vars + output_vars)

        if regime == 'positive':
            data_subset = data_subset[data_subset[output_var] > 1]
        else:
            data_subset = data_subset[data_subset[output_var] < 1]

        if data_subset.empty:
            continue

        X = data_subset[input_vars]

        display = PartialDependenceDisplay.from_estimator(
            model,
            X,
            features=features_to_plot,
            kind='average',
            ax=axes[row_idx],
            line_kw={
                'color': regime_config[regime]['color'],
                'linewidth': 2.5,
                'label': regime_config[regime]['label']
            }
        )

        for pd_result in display.pd_results:
            row_y_values.extend(pd_result['average'].flatten())

    for col_idx, feature_name in enumerate(features_to_plot):
        sub_ax = axes[row_idx, col_idx]

        if row_y_values:
            ymin, ymax = min(row_y_values), max(row_y_values)
            padding = (ymax - ymin) * 0.1 if ymax != ymin else 0.1
            sub_ax.set_ylim(ymin - padding, ymax + padding)

        sub_ax.set_title('')

        if row_idx == len(output_vars) - 1:
            sub_ax.set_xlabel(variable_labels.get(feature_name, feature_name), fontsize=11)
        else:
            sub_ax.set_xlabel('')

        if col_idx == 0:
            sub_ax.set_ylabel(variable_labels.get(output_var), fontsize=11)
        else:
            sub_ax.set_ylabel('')

        sub_ax.axhline(1, linestyle='--', linewidth=1, alpha=0.7, color='black')
        sub_ax.axvline(1, linestyle='--', linewidth=1, alpha=0.7, color='black')

        sub_ax.grid(False)

        label = chr(97 + panel_counter)
        sub_ax.text(
            0.02, 0.98,
            f'({label})',
            transform=sub_ax.transAxes,
            ha='left',
            va='top',
            fontsize=11
        )
        panel_counter += 1

        if row_idx == 0 and col_idx == 0:
            handles, labels = sub_ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            sub_ax.legend(
                by_label.values(),
                by_label.keys(),
                loc='upper right',
                fontsize=9
            )
        else:
            legend = sub_ax.get_legend()
            if legend:
                legend.remove()

fig.suptitle(
    'Partial Dependence Plots: Positive vs Negative',
    fontsize=18,
    y=1.02
)

plt.tight_layout(rect=[0, 0, 1, 0.98])

plt.savefig(
    f'{output_plots_dir}/PDP_Combined.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"Plot saved to {output_plots_dir}.")