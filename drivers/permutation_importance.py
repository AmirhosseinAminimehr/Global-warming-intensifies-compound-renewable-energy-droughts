import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.inspection import permutation_importance
import os
import tqdm

def constrained_sum_round(values, target=100):
    values = np.array(values, dtype=float)

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    total_values = values.sum()

    if total_values <= 0:
        return [0] * len(values)

    values = values / total_values * target

    floors = np.floor(values).astype(int)

    diff = int(round(target - floors.sum()))

    remainders = values - floors
    order = np.argsort(remainders)[::-1]

    diff = min(diff, len(values))

    for i in range(diff):
        floors[order[i]] += 1

    return floors.tolist()

prepared_data_file = "prepared_data_CorrectedFormula.pkl"
trained_models_file = "trained_models_global_Pos_Neg_CorrectedFormula.pkl"

print("--- Permutation Importance ---")

try:
    mean_df = pd.read_pickle(prepared_data_file)
    with open(trained_models_file, 'rb') as f:
        trained_models = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

input_vars = ['diff_rsds_mean', 'diff_uas_mean', 'diff_vas_mean', 'warming_level', "lon", "lat"]
output_vars = ['ratio_intensity_mean', 'ratio_duration_mean', 'ratio_frequency']

warming_levels = sorted(mean_df['warming_level'].unique())

variable_labels = {
    'diff_rsds_mean': r'Δsolar radiation',
    'diff_uas_mean': r'Δzonal wind speed',
    'diff_vas_mean': r'Δmeridional wind speed',
    'ratio_intensity_mean': r'Δintensity',
    'ratio_duration_mean': r'Δduration',
    'ratio_frequency': r'Δfrequency',
}

variable_colors_raw = {
    'diff_vas_mean': '#2c7fb8',
    'diff_uas_mean': '#7fcdbb',
    'diff_rsds_mean': '#edf8b1',
}

ordered_features = ['diff_vas_mean', 'diff_uas_mean', 'diff_rsds_mean']
plot_colors = [variable_colors_raw[var] for var in ordered_features]

output_plots_dir = "CorrectedFormulaPlots/Drivers"
os.makedirs(output_plots_dir, exist_ok=True)

for regime in ['positive', 'negative']:

    print(f"\nProcessing {regime} regime...")
    normalized_importance_data = {}

    for wl in tqdm.tqdm(warming_levels, desc=f"{regime} importance"):
        normalized_importance_data[wl] = {}
        wl_data = mean_df[mean_df['warming_level'] == wl].dropna(subset=input_vars + output_vars)

        if wl_data.empty:
            continue

        for output_var in output_vars:
            if regime not in trained_models.get(output_var, {}):
                continue

            if regime == 'positive':
                subset = wl_data[wl_data[output_var] > 1]
            else:
                subset = wl_data[wl_data[output_var] < 1]

            if len(subset) < 20:
                continue

            X = subset[input_vars]
            y = subset[output_var]
            model = trained_models[output_var][regime]['model']

            result = permutation_importance(
                model, X, y, scoring='r2', n_repeats=10, random_state=30, n_jobs=-1
            )

            importance = pd.Series(result.importances_mean, index=X.columns)
            importance = importance.drop('warming_level', errors='ignore')
            importance = importance.clip(lower=0)
            importance = importance[ordered_features]

            total = importance.sum()

            if total > 1e-6:
                importance_pct = (importance / total) * 100
                correction = 100 - importance_pct.sum()
                importance_pct[importance_pct.idxmax()] += correction
            else:
                importance_pct = pd.Series([0] * len(ordered_features), index=ordered_features)

            normalized_importance_data[wl][output_var] = importance_pct

    stack_data = []
    for wl, wl_data in normalized_importance_data.items():
        for output_var, series in wl_data.items():
            row = series.to_dict()
            row['warming_level'] = wl
            row['output_var'] = output_var
            stack_data.append(row)

    if not stack_data:
        print(f"No data for {regime} regime.")
        continue

    df_plot = pd.DataFrame(stack_data)

    df_plot = df_plot.rename(columns={
        'diff_vas_mean': variable_labels['diff_vas_mean'],
        'diff_uas_mean': variable_labels['diff_uas_mean'],
        'diff_rsds_mean': variable_labels['diff_rsds_mean']
    })

    stack_columns = [
        variable_labels['diff_vas_mean'],
        variable_labels['diff_uas_mean'],
        variable_labels['diff_rsds_mean']
    ]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(output_vars),
        figsize=(5 * len(output_vars), 6),
        squeeze=True
    )

    if regime == 'negative':
        panel_letters = ['(d)', '(e)', '(f)']
    else:
        panel_letters = ['(a)', '(b)', '(c)']

    for i, output_var in enumerate(output_vars):
        ax = axes[i]
        plot_df = df_plot[df_plot['output_var'] == output_var]

        if plot_df.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue

        plot_df = plot_df.set_index('warming_level')

        plot_df[stack_columns].plot(
            kind='bar',
            stacked=True,
            ax=ax,
            color=plot_colors,
            legend=False
        )

        ax.text(
            0.02,
            0.95,
            panel_letters[i],
            transform=ax.transAxes,
            fontsize=16,
            va='top',
            ha='left'
        )

        ax.set_title(variable_labels[output_var], fontsize=14)
        ax.set_xlabel(r'Warming level ($^\circ$C)', fontsize=14)

        if i == 0:
            ax.set_ylabel('Permutation importance (%)', fontsize=14)

        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=0)

        bottoms = np.zeros(len(plot_df))

        for col in stack_columns:
            values = plot_df[col].values

            for x, val in enumerate(values):
                if val >= 5:
                    ax.text(
                        x,
                        bottoms[x] + val / 2,
                        f'{round(val):.0f}%',
                        ha='center',
                        va='center',
                        fontsize=12,
                        color='black'
                    )
            bottoms += values

    handles, labels = axes[-1].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        fontsize=12,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(stack_columns)
    )

    fig.suptitle(
        f'Driver importance ({regime} changes)',
        fontsize=16,
        y=1.0
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fig.subplots_adjust(top=0.80)

    plt.savefig(
        f'{output_plots_dir}/Importance_{regime}.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

print("\nPlots created.")