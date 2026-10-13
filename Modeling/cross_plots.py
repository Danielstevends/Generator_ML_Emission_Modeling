import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os
import argparse

parser = argparse.ArgumentParser(description="Cross-site validation for generator data")
parser.add_argument('--tune_hyperparams', action='store_true', help="Enable hyperparameter tuning")
parser.add_argument('--root_path', type=str, default="/Users/sylviaimanirakiza/Documents/GitHub/", help="Root path of the project")
# Parse command line arguments
args = parser.parse_args()
# Check if hyperparameter tuning is enabled
tune_hyperparams = args.tune_hyperparams
#check if rootpath exists
root_path = args.root_path
if os.path.exists(root_path):
    print(f"Root path exists {root_path}")
else:
    print(f"Root path does not exist: {root_path}. Please check the path.")

print(f"Hyperparameter tuning is {'enabled' if tune_hyperparams else 'disabled'}.")

savepath = f"{root_path}Generator_ML_Emission_Modeling/Modeling/summer25_results/cross_site_models/"
os.makedirs(savepath, exist_ok=True)


results = pd.read_csv(f"{root_path}Generator_ML_Emission_Modeling//Modeling/summer25_results/cross_site_models/results_crossval.csv")
df = results


mpl.rcParams.update({
    'font.family': 'Arial',
    'axes.titlesize': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
    'figure.titlesize': 22
})
sns.set_style("white")
sns.despine(trim=True)

if tune_hyperparams: # If hyperparameter tuning is enabled, filter for tuned models
    models_of_interest = ['RandomForest_tuned', 'XGBoost_tuned']
else:
    # If hyperparameter tuning is disabled, filter for base models
    models_of_interest = ['RandomForest', 'XGBoost']

df_filtered = df[df['model'].isin(models_of_interest)].copy()
df_filtered['abs_gen_error'] = df_filtered['Percentage generator on-time detection error'].abs()

# Define categories
feature_groups = sorted(df_filtered['featuregroup'].unique())
print(feature_groups)
facilities = sorted(df_filtered['heldout_site'].unique())

# Color palette for feature groups
palette_fg = dict(zip(feature_groups, sns.color_palette("colorblind", len(feature_groups))))

# Save path setup
os.makedirs(savepath, exist_ok=True)

fig, axes = plt.subplots(1, len(models_of_interest), figsize=(7 * len(models_of_interest), 9), sharey=True)

if len(models_of_interest) == 1:
    axes = [axes]

for ax, model in zip(axes, models_of_interest):
    df_model = df_filtered[df_filtered['model'] == model]

    summary = df_model.groupby(['heldout_site', 'featuregroup'])['abs_gen_error'].agg(['mean', 'std']).reset_index()

    pivot_mean = summary.pivot(index='heldout_site', columns='featuregroup', values='mean')
    pivot_std = summary.pivot(index='heldout_site', columns='featuregroup', values='std')

    x = np.arange(len(facilities))
    total_width = 0.8
    n_bars = len(feature_groups)
    bar_width = total_width / n_bars

    for i, fg in enumerate(feature_groups):
        means = pivot_mean[fg].reindex(facilities)
        stds = pivot_std[fg].reindex(facilities)
        bars = ax.bar(x + i*bar_width, means, width=bar_width, yerr=stds, capsize=5,
                      label=fg, color=palette_fg[fg], edgecolor='k')

    ax.set_title(f"{model}", fontsize=22)
    ax.set_xticks(x + total_width/2 - bar_width/2)
    ax.set_xticklabels(facilities, fontsize=20, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylim(0, None)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.minorticks_on()
    if ax == axes[0]:
        ax.set_ylabel("Generator On-Time Detection Error (\%)", fontsize=20, fontweight='bold')

    ax.legend_.remove() if ax.get_legend() else None

handles, labels = axes[0].get_legend_handles_labels()

# Wrap long labels at a certain character limit
def wrap_label(label, width=24):
    return '\n'.join(label[i:i+width] for i in range(0, len(label), width))

wrapped_labels = [wrap_label(lab, width=24) for lab in labels]

# Update labels with wrapped versions
labels = wrapped_labels
fig.legend(
    handles,
    labels,
    title="Feature Group",
    title_fontsize=18,
    fontsize=18,
    loc='center left',
    bbox_to_anchor=(1, 0.5),
    frameon=True
)

# Shared x-axis label
fig.text(0.5, 0.01, 'Held Out Facility', ha='center', fontsize=20, fontweight='bold')
#add padding


# Layout adjustments (make room for left-side legend)
plt.tight_layout()  # Leaves more space on the left
#add more space
# rect=[0.2, 0.05, 1, 0.95]
plt.savefig(f'{savepath}/Gen_error_With_HMM features_Cross site.png', dpi=300, bbox_inches='tight')
plt.show()



# Global Plot Styling
mpl.rcParams.update({
    'font.family': 'Arial',
    'axes.titlesize': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
    'figure.titlesize': 22
})
sns.set_style("white")
sns.despine(trim=True)

# Data filtering
models_of_interest = ['RandomForest', 'XGBoost']
df_filtered = df[df['model'].isin(models_of_interest)].copy()
# df_filtered['abs_gen_error'] = df_filtered['Percentage generator on-time detection error'].abs()

# Define categories
feature_groups = sorted(df_filtered['featuregroup'].unique())
print(feature_groups)
facilities = sorted(df_filtered['heldout_site'].unique())

# Color palette for feature groups
palette_fg = dict(zip(feature_groups, sns.color_palette("colorblind", len(feature_groups))))

# Save path setup
os.makedirs(savepath, exist_ok=True)

fig, axes = plt.subplots(1, len(models_of_interest), figsize=(7 * len(models_of_interest), 9), sharey=True)

if len(models_of_interest) == 1:
    axes = [axes]  # Ensure iterable

for ax, model in zip(axes, models_of_interest):
    df_model = df_filtered[df_filtered['model'] == model]

    # Aggregate data: mean and std of abs_gen_error grouped by facility & feature group
    summary = df_model.groupby(['heldout_site', 'featuregroup'])['f1 score'].agg(['mean', 'std']).reset_index()

    # Pivot for grouped barplot (mean as height)
    pivot_mean = summary.pivot(index='heldout_site', columns='featuregroup', values='mean')
    pivot_std = summary.pivot(index='heldout_site', columns='featuregroup', values='std')

    # Facilities as x axis positions
    x = np.arange(len(facilities))
    total_width = 0.8
    n_bars = len(feature_groups)
    bar_width = total_width / n_bars

    for i, fg in enumerate(feature_groups):
        means = pivot_mean[fg].reindex(facilities)
        stds = pivot_std[fg].reindex(facilities)
        bars = ax.bar(x + i*bar_width, means, width=bar_width, yerr=stds, capsize=5,
                      label=fg, color=palette_fg[fg], edgecolor='k')

    ax.set_title(f"{model}", fontsize=22)
    ax.set_xticks(x + total_width/2 - bar_width/2)
    ax.set_xticklabels(facilities, fontsize=20, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylim(0, None)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.minorticks_on()
    if ax == axes[0]:
        ax.set_ylabel("F1 score", fontsize=20, fontweight='bold')

    # Remove individual subplot legends
    ax.legend_.remove() if ax.get_legend() else None

# Shared legend on the left side

handles, labels = axes[0].get_legend_handles_labels()

# Wrap long labels at a certain character limit
def wrap_label(label, width=24):
    return '\n'.join(label[i:i+width] for i in range(0, len(label), width))

wrapped_labels = [wrap_label(lab, width=24) for lab in labels]

# Update labels with wrapped versions
labels = wrapped_labels
fig.legend(
    handles,
    labels,
    title="Feature Group",
    title_fontsize=18,
    fontsize=18,
    loc='center left',
    bbox_to_anchor=(1, 0.5),
    frameon=True
)
#  bbox_to_anchor=(0.01, 0.5),




# Shared x-axis label
fig.text(0.5, 0.01, 'Held Out Facility', ha='center', fontsize=20, fontweight='bold')

# Layout adjustments (make room for left-side legend)
# plt.tight_layout(rect=[0.15, 0.05, 1, 0.95])
plt.tight_layout()  # Leaves more space on the left
# rect=[0.2, 0.05, 1, 0.95]
plt.savefig(f'{savepath}/F1_score_With_HMM features_Cross site.png', dpi=300, bbox_inches='tight')
plt.show()


