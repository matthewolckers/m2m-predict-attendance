# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: m2m
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter


from my_functions import load_data, create_descriptive_stats_table
# -

(train, test, comp,
y_train_miss_first, X_train, 
y_test_miss_first, X_test, 
y_comp_miss_first, X_comp) = load_data()

create_descriptive_stats_table()

train.describe().transpose()


test.describe().transpose()


comp.describe().transpose()


def plot_miss_first_by_month_across_sources(train, test, comp, save_path=None):
    # Ensure started_time is datetime
    for df in [train, test, comp]:
        df['started_time'] = pd.to_datetime(df['started_time'])

    # Add source column
    train['source'] = 'Train'
    test['source'] = 'Test'
    comp['source'] = 'Comparison'

    # Concatenate dataframes
    all_df = pd.concat([train, test, comp], ignore_index=True)

    # Create month column
    all_df['month'] = all_df['started_time'].dt.to_period('M').dt.to_timestamp()

    # Group by source and month, calculate mean miss_first
    monthly_stats = (
        all_df.groupby(['source', 'month'])['miss_first']
        .mean()
        .reset_index()
    )

    # Set up seaborn style for publication
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.3)
    plt.figure(figsize=(14, 7))

    # Lineplot with source as hue
    ax = sns.lineplot(
        data=monthly_stats,
        x='month',
        y='miss_first',
        hue='source',
        marker='o',
        linewidth=2.5,
        palette='Set2'
    )

    # Set y-axis limits between 0 and 1
    ax.set_ylim(0, 1)

    # Title and labels
    # ax.set_title('Share of clients that missed their first follow-up appointment', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=20, labelpad=10)
    ax.set_ylabel('Share of clients', fontsize=20, labelpad=10)

    # Format x-axis
    ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
    plt.xticks(rotation=45, ha='right')

    # Set font size for axis tick labels
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # Legend
    ax.legend(title='Data Source', fontsize=18, title_fontsize=15)

    # Tight layout for publication
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


plot_miss_first_by_month_across_sources(train, test, comp, save_path='../figures/miss_first_by_month_across_sources.png')
