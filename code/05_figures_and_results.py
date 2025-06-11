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


train.dtypes

train['started_time'] = pd.to_datetime(train['started_time'])
train["month"] = train['started_time'].dt.to_period('M')

train.groupby('month').agg({'miss_first': ['mean', 'count']})


def plot_monthly_miss_first(df, save_path=None):
    # Convert started_time to datetime if it's not already
    df['started_time'] = pd.to_datetime(df['started_time'])

    # Group by month and calculate mean of miss_first
    monthly_miss_first = df.groupby(df['started_time'].dt.to_period('M'))['miss_first'].mean().reset_index()
    monthly_miss_first['started_time'] = monthly_miss_first['started_time'].dt.to_timestamp()

    print(monthly_miss_first)

    # Set up the plot style
    sns.set_style('whitegrid')
    sns.set_context("paper", font_scale=1.2)
    sns.set_palette("deep")

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the data
    sns.lineplot(data=monthly_miss_first, x='started_time', y='miss_first', ax=ax, linewidth=2, color='#1f77b4')

    # Customize the plot
    ax.set_title('Monthly Average of Miss First', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=14, labelpad=10)
    ax.set_ylabel('Average Miss First', fontsize=14, labelpad=10)

    # # Format x-axis ticks
    # ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    # plt.xticks(rotation=45, ha='right')


    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path)

    # Show the plot
    plt.show()
