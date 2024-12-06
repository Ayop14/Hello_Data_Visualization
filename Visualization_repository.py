# Import data processing libraries
import pandas as pd
import numpy as np
from itertools import chain

# Import visualization libraries
import seaborn as sns
from joypy import joyplot
import plotly.express as px

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter

# Import math libraries
from scipy.stats import probplot, gaussian_kde, t

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from statsmodels.graphics.mosaicplot import mosaic
from statsmodels.tsa.seasonal import seasonal_decompose


def histogram(data, bins, ax):
    '''
        data: pandas series to plot
        bins: number of bins to make the histogram with
        ax: pyplot ax to plot the visualization in
    '''
    assert type(data) == pd.Series, 'Array Input is not a pandas Series'
    # Visualize weights using a histogram
    ax.hist(data, bins = bins)
    # Plot 0.25, 0.75 quartiles as vertical bars
    ax.axvline(data.quantile(0.75), color='red')
    ax.axvline(data.quantile(0.25), color='red')

    # Format the axes
    ax.set_xlabel(data.name)
    ax.set_ylabel('Value counts')
    ax.set_title('Histogram plot')


def polar(data, times_around):
    '''
    Highly customizable plot, this just works as a base
        data: pandas series to plot
        times_around: number of times to make the data go around in the polar axes
    '''
    
    # Not included as a parameter because it requires additional parameters in creation
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    # Create angles from 0 to 2π (one time around)
    angles = np.linspace(0, 2 * np.pi * times_around, len(data))
    ax.plot(angles, data, marker='o', markersize=4)
    ax.set_title("Polar plot")

    return fig, ax


def scatterplot(x, y, ax):
    '''
        x: pandas series to plot on x axis
        y: pandas series to plot on y axis
        ax: pyplot axes to make the scatterplot
    '''
    # Make the scatterplot. Simple
    ax.scatter(x, y, s=50, edgecolor='k')

    # Add axis labels and title
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    ax.set_title('Scatterplot')
    

def color_scatterplot(x, y, color_variable, ax):
    '''
        x: pandas series to plot on x axis
        y: pandas series to plot on y axis
        color_variable: pandas series to plot as a continous variable using color
        ax: pyplot ax to plot the visualization in
    '''
    # Obtain min and max for scaling
    color_min = color_variable.min()
    color_max = color_variable.max()

    # Obtain colors from continuous variable
    color_normalized = Normalize(vmin=color_min, vmax=color_max) # Normalize values
    cmap = plt.cm.coolwarm

    # Make the scatterplot
    scatter = ax.scatter(x, y, c=color_variable, cmap=cmap, norm=color_normalized, s=50, edgecolor='k')

    # Add a colorbar to map colors to values
    # Add a colorbar that reflects real values
    colorbar = plt.colorbar(plt.cm.ScalarMappable(norm=color_normalized, cmap=cmap), ax=ax)
    colorbar.set_label(color_variable.name)

    # Add axis labels and title
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    ax.set_title('Scatterplot with color-encoded continous variable')

def buble_chart(x, y, size_variable, ax):
    '''
        x: pandas series to plot on x axis
        y: pandas series to plot on y axis
        size_variable: pandas series to plot as a continous variable using size
        ax: pyplot ax to plot the visualization in
    '''
    # Obtain min and max for scailing
    size_min = size_variable.min()
    size_max = size_variable.max()

    # Obtain colors from continous variable
    sizes = (size_variable - size_min) / (size_max - size_min) * 500 + 10  # Normalize values and *500 for adequate size

    # Scatterplot with torque, weight and seat height
    ax.scatter(x, y, s=sizes, edgecolor='k')

    # Add axis labels and title
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    ax.set_title('Scatterplot with color-encoded continous variable')


def grouped_bars_plot(matrix_data, ax):
    '''
    :param matrix_data: Matrix where every row is a category (group of bars), each column is a subcategory/feature to measure about the group (individual bar, the blue one for example)
        make sure rows have string index with the name of the feature they are measuring
    :param ax: axis to make the plot
    '''
    # pivot_table with aggfunc=size is very useful for this format

    # Set the positions for the bars on the x-axis
    bar_width = 0.2  # Width of the bars

    n_categories = len(matrix_data.index)
    n_subcategories = len(matrix_data.columns)

    # Create grouped bars plot
    x = np.arange(n_categories)  # Positions for the groups (categories)

    # Plot each subgroup in the bars, with slight shifts for each bar within a group
    for i in range(n_subcategories):
        ax.bar(x + i * bar_width, matrix_data.iloc[:, i], width=bar_width, label=matrix_data.columns[i])

    # Ensure integer y-axis ticks
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Create the legend for each of the features
    ax.legend()

    # Set the x-axis labels and ticks
    ax.set_xticks(x + bar_width * (n_subcategories - 1) / 2, matrix_data.index)  # Center the ticks

    # Add labels and title
    ax.set_ylabel('Value Count')
    ax.set_title('Grouped bar plot')


def stacked_bars_plot(matrix_data, ax):
    '''
    :param matrix_data: Matrix where every row is a category (group of bars), each column is a subcategory/feature to measure about the group (individual bar, the blue one for example)
        make sure rows have string index with the name of the feature they are measuring. shape ngroups, n_features
    :param ax: axis to make the plot
    '''
    # pivot_table with aggfunc=size is very useful for this format
    # Define variables
    n_categories = len(matrix_data.index)
    n_subcategories = len(matrix_data.columns)
    x = np.arange(n_categories)  # Positions for the groups (categories)

    # Initialize the bottom of the bars to zero (stacking from the bottom)
    bottom = np.zeros(n_categories)

    # Create the stacked bar chart
    for i in range(n_subcategories):
        ax.bar(x, matrix_data.iloc[:, i], bottom=bottom, label=matrix_data.columns[i])
        bottom += matrix_data.iloc[:, i]  # Update the bottom position for stacking

    # Ensure integer y-axis ticks
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Create the legend for each of the features
    ax.legend()

    # Set the x-axis ticks and labels
    ax.set_xticks(x, matrix_data.index)

    # Add labels and title
    ax.set_ylabel('Value Count')
    ax.set_title('Grouped bar plot')


def heatmap_plot(matrix_data, ax):
    '''
        :param matrix_data: Matrix where every row is a category (group of bars), each column is a subcategory/feature to measure about the group (individual bar, the blue one for example)
            make sure rows have string index with the name of the feature they are measuring. shape ngroups, n_features
        :param ax: axis to make the plot
    '''
    # Create the heatmap using imshow
    cax = ax.imshow(matrix_data, cmap='Reds', aspect='auto')

    # Set row and column labels
    ax.set_xticks(np.arange(len(matrix_data.columns)), matrix_data.columns)
    ax.set_yticks(np.arange(len(matrix_data.index)), matrix_data.index)

    # Rotate the x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Set as text the number of values in each spot
    # Annotate the heatmap with the cell values
    for i in range(matrix_data.shape[0]):  # Iterate over rows
        for j in range(matrix_data.shape[1]):  # Iterate over columns
            ax.text(j, i, f'{matrix_data.iloc[i, j]:.0f}',
                    ha='center', va='center', color='black')

    # Set labels and title
    ax.set_title('Heatmap plot')


def density_plot(data, ax):
    '''
        data: pandas series to plot
        ax: pyplot ax to plot the visualization in
    '''
    # Create density plot with seaborn
    sns.kdeplot(data, ax=ax, fill=True, color='steelblue', linewidth=2)

    # Customize the plot
    ax.set_title(f'Density Plot', fontsize=14)
    ax.set_xlabel(data.name)
    ax.set_ylabel('Density')

def cumulative_density_plot(data, ax):
    '''
        data: pandas series to plot
        ax: pyplot ax to plot the visualization in
    '''
    # Make a cumulative density plot
    sorted_feature = np.sort(data)

    # Compute the cumulative values (normalized to 1)
    cumulative_values = np.arange(1, len(sorted_feature) + 1) / len(sorted_feature)

    # Plot the cumulative density function (CDF)
    ax.step(sorted_feature, cumulative_values, color='steelblue')

    # Set yticks manually
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # Set grid lines for y axis
    ax.grid(which='major', axis='y', linestyle='--', linewidth=1, color='gray')

    # Customize the plot with title and labels
    ax.set_title(f'Cumulative Density Plot')
    ax.set_xlabel(data.name)
    ax.set_ylabel('Cumulative Density')


def quantile_quantile_plot(data, ax):
    '''
        data: pandas series to plot
        ax: pyplot ax to plot the visualization in
    '''
    # Quantile quantile plots are used to compare the data distribution to a certain distribution. In this case normal
    # Remove null values
    clean_feature = data.dropna()
    # Generate Q-Q plot data using scipy's probplot function
    (theoretical_quantiles, sample_quantiles), (slope, intercept, _) = probplot(clean_feature, dist="norm") # Comparison with a normal distribution

    # Plot the Q-Q plot (theoretical quantiles vs. sample quantiles)
    ax.scatter(theoretical_quantiles, sample_quantiles, color='blue', label='Sample Quantiles')

    # Plot the 45-degree line using the slope and intercept from probplot
    ax.plot(theoretical_quantiles, slope * theoretical_quantiles + intercept, color='red', linestyle='--',
                    label='y = x (reference)')

    # Set a grid for clearer visualization
    ax.grid(True)

    # Add legend
    ax.legend()

    # Customize the plot with title and labels
    ax.set_title(f'Q-Q Plot')
    ax.set_xlabel('Theoretical Quantiles - Standard deviations from the mean')
    ax.set_ylabel('Sample Quantiles')


def box_plot(matrix_data, ax):
    '''
        :param matrix_data: Matrix where every row is a category (group of bars), each column is a subcategory/feature to measure about the group (individual bar, the blue one for example)
        :param ax: axis to make the plot
    '''
    # Transform data for the plot
    columns= matrix_data.columns
    distributions = [matrix_data[col].dropna() for col in matrix_data.columns] # Filter null values

    # Plot the boxplot
    ax.boxplot(distributions, tick_labels=columns,
               meanline=None, # If showmeans=True, show mean as a line spanning the full width of the box
               showmeans=None, # Show the arithmetic means
               showcaps=None, # Caps at the end of the whiskers
               showbox=None, # Show the central box
               showfliers=None) # Show the outliers beyond the caps
    # Rotate labels
    ax.set_xticks(np.arange(0,len(columns)) + 1, columns, rotation=45)

    # Set grid lines
    ax.grid(axis='y', which='major', linestyle='--', lw=1, color='gray', alpha=0.4)

    # Add title and labels
    ax.set_title('Boxplot of Multiple Distributions')
    ax.set_ylabel('Value units')


def violin_plot(matrix_data, ax):
    # Transform data for the plot
    columns = matrix_data.columns
    distributions = [matrix_data[col].dropna() for col in matrix_data.columns]  # Filter null values

    # Plot the violin plot
    ax.violinplot(distributions,
                  showmeans=False,
                  showextrema=True,
                  showmedians=False,
                  quantiles=None)

    # Set the labels for the x-axis
    ax.set_xticks([i + 1 for i in range(len(distributions))], columns, rotation=45)

    # Set grid lines
    ax.grid(axis='y', which='major', linestyle='--', color='gray', alpha=0.4)

    # Add title and labels
    ax.set_title('Violin Plot of Multiple Distribution')
    ax.set_ylabel('Values')


from project.read_data import obtain_dataset

# obtain data
# Pivot data so bike types are column names, and consumption are values
df = obtain_dataset()
df = df.pivot(columns='bike type', values='consumption')

fig, ax = plt.subplots()
violin_plot(df, ax)
fig.tight_layout()
fig.show()




    


