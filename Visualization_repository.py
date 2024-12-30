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


def stackedbars_plot(matrix_data, ax):
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


def box_plot(data, ax):
    '''
        :param data: pandas DataFrame or Series where every row is a category (group of bars), each column is a subcategory/feature to measure about the group (individual bar, the blue one for example)
        :param ax: axis to make the plot
    '''
    # Transform data for the plot
    if type(data) is pd.DataFrame:
        columns= data.columns
        distributions = [data[col].dropna() for col in data.columns] # Filter null values
    elif type(data) is pd.Series:
        columns = [data.name]
        distributions = [data.dropna()]
    else:
        raise TypeError('Input must be pandas DataFrame or Series')

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
    ax.set_title('Boxplot')
    ax.set_ylabel('Value units')


def violin_plot(data, ax):
    '''
        :param data: pandas DataFrame or Series where every row is a category (group of bars), each column is a subcategory/feature to measure about the group (individual bar, the blue one for example)
        :param ax: axis to make the plot
    '''
    # Transform data for the plot
    if type(data) is pd.DataFrame:
        columns = data.columns
        distributions = [data[col].dropna() for col in data.columns]  # Filter null values
    elif type(data) is pd.Series:
        columns = [data.name]
        distributions = [data.dropna()]
    else:
        raise TypeError('Input must be pandas DataFrame or Series')


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

    # For the plot to be centered, set x limitations
    ax.set_xlim(0.5, len(distributions) + 0.5)

    # Add title and labels
    ax.set_title('Violin Plot')
    ax.set_ylabel('Values')

def strip_chart(data, ax, jittering_strength=0.1):
    '''
        :param data: pandas DataFrame or Series where every row is a category (group of bars), each column is a subcategory/feature to measure about the group (individual bar, the blue one for example)
        :param ax: axis to make the plot
        :param jittering_strength: How much jittering can be able to move from the center
    '''
    # You can also change uniform jitter to density-based jitter. (The bigger the density the stronger the jitter)
    def density_based_jitter(distribution):
        # Obtain density
        kde = gaussian_kde(distribution)
        densities = kde(distribution)
        # Normalize densities to range between 0 and 0.3
        densities = (densities - densities.min()) / (densities.max() - densities.min())
        # Jittering inversely proportional to density (higher density -> larger jitter)
        jitter = np.random.uniform(-1, 1, size=len(distribution)) * densities * jittering_strength
        return jitter

    # Transform data for the plot
    if type(data) is pd.DataFrame:
        columns = data.columns
        distributions = [data[col].dropna() for col in data.columns]  # Filter null values
    elif type(data) is pd.Series:
        columns = [data.name]
        distributions = [data.dropna()]
    else:
        raise TypeError('Input must be pandas DataFrame or Series')

    # Plot each of the distributions
    for i, distribution in enumerate(distributions):
        jittering = np.random.uniform(-jittering_strength, jittering_strength, size=len(distribution))
        ax.scatter(np.ones(len(distribution)) * (i+1) + jittering, distribution, color='blue', alpha=0.7, s=40)

    # Set the labels for the x-axis
    ax.set_xticks([i+1 for i in range(len(distributions))], columns, rotation=45)

    # For the plot to be centered, set x limitations
    ax.set_xlim(0.5, len(distributions) + 0.5)

    # Add title and labels
    ax.set_title('Strip Chart')
    ax.set_ylabel('Values')


def sina_plot(data, ax, jittering_strength=0.1):
    '''
        :param data: pandas DataFrame or Series where every row is a category (group of bars), each column is a subcategory/feature to measure about the group (individual bar, the blue one for example)
        :param ax: axis to make the plot
    '''

    def density_based_jitter(distribution):
        # Obtain density
        kde = gaussian_kde(distribution)
        densities = kde(distribution)
        # Normalize densities to range between 0 and 0.3
        densities = (densities - densities.min()) / (densities.max() - densities.min())
        # Jittering inversely proportional to density (higher density -> larger jitter)
        jitter = np.random.uniform(-1, 1, size=len(distribution)) * densities * jittering_strength
        return jitter

    # Transform data for the plot
    if type(data) is pd.DataFrame:
        columns = data.columns
        distributions = [data[col].dropna() for col in data.columns]  # Filter null values
    elif type(data) is pd.Series:
        columns = [data.name]
        distributions = [data.dropna()]
    else:
        raise TypeError('Input must be pandas DataFrame or Series')

    # Violin plot
    ax.violinplot(distributions,
                  positions=np.arange(len(distributions)) + 1,
                  showmeans=False,
                  showextrema=True,
                  showmedians=False,
                  quantiles=None)

    # Strip chart
    # Plot each of the distributions
    for i, distribution in enumerate(distributions):
        jittering = density_based_jitter(distribution)
        ax.scatter(np.ones(len(distribution)) * (i+1) + jittering, distribution, color='blue', alpha=0.7, s=30)

    # Set the labels for the x-axis
    ax.set_xticks([i + 1 for i in range(len(distributions))], columns,
                  rotation=45)

    # For the plot to be centered, set x limitations
    ax.set_xlim(0.5, len(distributions) + 0.5)

    ax.set_title('Sina Plot with Density-Based Jitter')
    ax.set_ylabel('Values')


def overlapping_density_plot(data, ax):
    '''
        :param data: pandas DataFrame or Series where every row is a category (group of bars), each column is a subcategory/feature to measure about the group (individual bar, the blue one for example)
        :param ax: axis to make the plot
    '''
    # Transform data for the plot
    if type(data) is pd.DataFrame:
        columns = data.columns
        distributions = [data[col].dropna() for col in data.columns]  # Filter null values
    elif type(data) is pd.Series:
        distributions = [data.dropna()]
    else:
        raise TypeError('Input must be pandas DataFrame or Series')

    # Plot both distributions in ax:
    # Create density plot (With Seaborn!)
    for distribution in distributions:
        sns.kdeplot(distribution, ax=ax, fill=True, linewidth=2, label=distribution.name)

    ax.legend(fontsize='small', loc='upper left')
    ax.set_title('Overlapping density plot')
    ax.set_ylabel('Density')
    ax.set_xlabel('Values')


def desity_plot_comparison(data1, data2, ax1, ax2, flexibility=5):
    '''
    Plots the general distributions of two data sets, alongside each independent distribution part of the whole
        :param data1: pandas Series to plot on ax1
        :param data2: pandas Series to plot on ax2
        :param ax1: axis to make the data1 plot
        :param ax2: axis to make the data2 plot
        :param flexibility: Set the limit to distribution visualization. It will help avoid cutting the plots
    '''
    # Join the yaxis
    # Optionally share y-axis between more subplots
    ax1.sharey(ax2)

    # Obtain general distribution
    distribution = pd.concat([data1, data2])

    # Plot general distribution in both axes:
    # Create density plot (With Seaborn!)
    sns.kdeplot(distribution, ax=ax1, fill=True, color='lightgrey', linewidth=2, label='Full distribution')
    sns.kdeplot(distribution, ax=ax2, fill=True, color='lightgrey', linewidth=2, label='Full distribution')

    # Obtain density scaling factor for each distribution
    scale_factor1 = len(data1) / len(distribution)
    scale_factor2 = 1 - scale_factor1

    # For distribution1:
    kde_distrib1 = gaussian_kde(data1)

    # Generate the X values for the kde plot
    x_vals = np.linspace(data1.min()-1, data1.max()+1, 100)
    y_vals = kde_distrib1(x_vals)

    # Plot the distribution 1, scaled down by the factor
    ax1.fill_between(x_vals, y_vals * scale_factor1, color='blue', alpha=0.6, linewidth=3)
    ax1.set_title(f'{data1.name} Distribution')

    # For distribution2:
    kde_distrib2 = gaussian_kde(data2)

    # Generate the X values for the kde plot
    x_vals = np.linspace(data2.min() - flexibility, data2.max()+ flexibility, 100)
    y_vals = kde_distrib2(x_vals)

    # Plot the distribution 2, scaled down by the factor
    ax2.fill_between(x_vals, y_vals * scale_factor2, color='blue', alpha=0.6, linewidth=3)
    ax2.set_title(f'{data2.name} Distribution')

    # Customize the plot
    # Add a shared title for the subplots
    # fig.text(0.275, 0.48, "Distribution comparison", ha='center', va='center', fontsize=14)
    ax1.set_ylabel(f'Scaled density')
    ax1.set_xlabel('Values')
    ax2.set_ylabel('Scaled Density')
    ax2.set_xlabel('Values')



def age_pyramid_plot(data1, data2, number_of_groups, ax):
    '''
        Plots the general distributions of two data sets, alongside each independent distribution part of the whole
        :param data1: pandas Series to plot on ax1
        :param data2: pandas Series to plot on ax2
        :param ax1: axis to make the data1 plot
        :param ax2: axis to make the data2 plot
        :param flexibility: Set the limit to distribution visualization. It will help avoid cutting the plots
    '''
    def separate_groups():
        # Create the different bins
        bins = np.linspace(min(data1.min(), data2.min()), max(data1.max(), data2.max()), number_of_groups + 1)
        # Bin data1 and order by bin (index)
        separated_data1 = pd.cut(data1, bins).value_counts().sort_index()
        # Bin data2 and order by bin (index)
        separated_data2 = pd.cut(data2, bins).value_counts().sort_index()

        # Change bin index name for clarity
        index_names = [f'({int(interval.left)}, {int(interval.right)}]' for interval in separated_data1.index]
        separated_data1.index = index_names
        separated_data2.index = index_names

        return separated_data1, separated_data2

    # Separate data into bins
    separated_data1, separated_data2 = separate_groups()

    ax.barh(separated_data1.index, -1 * separated_data1, color='blue', label=data1.name)

    # Plot female data with positive values (right side)
    ax.barh(separated_data2.index, separated_data2, color='red', label=data2.name)

    # Add labels and title
    ax.set_title('Age Pyramid')
    ax.set_xlabel('Price range Count')

    # Add grid and legend
    ax.grid(True, which='major', axis='x', linestyle='--')
    ax.legend()

    # Remove x-ticks on the negative side and format labels correctly
    ax.set_xticks(ax.get_xticks(), [int(abs(tick)) for tick in ax.get_xticks()])
    ax.set_xlabel('Values')


def piechart(data, ax):
    '''
    Something like df[feature].value_counts() should be useful for a categorical variable for example
        :param data: pandas Series where every value is a slice of the pie. The index will be used to name the values
        :param ax: axis to make the plot
    '''
    # Plot the pie chart
    _, _, autotexts = ax.pie(
        data,
        labels=data.index,
        autopct=lambda pct: f'{int(pct * sum(data) / 100)}',  # Shows exact amounts
        startangle=90
    )

    # Display the quantities in white and bold font
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(9)
        autotext.set_weight('bold')

    ax.set_title('Piechart Plot')


def vertical_barplot(data, ax):
    '''
    Something like df[feature].value_counts() should be useful for a categorical variable for example
        :param data: pandas Series where every value is bar. The index will be used to name each of the bars
        :param ax: axis to make the plot
    '''

    # Make the plot
    ax.bar(data.index, data.values, orientation='vertical')

    # Customize xticks
    ax.set_xticklabels(data.index, rotation=45, ha='right')

    # Customize labels and title
    ax.set_xlabel('Categories')
    ax.set_ylabel('Value Count')
    ax.set_title('Vertical barplot')


def horizontal_barplot(data, ax):
    '''
    Something like df[feature].value_counts() should be useful for a categorical variable for example
        :param data: pandas Series where every value is bar. The index will be used to name each of the bars
        :param ax: axis to make the plot
    '''
    # Make the plot
    ax.barh(data.index, data.values, orientation='horizontal')

    # Customize labels and title
    ax.set_xlabel('Value Count')
    ax.set_ylabel('Categories')
    ax.set_title('Horizontal barplot')


def single_stackedbar_plot(data, ax):
    '''
    Something like df[feature].value_counts() should be useful for a categorical variable for example
        :param data: pandas Series where every value is bar. The index will be used to name each of the bars
        :param ax: axis to make the plot
    '''

    # Store the different categories names
    categories = data.index

    bottom = 0

    # Create the stacked bar chart
    for i, value in enumerate(data.values):
        ax.bar(1.5, value, bottom=bottom, label=categories[i], width=1.5)

        # Add a text with the specific amount
        ax.text(1.5,  bottom + value / 2, str(value), ha='center', va='center', color='white',
                fontweight='bold')

        bottom += value  # Update the bottom position for stacking

    # Remove xticks
    ax.set_xticks([])

    # Activate legend
    ax.legend(title='categories', fontsize='x-small')

    # Set axes limits
    ax.set_xlim(0, 3)
    ax.set_ylim(0, data.sum() + data.sum() * 0.3)

    # Customize axis labels and title
    ax.set_ylabel('Value counts')
    ax.set_title('Stacked barplot')

def mosaic_plot(matrix_data, ax):
    '''
        :param matrix_data: pandas dataframe or Series where every variable must be Categorical
        :param ax: axis to make the plot
    '''

    # Store all feature names in a list
    features = matrix_data.columns.tolist()

    # Obtain how many times each combination apears
    mosaic_data = matrix_data.groupby(features).size()

    # Custom labelizer: do not show anything if it has 0 values
    def dataframe_labelizer(x):
        if mosaic_data[x] == 0:
            return ''
        else:
            return ('\n ').join(x)

    def series_labelizer(x):
        if mosaic_data[x[0]] == 0:
            return ''
        else:
            return ('\n ').join(x)

    # Make the plot
    if type(mosaic_data) == pd.DataFrame:
        mosaic(mosaic_data, title='Mosaic Plot', labelizer=dataframe_labelizer, ax=ax)
    elif type(mosaic_data) == pd.Series:
        mosaic(mosaic_data, title='Mosaic Plot', labelizer=series_labelizer, ax=ax)


def tree_map_plot(data, ax):
    '''
    THIS METHOD STORES THE RESULT AS AN IMAGE. NO SINERGY WITH PYPLOT AXES
        :param matrix_data: pandas dataframe or Series where every variable must be Categorical
        :param ax: axis to make the plot
    '''
    # Store all feature names in a list
    features = data.columns.tolist()

    # Obtain how many times each combination apears
    tree_map_data = data.groupby(features).size().reset_index(name='values')

    # Create a treemap
    fig = px.treemap(
        tree_map_data,
        path=features,  # Hierarchical paths
        values='values',  # Values determine the size
        title="Treemap visualization"
    )

    fig.write_image("treemap_plot.png")


def paralel_set_plot(data):
    '''
    THIS METHOD STORES THE RESULT AS AN IMAGE. NO SINERGY WITH PYPLOT AXES
        :param matrix_data: pandas dataframe or Series where every variable must be Categorical
        :param ax: axis to make the plot
    '''

    features = data.columns.to_list()

    # OPTIONAL - You need a column indication color of the sample
    #data['color'] = df[feature].map(feature_to_color_dict)

    # Create the parallel sets plot
    fig = px.parallel_categories(
        df,
        dimensions=features,
        # color='color',  # Optional: assign a constant color for all links
        labels=features,
        title='Parallel set visualization'
    )

    # Show the plot
    fig.write_image('paralel_set_plot.png')


def density_comparison_matrix(data, plot_range=25):
    '''
    THIS METHOD STORES THE RESULT AS AN IMAGE. NO SINERGY WITH PYPLOT AXES
        :param data: dataframe input with 3 cols. two first categorical, last one continous. First column in y axis, second in x axis
        :return: density comparison matrix
    '''
    # Optionally share y-axis between more subplots
    category_values = [data[column].unique() for column in data.columns[:-1]]

    category_name = data.columns[0]
    subcategory_name = data.columns[1]
    continous_variable = data.columns[2]

    total_distribution_elements = len(data)

    # Create plot
    fig, axes = plt.subplots(len(category_values[0]), len(category_values[1]), figsize=(16,6), sharex=True, sharey=True)

    for i, category in enumerate(category_values[0]):
        for j, subcategory in enumerate(category_values[1]):
            # In every cell, plot the general distribution
            sns.kdeplot(data[continous_variable], ax=axes[i,j], fill=True, color='lightgrey', linewidth=2, label='Full distribution')
            # Obtain the subset of the data
            data_subset = data.loc[(data[category_name] == category) & (df[subcategory_name] == subcategory), continous_variable]

            axes[i,j].set_ylabel(category)
            axes[i, j].set_xlabel(subcategory)

            if len(data_subset) == 0:
                # If there is no values, dont plot anything
                continue
            elif len(data_subset) == 1:
                # If there is only one point, plot a vertical line
                axes[i,j].vlines(data_subset, ymin=0, ymax=0.01)
                continue
            # If there is more than one element, plot the distribution
            # Obtain scale factor
            scale_factor = len(data_subset)/total_distribution_elements
            # Obtain kde from the subset
            kde_subset = gaussian_kde(data_subset)
            # Obtain x and y values to plot
            x_vals = np.linspace(data_subset.min() - plot_range, data_subset.max() + plot_range, 100)
            y_vals = kde_subset(x_vals)

            # Plot the scaled subset density
            axes[i,j].fill_between(x_vals, y_vals * scale_factor, color='blue', alpha=0.6, linewidth=3)

            #ax1.set_title(f'{distrib1.name} Distribution')

    # Set a single x and y label for the entire figure
    fig.supylabel(category_name, fontsize=14)
    fig.supxlabel(subcategory_name, fontsize=14)
    fig.suptitle('Density comparation matrix')

    # Save the figure
    fig.tight_layout()
    fig.savefig('Visualizing_Density_Comparison.png')


def frequency_plot(probability, ax, grid_size = 10):
    '''
        :param prob: probability (0.1-1)
        :param ax: axis to make the frequency plot
    '''
    total_squares = grid_size ** 2
    colored_squares = int(probability * total_squares)

    # Create an array with 1s (colored) and 0s (not colored) based on probability
    squares = np.zeros(total_squares) + 0.3
    squares[:colored_squares] = 1
    np.random.shuffle(squares)  # Shuffle to distribute colored squares randomly

    # Reshape the array to a grid
    grid = squares.reshape((grid_size, grid_size))

    # Plot the grid
    sns.heatmap(grid, ax=ax, cmap=["white", "blue"], linewidths=0.5, linecolor='gray',
                cbar=False, square=True, xticklabels=False, yticklabels=False)

    # Set plot limits to ensure the entire grid is shown
    ax.set_xlim(0, grid_size+0.1)
    ax.set_ylim(-0.1, grid_size)

    # Title with probability
    ax.set_title(f"Probability: {probability * 100:.1f}%")

def error_bar_plot(data, ax):
    '''
        :param data: pandas DataFrame or Series with continous values. Each column will be a different bar
        :param ax: axis to make the error bar plot
    '''
    # Obtain mean from the data for the bar plot
    data_average = data.mean(axis=0)

    # Obtain sample standard error as a measure of error
    bar_error = data.std(axis=0)

    label = data_average.index if type(data)==pd.DataFrame else data.name

    # Make the barplot
    ax.bar(label, data_average, yerr=bar_error, capsize=5)

    # Rotate labels
    ax.set_xticklabels(label, rotation=45, ha='right')

    ax.set_title('Error bar plot')



from project.read_data import obtain_dataset

# obtain data
# Pivot data so bike types are column names, and consumption are values
#df = obtain_dataset()
#df = df.pivot(columns='bike type', values='consumption')
#df = df.iloc[:,1]

df = obtain_dataset()
df = df[['bike type', 'weight full', 'acceleration']]
#df['acceleration'] = pd.cut(df['acceleration'], bins=(0, 25, 1000), labels=['fast', 'slow']).fillna('slow') # Nulls are slow bikes
#weight_continous = df[['acceleration', 'bike type', 'weight full']].copy()
#df['weight full'] = pd.cut(df['weight full'], bins=(0,150,1000), labels=['light', 'heavy'])


# Any bike that does not reach 100 km/h (null values) is considered slow
#df.loc[df['acceleration'].isnull(), 'acceleration'] = 'slow'
df = df[['weight full']]

fig, ax = plt.subplots()
error_bar_plot(df, ax)
fig.tight_layout()
fig.show()



