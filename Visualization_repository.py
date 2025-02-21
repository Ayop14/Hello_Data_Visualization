# Import data processing libraries
import pandas as pd
import numpy as np
from itertools import chain

# Import visualization libraries
import seaborn as sns
import plotly.express as px

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter

# Import math libraries
from scipy.stats import probplot, gaussian_kde, t


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



def bubble_chart(x, y, size_variable, ax):
    '''
    x: pandas series to plot on x axis
    y: pandas series to plot on y axis
    size_variable: pandas series to plot as a continous variable using size
    ax: pyplot ax to plot the visualization in
    '''

    # Obtain maximum value for size variable
    max_size = size_variable.max()

    # Normalize and create adequate buble sizes
    sizes = size_variable / max_size * 500 # * 500 for adequate buble sizes

    # Create the plot
    ax.scatter(x, y, s=sizes, edgecolors='k', lw=0.4)

    # Add a legend for bubble sizes
    size_values = [max_size * factor for factor in [0.25, 0.5, 0.75, 1.0]]  # Example size values
    size_labels = [f"{val:.1f}" for val in size_values]
    size_markers = [val / max_size * 500 for val in size_values]  # Match normalization for legend

    for size, label in zip(size_markers, size_labels):
        ax.scatter([], [], s=size, edgecolors='k', lw=0.4, color='steelblue', label=f"Size: {label}")

    ax.legend(title=size_variable.name, title_fontsize=10, fontsize=8,  labelspacing=1.5)

    # Format the axes
    ax.set_title('Buble chart plot')
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)



def grouped_bars_plot(matrix_data, ax):
    ''' Pivot table + aggfunc=size will probe useful
    :param matrix_data: Matrix where every row is a category (group of bars), each column is a subcategory/feature to measure about the group (individual bar, the blue one for example)
        make sure rows have string index with the name of the feature they are measuring
    :param ax: axis to make the plot
    '''

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
    ''' pivot_table with aggfunc="size" is very useful for this format
    :param matrix_data: Matrix where every row is a category (group of bars), each column is a subcategory/feature to measure about the group (individual bar, the blue one for example)
        make sure rows have string index with the name of the feature they are measuring. shape ngroups, n_features
    :param ax: axis to make the plot
    '''

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
        :param matrix_data: Matrix where every row will be a row of the heatmap, and columns will be the columns of the heatmap
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
        :param data: pandas DataFrame or Series where every column is a distribution to make a box plot of
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

def density_plot_comparison(data1, data2, ax1, ax2, flexibility=5):
    '''
    Plots the general distributions of two data sets, alongside each independent distribution part of the whole
        :param data1: pandas Series to plot on ax1
        :param data2: pandas Series to plot on ax2
        :param ax1: axis to make the data1 plot
        :param ax2: axis to make the data2 plot
        :param flexibility: Set the limit to distribution visualization. It will help avoid cutting the plots
    '''
    # Clean data from nulls
    clean_data1 = data1.dropna()
    clean_data2 = data2.dropna()
    # Join the yaxis
    # Optionally share y-axis between more subplots
    ax1.sharey(ax2)

    # Obtain general distribution
    distribution = pd.concat([clean_data1, clean_data2])

    # Plot general distribution in both axes:
    # Create density plot (With Seaborn!)
    sns.kdeplot(distribution, ax=ax1, fill=True, color='lightgrey', linewidth=2, label='Full distribution')
    sns.kdeplot(distribution, ax=ax2, fill=True, color='lightgrey', linewidth=2, label='Full distribution')

    # Obtain density scaling factor for each distribution
    scale_factor1 = len(clean_data1) / len(distribution)
    scale_factor2 = 1 - scale_factor1

    # For distribution1:
    kde_distrib1 = gaussian_kde(clean_data1)

    # Generate the X values for the kde plot
    x_vals = np.linspace(clean_data1.min()-1, clean_data1.max()+1, 100)
    y_vals = kde_distrib1(x_vals)

    # Plot the distribution 1, scaled down by the factor
    ax1.fill_between(x_vals, y_vals * scale_factor1, color='blue', alpha=0.6, linewidth=3)
    ax1.set_title(f'{clean_data1.name} Distribution')

    # For distribution2:
    kde_distrib2 = gaussian_kde(clean_data2)

    # Generate the X values for the kde plot
    x_vals = np.linspace(clean_data2.min() - flexibility, clean_data2.max()+ flexibility, 100)
    y_vals = kde_distrib2(x_vals)

    # Plot the distribution 2, scaled down by the factor
    ax2.fill_between(x_vals, y_vals * scale_factor2, color='blue', alpha=0.6, linewidth=3)
    ax2.set_title(f'{clean_data2.name} Distribution')

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
    # Obtain value counts to plot
    aux = data.value_counts()

    # Plot the pie chart
    _, _, autotexts = ax.pie(
        aux,
        labels=aux.index,
        autopct=lambda pct: f'{int(pct * sum(aux) / 100)}',  # Shows exact amounts
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
    # Obtain value counts to plot
    aux = data.value_counts()

    # Make the plot
    ax.bar(aux.index, aux.values, orientation='vertical')

    # Customize xticks
    ax.set_xticklabels(aux.index, rotation=45, ha='right')

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
    # Obtain value counts to plot
    aux = data.value_counts()

    # Make the plot
    ax.barh(aux.index, aux.values, orientation='horizontal')

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

    # Create labelizer
    def labelizer(x):
        N = mosaic_data.get(x,0)
        if N == 0:
            return ''
        else:
            return ('\n ').join(x)

    # Make the plot
    mosaic(mosaic_data, title='Mosaic Plot', labelizer=labelizer, ax=ax, properties={'color': 'white'})



def tree_map_plot(data, file_path):
    '''
    THIS METHOD STORES THE RESULT AS AN IMAGE. NO SINERGY WITH PYPLOT AXES
        :param matrix_data: pandas dataframe or Series where every variable must be Categorical
        :param file_path: path to store the resulting image. Must include the file name and file type (png)
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

    fig.write_image(file_path)


def paralel_set_plot(data, file_path):
    '''
    THIS METHOD STORES THE RESULT AS AN IMAGE. NO SINERGY WITH PYPLOT AXES
        :param matrix_data: pandas dataframe or Series where every variable must be Categorical
        :param file_path: path to store the resulting image. Must include the file name and file type (png)
    '''

    features = data.columns.to_list()

    # OPTIONAL - You need a column indication color of the sample
    #data['color'] = df[feature].map(feature_to_color_dict)

    # Create the parallel sets plot
    fig = px.parallel_categories(
        data,
        dimensions=features,
        # color='color',  # Optional: assign a constant color for all links
        labels=features,
        title='Parallel set visualization'
    )

    # Show the plot
    fig.write_image(file_path)


def density_comparison_matrix(data, file_path, plot_range=25):
    '''
    THIS METHOD STORES THE RESULT AS AN IMAGE. NO SINERGY WITH PYPLOT AXES
        :param data: dataframe input with 3 cols. two first categorical, last one continous. First column in y axis, second in x axis
        :param file_path: path to store the resulting image. Must include the file name and file type (png)
        :param plot_range:
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
            data_subset = data.loc[(data[category_name] == category) & (data[subcategory_name] == subcategory), continous_variable]

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
    fig.savefig(file_path)


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

def graded_error_bar_plot(data, ax, ratio):
    '''
        :param data: pandas DataFrame or Series with continous values. Each column will be a different graded bar
        :param ax: axis to make the graded error bar plot
        :param ratio: Measures the height between bars. Different value ranges will need different ratios
    '''
    def calculate_confidence_interval(data, confidence=0.95):
        mean_std = data.sem()
        critical_value = t.ppf(1 - (1 - confidence) / 2, len(data) - 1)
        error_margin = critical_value * mean_std
        return error_margin

    # Plot each graded confidence interval as a layered bar
    for i, column in enumerate(data.columns):
        # Obtain the variable from the dataframe
        df = data[column]

        # Obtain statistical information
        mean = df.mean()
        ci80 = calculate_confidence_interval(df, confidence=0.80)
        ci95 = calculate_confidence_interval(df, confidence=0.95)
        ci99 = calculate_confidence_interval(df, confidence=0.99)
        # Plot the 99% confidence interval and plot limit lines for every bar
        ax.barh(i*ratio, 2 * ci99, left=mean - ci99, color='skyblue', edgecolor='black', height=0.05,
                label='99% CI' if i == 0 else "")
        ax.plot([mean - ci99, mean - ci99], [i*ratio - 0.05, i*ratio + 0.05], color='black', linewidth=1.5)
        ax.plot([mean + ci99, mean + ci99], [i*ratio - 0.05, i*ratio + 0.05], color='black', linewidth=1.5)
        # Plot the 95% confidence interval and plot limit lines for every bar
        ax.barh(i*ratio, 2 * ci95, left=mean - ci95, color='cornflowerblue', edgecolor='black', height=0.1,
                label='95% CI' if i == 0 else "")
        ax.plot([mean - ci95, mean - ci95], [i*ratio - 0.075, i*ratio + 0.075], color='black', linewidth=1.5)
        ax.plot([mean + ci95, mean + ci95], [i*ratio - 0.075, i*ratio + 0.075], color='black', linewidth=1.5)
        # Plot the 80% confidence interval and plot limit lines for every bar
        ax.barh(i*ratio, 2 * ci80, left=mean - ci80, color='royalblue', edgecolor='black', height=0.15,
                label='80% CI' if i == 0 else "")
        ax.plot([mean - ci80, mean - ci80], [i*ratio - 0.1, i*ratio + 0.1], color='black', linewidth=1.5)
        ax.plot([mean + ci80, mean + ci80], [i*ratio - 0.1, i*ratio + 0.1], color='black', linewidth=1.5)

        # Plot the mean value as a vertical line
        ax.plot([mean, mean], [i*ratio - 0.15, i*ratio + 0.15], color='black', linewidth=1.5)

    # Plot mean values using a scatterplot
    mean_values = data.mean(axis=0)
    ax.scatter(mean_values, np.arange(len(mean_values)) *ratio, color='orange', marker='o', s=150)

    # Label and ticks
    ax.set_yticks(np.arange(len(data.columns))*ratio)
    ax.set_yticklabels(data.columns)
    ax.set_xlabel('Value')
    ax.set_title('Graded Error Bars with Confidence Intervals')

    # Add a legend
    ax.legend(loc='upper right')

def quantile_dot_plot(data, ax, total_dots, cols):
    '''
        :param data: pandas Series with continous values to make the dot plot
        :param ax: axis to make the graded error bar plot
        :param Total_dots: Total dots to represent the entire distribution
        :param cols: Number of dot columns
    '''
    # Define quantiles (25% and 75%)
    quantile_25 = data.quantile(0.25)
    quantile_75 = data.quantile(0.75)

    # Plot density curve
    sns.kdeplot(data, ax=ax, fill=True, color="lightgray", alpha=0.5, linewidth=2)

    # Generate bin edges and midpoints
    x_min, x_max = data.min(), data.max()
    bin_edges = np.linspace(x_min, x_max, cols + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate the number of dots per bin based on data distribution
    bin_counts = pd.cut(data, bins=bin_edges).value_counts().sort_index()

    # Radius in x-units for the circles
    radius = (bin_edges[1] - bin_edges[0]) / 2  # Adjusted to fit within each bin
    diameter = radius * 2

    # Recompute aspect ratio after fixing limits
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    aspect_ratio = (y_range * 1.05) / x_range

    # Fix the y-axis range to prevent dynamic resizing and calculate aspect_ratio
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_range * 1.05)

    # Helper function to determine the color of a dot based on its x position
    def get_color(x):
        if x <= quantile_25:
            return "yellow"
        elif x <= quantile_75:
            return "lightblue"
        else:
            return "blue"

    # Plot the grid of dots
    for i, center in enumerate(bin_centers):
        color = get_color(center)

        # Proportion of dots in this bin based on data distribution
        proportion = bin_counts.iloc[i] / len(data)  # Fraction of data in this bin
        num_dots = int(proportion * total_dots)  # Number of dots for this bin

        for j in range(num_dots):
            # Calculate y positions with fixed scaling
            y_position = diameter * aspect_ratio * (2 * j + 1)

            # Plot the dot as a perfect circle with height adjusted by aspect_ratio
            circle = Ellipse(
                (center, y_position),
                width=diameter,
                height=diameter * aspect_ratio * 2,
                color=color,
                edgecolor='black',
                lw=0.5
            )
            ax.add_patch(circle)

    # Adjust plot aesthetics
    ax.set_title("Quantile Dot Plot")
    ax.set_xlabel("Value")
    ax.axvline(quantile_25, color='gray', linestyle='--', linewidth=1)
    ax.axvline(quantile_75, color='gray', linestyle='--', linewidth=1)



def hyphothetical_outcome_plot(data1, data2, sample_size, file_path):
    '''
        :param data1: pandas Series with continous values for upper bar values
        :param data1: pandas Series with continous values for lower bar values
        :param sample_size: number of pairs to make the plot with
    '''

    # Create plot
    fig, ax = plt.subplots()

    # Set vertical line positions
    y1 = 1
    y2 = 2

    ax.set_ylim(0, 3)

    # Set vertical bar size
    ybar_size = 0.25

    # Set ax ticks
    ax.set_yticks([y1, y2], [data1.name, data2.name])

    # Set x limits to max and min from the data (Matplotlib doesn't update it automatically)
    xmax = max(chain(data1, data2))
    xmin = min(chain(data1, data2))

    range = (xmax - xmin) * 0.15
    ax.set_xlim(xmin - range, xmax + range)

    # Draw horizontal bars
    ax.grid(which='major', axis='y', lw=3)

    # Initialize vertical lines
    line1, = ax.plot([], [], 'r-', lw=5)  # Vertical line for bar 1
    line2, = ax.plot([], [], 'r-', lw=5)  # Vertical line for bar 2

    # Title
    ax.set_title("Hypothetical outcome plot")
    ax.set_xlabel('Feature values')

    sample1 = np.random.choice(data1, sample_size)
    sample2 = np.random.choice(data2, sample_size)

    # Update function for animation
    def update(frame):
        # Update vertical lines
        line1.set_data([sample1[frame], sample1[frame]], [y1 - ybar_size, y1 + ybar_size])  # Sample data 1 bar
        line2.set_data([sample2[frame], sample2[frame]], [y2 - ybar_size, y2 + ybar_size])  # Sample data 2 bar
        return line1, line2

    # Create animation
    anim = FuncAnimation(fig, update, frames=sample_size, interval=500, blit=False)

    # Save as GIF
    anim.save(file_path, writer=PillowWriter(fps=2))
    plt.close()




def ordered_scatter_plot(data, ax):
    """
    Makes a scatterplot where every value is in its own independent line, ordered incrementally
    :param data: Pandas Series or Dataframe with 2 columns (First continous, second catgorical) to plot
    :param ax: axes to make the plot
    """

    # Obtain y values for every point and set max y limit
    y_values = np.arange(len(data))
    ax.set_ylim(-1, len(data))

    # Set grid lines and delete ylabels for everypoint
    ax.set_yticks(y_values, [])
    ax.grid(which='major', axis='y', zorder=0)

    # make the scatterplot
    if isinstance(data, pd.Series):
        # Order the data
        ordered_data = data.sort_values()
        # Plot the values
        ax.scatter(ordered_data, y_values, s=50, color=None, zorder=2)

    elif isinstance(data, pd.DataFrame):
        # Order the data
        ordered_data = data.sort_values(by=data.columns[0])

        # Map every category (second column) to a color automatically
        categories = data.iloc[:,1].unique()
        palette = sns.color_palette("Set1", len(categories))  # Use a Set1 palette for distinct colors
        color_map = {category: palette[i] for i, category in enumerate(categories)}
        colors = data.iloc[:,1].map(color_map)

        # Plot the values
        ax.scatter(ordered_data.iloc[:,0], y_values, s=50, color=colors, zorder=2)

        # Make a customized legend
        handles = [
            plt.Line2D([0], [0], marker="o", color=color, label=category, markersize=10, linestyle="None")
            for category, color in color_map.items()
        ]
        plt.legend(handles=handles, title="Category", fontsize=12, title_fontsize=14)
    else:
        raise ValueError("Incorrect data type passed to ordered scatterplot")

    # Add the legend with color information
    #ax.legend(handles=legend, title="Legend")

    # Set plot aesthetics
    ax.set_xlabel('Feature values')
    ax.set_title('Ordered scatter plot')

def ordered_heatmap(df, ax, columns):
    """
    Creates a mean ordered heatmap
    :param data:Pandas dataframe with two columns: First continous and second categorical. there will be as much rows as different category values
    :param ax:Axis on which to plot make the plot
    :param columns: Number of columns/splits along the x axis
    """
    # Transform the data for visualizations
    data = [pd.Series(df[df.iloc[:,1]==category][df.columns[0]].dropna(), name=category) for category in df.iloc[:,1].unique()]

    # Obtain data range
    xmin = min(chain(*data))
    xmax = max(chain(*data))

    # Step 1: Define intervals and bin data
    bins = np.linspace(xmin, xmax, columns)
    binned_data = {
        series.name: np.histogram(series, bins=bins)[0]
        for series in data
    }

    # Step 2: Create a DataFrame from binned data
    heatmap_data = pd.DataFrame(binned_data).T  # Transpose to get series as rows
    heatmap_data.columns = [f'>{int(range)}' for range in bins[:-1]]

    # Step 3: Sort rows based on series means
    heatmap_data['Series_means'] = [series.mean() for series in data]
    heatmap_data = heatmap_data.sort_values('Series_means', ascending=False).drop(columns='Series_means')

    # Step 4: Plot the heatmap
    sns.heatmap(
        heatmap_data,
        cmap="YlGnBu",
        ax=ax,
        cbar_kws={'label': 'Count'},
        linewidths=0.5
    )
    ax.set_title("Heatmap of Series by Interval Counts")
    ax.set_xlabel("Intervals")
    ax.set_ylabel("Series")



def correlogram(df, ax):
    """
    Makes a correlation plot based on the variables from the dataframe. Ensure they all are continous
    :param df: dataframe with all variables wanted to make a correlogram plot
    :param ax: axis to make the plot
    """

    # Obtain variables correlation
    correlation = df.corr()

    # Obtain the size and labels of the matrix
    n_variables, _ = correlation.shape
    labels = correlation.columns

    # Flatten the correlation matrix to a 1D array
    correlation = correlation.to_numpy().flatten()

    # Obtain x and y coordinate values
    # Step 1: Obtain the x and y coordinates of every point in the scatter. Here, first param are the row values and second the column values
    x_values, y_values = np.meshgrid(np.arange(n_variables), np.arange(n_variables)[::-1])
    # Step 2: Flatten the resulting coordinates
    x_values, y_values = x_values.ravel(), y_values.ravel()

    # Obtain sizes for the scatterplot
    sizes = np.abs(correlation) * 1000

    # Obtain colors for the scatterplot
    colors = plt.cm.coolwarm((correlation + 1)/2) # normalize from 0 to 1 values

    # Create the correlogram using scatter
    ax.scatter(x_values, y_values, s=sizes, color=colors, edgecolors="k", lw=0.5 )

    # Set limits for the plot to be correctly seen
    ax.set_xlim(-0.5, n_variables - 0.5)
    ax.set_ylim(-0.5, n_variables -0.5)

    # Format axes
    ax.set_xticks(range(n_variables))
    ax.set_yticks(range(n_variables))
    ax.set_xticklabels(labels, rotation=45, ha='left')
    ax.set_yticklabels(labels[::-1])
    ax.xaxis.tick_top()  # Move x-axis labels to the top

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Correlation Coefficient')

    ax.set_title('Correlogram plot', pad=20)



def scatterplot_matrix(data, file_path):
    """ NO AX- DEPENDS ON THE SHAPE OF THE DATA
    Makes a correlation scatterplot matrix: one scatterpplot for every pair f variables in the dataframe
        :param df: Pandas dataframe with continous values
        :param file_path: Path to store the figure result (Including folder, file name and file type (.png))
    """
    # Obtain labels and number of variables
    labels = data.columns
    n_variables = len(labels)

    # Create the plot
    fig, axes = plt.subplots(n_variables, n_variables, figsize=(n_variables*2, n_variables*2))

    for i, x_label in enumerate(labels):
        for j, y_label in enumerate(labels):
            # Obtain the value of each feature
            x_data = data[x_label]
            y_data = data[y_label]

            # Make a scatterplot on the corresponding axes
            axes[j, i].scatter(x_data, y_data, edgecolors='k', lw=0.2)

            # Set axis labels if nbecessary
            if i == 0:
                axes[j, i].set_ylabel(y_label)

            if j == n_variables-1:
                axes[j, i].set_xlabel(x_label)

    # Set suptitle
    fig.suptitle('Scatterplot matrix')

    # Save the image
    fig.tight_layout()
    fig.savefig(file_path)


def slopegraph(data, ax):
    """
    :param data: Pandas dataframe with continous values. As much columns as stages in the slopegraph
    :param ax: axis to make the plot
    """
    # Obtain shape
    n_rows, n_columns = data.shape

    # Plot each row as a time series
    for i in range(n_rows):
        # Plot the sample as a time series
        ax.plot(data.iloc[i], marker='o')

        # Set the name of each plot right in last column
        ax.text(n_columns-1 + 0.1, data.iloc[i, -1], f'Sample {i}', fontsize=10, ha='left', va='center')


    ax.set_xticks(np.arange(n_columns), data.columns) # Format ticks
    ax.set_title('Slopegraph plot')
    ax.set_ylabel('Slopegraph Units')

    # Deactivate black box lines (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)



def connected_scatterplot(data, ax, label_offset=(0.5,0.5)):
    """
    :param data: Pandas dataframe with 2 columns. First will be x and second will be y values
    :param ax: axis to make the plot
    :param label_offset: tuple (x,y) offset to plot the label respective to every point
    """
    # Obtain data from dataframe
    labels = [f'Step {i + 1}' for i in range(len(data))] # Example labels
    x = data.iloc[:,0]
    y = data.iloc[:,1]

    # Plot the connected scatterplot
    ax.plot(x, y, marker='o', linestyle='-', color='blue', label='Path')

    # Plot the labels asociated with each point
    for i, label in enumerate(labels):
        ax.text(x.iloc[i] + label_offset[0], y.iloc[i] + label_offset[1], label, fontsize=8, ha='center')

    # Format the axes
    ax.set_title("Connected Scatterplot")
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    ax.grid(alpha=0.3)
    # ax.legend() Dont add legend with just one


def divide_time_series_into_components(data, file_path):
    """
    NO AX- IT REQUIRES 4 AXES SPECIFICALLY THE DATA IN
    :param data: Pandas Series to decompose into components trend, seasonality and noise
    """

    # Clean nan values
    clean_data = data.dropna()

    # Perform seasonal decomposition
    result = seasonal_decompose(clean_data, model='additive', period=10)  # Assuming monthly seasonality (30 days)

    # Extract trend, seasonal fluctuations and noise
    trend = result.trend
    seasonal_fluctuations = result.seasonal
    noise = result.resid

    # Create the plot
    fig, axes = plt.subplots(4,1, figsize=(6, 8))

    # Plot original Data
    axes[0].plot(clean_data)
    axes[0].set_title('Original data')

    # Plot trend
    axes[1].plot(trend)
    axes[1].set_title('Trend')

    # Plot seasonal fluctuations
    axes[2].plot(seasonal_fluctuations)
    axes[2].set_title('Seasonal fluctuations')

    # Plot noise
    axes[3].plot(noise)
    axes[3].set_title('Noise')

    fig.suptitle(f'{data.name} decomposition')
    fig.tight_layout()
    fig.savefig(file_path)


