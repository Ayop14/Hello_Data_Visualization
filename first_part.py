
import pandas as pd
from read_data import obtain_dataset
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
from scipy.stats import probplot, gaussian_kde
from joypy import joyplot
from statsmodels.graphics.mosaicplot import mosaic
import plotly.express as px
from matplotlib.patches import Ellipse
from scipy import stats
from matplotlib.animation import FuncAnimation, PillowWriter
from itertools import chain


type_to_shape_dict = {
        'Naked':'^',
        'Sport':'o',
        'Custom':'s',
        'Scrambler':'p',
        'Scooter':'*',
        'Trail': 'P'
    }

type_to_color = {
        'Naked':'steelblue',
        'Sport':'darkorange',
        'Custom': 'goldenrod',
        'Scrambler':'crimson',
        'Scooter':'ForestGreen',
        'Trail': 'darkviolet'
    }

# Turn weight discrete. What are the different weight values? Lets see with a simple histogram
def weight_visualization():
    # Obtain data
    data = obtain_dataset()
    weight = data['acceleration']
    # Visualize weights using a histogram
    weight.plot(kind='hist', bins = 25)
    # Plot 0.25, 0.75 quartiles as vertical bars
    plt.axvline(weight.quantile(0.75), color='red')
    plt.axvline(weight.quantile(0.25), color='red')
    # Save the figure
    plt.savefig('Images/weight_histogram.png')

# Plot the relation between consumption, acceleration, max speed, bike tipe and weight
def relation_5_features():
    '''
        Define type of features for the plot:
                consumption: continous - X axis
                acceleration: continous - Y axis
                max speed: continous - Size
                bike type: discrete - Shape
                weight : continous - turn discrete - Color
    :return:
    '''
    def scatter_5_f():
        pass
        # Very specific. Depends on continous or discrete variables as well

    # import data
    df = obtain_dataset()
    # Turn discrete the weight
    df['weight full'] = pd.cut(df['weight full'], bins=(0,140,160,1000), labels=['light', 'medium', 'heavy'])

    # Turn max speed to speed ranges. Nulls set to 0
    df.loc[df['max speed'].isna(), 'max speed'] = 1
    df['max speed'] = pd.cut(df['max speed'], bins=(0, 100, 105, 110, 115,130), labels=[20,40,60,80,100]).astype(int)


    # 5 feature plot
    weight_to_color_dict = {
        'light': 'red',
        'medium': 'green',
        'heavy': 'blue'
    }

    weight_to_color_func = lambda x: weight_to_color_dict[x]

    fig, ax = plt.subplots()

    for key in type_to_shape_dict.keys():
        aux = df[df['bike type'] == key]
        ax.scatter(
            aux['consumption'], # x
            aux['acceleration'], # Y
            s=aux['max speed'], # Size
            marker= type_to_shape_dict[key], # Shape
            c=aux['weight full'].apply(weight_to_color_func) # Color
        )


    ax.set_title('5 feature visualization')
    ax.set_xlabel('Consumption  (l/100km)')
    ax.set_ylabel('Acceleration  0-100km (s)')

    legend_elements = [Line2D([], [], color='red', label='Light', markersize=5, lw=8),
                       Line2D([], [], color='green', label='Medium', markersize=5, lw=8),
                       Line2D([], [], color='blue', label='Heavy', markersize=5, lw=8)]

    legend_elements2 = [Line2D([], [], color='w', marker='o',  label='Sport', markerfacecolor='grey', markersize=10),
                       Line2D([], [], color='w', marker='s', label='Custom', markerfacecolor='grey', markersize=10),
                       Line2D([], [], color='w', marker='p', label='Scrambler', markerfacecolor='grey', markersize=10),
                        Line2D([], [], color='w', marker='*', label='Scooter', markerfacecolor='grey', markersize=15),
                        Line2D([], [], color='w', marker='P', label='Trail', markerfacecolor='grey', markersize=10),
                        Line2D([], [], color='w',marker='^', label='Naked', markerfacecolor='grey', markersize=10)]

    legend_elements3 = [Line2D([], [], color='w', marker='o', label='x <= 100', markerfacecolor='grey', markersize=5),
                        Line2D([], [], color='w', marker='o', label='100 <= x < 105', markerfacecolor='grey', markersize=6),
                        Line2D([], [], color='w', marker='o', label='105 <= x < 110', markerfacecolor='grey', markersize=7),
                        Line2D([], [], color='w', marker='o', label='110 <= x < 115', markerfacecolor='grey', markersize=8),
                        Line2D([], [], color='w', marker='o', label='115 <= x', markerfacecolor='grey', markersize=9)]


    leg1 = fig.legend(handles=legend_elements, title='Weight', fontsize='small', bbox_to_anchor=(0.9,0.875))
    leg2 = fig.legend(handles=legend_elements2, title='Bike type', loc='upper right', fontsize='small', bbox_to_anchor=(0.9,0.7))
    leg3 = fig.legend(handles=legend_elements3, title='Max speed (Km/h)', loc='upper left', fontsize='small', bbox_to_anchor=(0.125,0.875))

    ax.add_artist(leg1)
    ax.add_artist(leg2)

    fig.savefig('Images/5 feature visualization initial')

def log_info():
    def log_visualization(data, store):
        # Obtain log transformed data
        log_data = np.log10(data)

        # Define figure
        fig, axes = plt.subplots(3, 1, )

        # deactivate y values
        plt.tick_params(left=False)

        # Draw the initial base for the plot
        for ax in axes:
            # Horizontal line
            ax.axhline(0, color='black', lw=1)
            # Turn off surrounding box
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            # Move bottom axis above
            ax.spines['bottom'].set_position(('data', 0))
            # Set padding between ticks and labels
            ax.tick_params(axis='x', pad=10)
            # Disable yaxis labels (Not required)
            ax.get_yaxis().set_visible(False)

        # Plot normal values
        axes[0].set_title('Original data, linear scale')
        axes[0].scatter(data, [0] * len(data))

        # Plot Log transformed values
        axes[1].set_title('Log transformed data, linear scale')
        axes[1].scatter(log_data, [0] * len(data))

        # Plot Log scale values
        axes[2].set_xscale('log')
        axes[2].set_title('Original data, logarithmic scale')
        axes[2].scatter(data, [0] * len(data))

        # Adjust vertical padding
        fig.subplots_adjust(hspace=1)  # adjust padding
        # Save fig
        fig.savefig(store)

    df = obtain_dataset()
    acceleration = df['acceleration']
    log_visualization(acceleration, 'Images/Visualizing log acceleration initial')

def polar_coordinate_visualization():
    def polar(data, categories, type_to_shape, store):
        '''

        :param data: Dataframe/Series with all plotting data
        :param categories: Type of every point in the data parameter (Same length)
        :param type_to_shape:Dict{'category name':matplotlib shape string}
        :param store:Where to store the resulting image
        '''

        # Create a figure and a polar subplot
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), subplot_kw={'projection': 'polar'})

        for x in range(2):
            for y in range(2):
                # Set label position
                axes[x, y].set_rlabel_position(45)
                # Turn on xticks, but without labels
                axes[x, y].set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2], [])

                # Set height values
                axes[x, y].tick_params(axis='y', pad=-1)
                # Customize label ticks
                #axes[x, y].set_yticks([5, 10, 15, 20])

                # Optionally add gridlines and a title
                axes[x, y].grid(True)

        # First plot: One lap around the polar coordinate system
        # Create angles from 0 to 4π (two times around)
        angles = np.linspace(0, 2 * np.pi, len(data))
        axes[0, 0].plot(angles, data, marker='o', markersize=4)
        axes[0, 0].set_title("One lap polar visualization")

        # Second Plot: Two laps around the polar coordinate system
        angles = np.linspace(0, 4 * np.pi, len(data))
        axes[0, 1].plot(angles, data, marker='o', markersize=4)
        axes[0, 1].set_title("Two lap polar visualization")

        # Third plot: Comparation against ech other bike type using line plot
        for category in categories.unique():
            filtered_data = data.loc[categories == category]
            angles = np.linspace(0, 2 * np.pi, len(filtered_data))
            axes[1, 0].plot(angles, filtered_data, marker=type_to_shape[category], markersize=4,
                            label=category)

        axes[1, 0].set_title("Polar plot divided by category")
        axes[1, 0].legend(loc='lower right', bbox_to_anchor=(1.25, -0.2))

        # Forth plot: Comparation against ech other bike type using line plot
        for category in categories.unique():
            filtered_data = data.loc[categories == category]
            angles = np.linspace(0, 2 * np.pi, len(filtered_data))
            axes[1, 1].scatter(angles, filtered_data, marker=type_to_shape[category], s=40,
                               label=category)

        axes[1, 1].set_title("Polar Scatterplot divided by category")
        axes[1, 1].legend(loc='lower right', bbox_to_anchor=(1.25, -0.2))

        fig.savefig(store)

    # obtain data
    df = obtain_dataset()
    polar(df['gasoline capacity'], df['bike type'], type_to_shape_dict, 'Images/Visualizing polar coordinates')

def color_powerpoint():
    def color_scatterplot(df_1, df_2, df_3, store):
        '''

        :param df_1: First dimension data to plot (X axis)
        :param df_2: Second dimension data to plot (Y axis)
        :param df_3: Third dimension data to plot (Color)
        :param store: Path to store the resulting image
        :return:
        '''
        # Create Scatterplot
        fig, ax = plt.subplots()

        # Scatterplot with torque, weight and seat height
        scatter = ax.scatter(df_1, df_2, c=df_3, s=50, edgecolor='k')

        # Add a colorbar to map colors to values
        colorbar = plt.colorbar(scatter, ax=ax)
        colorbar.set_label(df_3.name)

        # Add axis labels and title
        ax.set_xlabel(df_1.name)
        ax.set_ylabel(df_2.name)
        ax.set_title('Scatterplot with color-encoded continous variable')

        # Save the figure
        fig.savefig(store)


    def histogram_color_comparison(data, labels, interest_pos,  storage):
        '''

        :param data: Data to show in the histogram
        :param labels: Labels of each bar
        :param interest_pos: Position highlighted in left plot
        :param storage: Path to store the resulting visualization
        :return:
        '''
        # Create a histogram comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Plot regular barplot
        x_values = np.arange(len(data))
        axes[0].bar(x_values, data, color='blue', edgecolor='k')

        # Set Xtick label values
        axes[0].set_xticks(x_values, labels, rotation=20, fontsize=10)
        # Set title
        axes[0].set_title('Barplot comparing all Trail bikes prices')
        axes[0].set_ylabel('Price (Euros)')

        # Plot color variation barplot
        colors = ['lightsteelblue'] * len(data)
        colors[interest_pos] = 'blue'
        axes[1].bar(x_values, data, color=colors, edgecolor='k')

        # Set Xtick label values
        axes[1].set_xticks(x_values, labels, rotation=20, fontsize=10)
        # Set title
        axes[1].set_title('Barplot color powerpoint')

        plt.tight_layout()
        # plt.subplots_adjust(bottom=0.2)

        # Save figure
        fig.savefig(storage)


    # Obtain data
    df = obtain_dataset()
    # Scatterplot visualization
    color_scatterplot(df['torque'], df['weight full'], df['seat height'], 'Images/Visualizing color importance - scatterplot')

    # Histogram comparison
    aux = df.loc[df['bike type'] == 'Trail', ['name', 'price']]
    histogram_color_comparison(aux['price'], aux['name'],1, 'Images/Visualizing histogram color effect')

def different_amounts_visualization():
    def grouped_bars_plot(matrix_data, ax):
        '''

        :param matrix_data: Matrix where every row is a category (group of bars), each column is a subcategory (individual bar)
        :param ax: axis to make the plot
        :return: None
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

        # Set the x-axis labels and ticks
        ax.set_xticks(x + bar_width * (n_subcategories - 1) / 2, matrix_data.index)  # Center the ticks

        # Add labels and title
        ax.set_xlabel('Bike Brand')
        ax.set_ylabel('Quantity')
        ax.set_title('Bike types divided by brands')

    def stacked_bars_plot(matrix_data, ax):
        '''

        :param matrix_data: Matrix where every row is a category (group of bars), each column is a subcategory (individual bar)
        :param ax: axis to make the plot
        :return: None
        '''
        # Create stacked bars plot
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

        # Set the x-axis ticks and labels
        ax.set_xticks(x, matrix_data.index)

        # Add labels and title
        ax.set_xlabel('Bike Brand')
        ax.set_ylabel('Quantity')
        ax.set_title('Bike types divided by brand')


    def heatmap_plot(matrix_data, ax):
        '''

        :param matrix_data: Matrix that will be converted to a heatmap
        :param ax: axis to make the plot
        :return: None
        '''
        # Create Heatmap for last visualization
        # Create the heatmap using imshow
        cax = ax.imshow(matrix_data, cmap='viridis', aspect='auto')

        # Set row and column labels
        ax.set_xticks(np.arange(len(matrix_data.columns)), matrix_data.columns)
        ax.set_yticks(np.arange(len(matrix_data.index)), matrix_data.index)

        # Rotate the x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add a color bar to show the intensity scale
        fig.colorbar(cax, ticks=np.arange(matrix_data.max(axis=None) + 1))

        # Set labels and title
        ax.set_xlabel('Bike Brand')
        ax.set_ylabel('Quantity')
        ax.set_title('Bike types divided by brands')

    # obtain data
    df = obtain_dataset()
    # Transform data for visualization
    df = df.pivot_table(index='brand', columns='bike type', aggfunc='size', fill_value=0)
    df = df.loc[['Honda', 'Kawasaki', 'Suzuki', 'Yamaha']]

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Grouped bars plot
    grouped_bars_plot(df, axes[0])

    # Stacked bar plots
    stacked_bars_plot(df, axes[1])

    # Heatmap plot
    heatmap_plot(df, axes[2])

    # Show the plot
    plt.tight_layout()

    plt.savefig('Images/Visualizing_different_amounts.png')

def visualizing_single_distribution(feature):
    def histogram(data, ax):
        # Create a histogram
        ax.hist(data, bins=15)

        ax.set_title(f'Histogram of {feature}')
        ax.set_xlabel(f'{feature} values')
        ax.set_ylabel('Value counts')

    def density_plot(data, ax):
        # Create density plot (With Seaborn!)
        sns.kdeplot(data, ax=ax, fill=True, color='steelblue', linewidth=2)

        # Customize the plot
        ax.set_title(f'Density Plot of {feature}', fontsize=14)
        ax.set_xlabel(f'{feature} values')
        ax.set_ylabel('Density')

    def cumulative_plot(data, ax):
        # Make a cumulative density plot
        sorted_feature = np.sort(data)

        # Compute the cumulative values (normalized to 1)
        cumulative_values = np.arange(1, len(sorted_feature) + 1) / len(sorted_feature)

        # Plot the cumulative density function (CDF)
        ax.step(sorted_feature, cumulative_values, color='steelblue')

        # Set yticks manually
        ax.set_yticks(np.arange(0.2, 1.2, 0.2))

        # Customize the plot with title and labels
        ax.set_title(f'Cumulative Density Plot of {feature}')
        ax.set_xlabel(f'{feature} values')
        ax.set_ylabel('Cumulative Density')


    def quantile_quantile_plot(data, ax):
        # Make a quantile quantile plot
        # Remove null values
        clean_feature = data.loc[data.notnull()]
        # Generate Q-Q plot data using scipy's probplot function
        (theoretical_quantiles, sample_quantiles), (slope, intercept, _) = probplot(clean_feature, dist="norm")

        # Plot the Q-Q plot (theoretical quantiles vs. sample quantiles)
        ax.scatter(theoretical_quantiles, sample_quantiles, color='blue', label='Sample Quantiles')

        # Plot the 45-degree line using the slope and intercept from probplot
        ax.plot(theoretical_quantiles, slope * theoretical_quantiles + intercept, color='red', linestyle='--',
                        label='y = x (reference)')

        # Set a grid for clearer visualization
        ax.grid(True)

        # Customize the plot with title and labels
        ax.set_title(f'Q-Q Plot of {feature}')
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')


    def histogram_parameter_comparison(data, storage):
        fig, axes = plt.subplots(4, 4, figsize=(10, 10), sharey=True)

        for i, ax in enumerate(axes.flatten()):
            # Create a histogram
            ax.hist(data, bins=2 * i + 1)

            # Set title
            ax.set_title(f'Bins: {2 * i + 1}')

            # Turn off axis labels to declutter the grid
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f'Histogram bin size comparison for {feature}', fontsize=14)

        fig.tight_layout()
        fig.savefig(storage)

    # Obtain dataset
    df = obtain_dataset()

    # Figure 1: Single distribution visualization
    # Create a 2,2 subplot
    fig, axes = plt.subplots(2,2, figsize=(8,8))

    histogram(df[feature], axes[0,0])

    density_plot(df[feature], axes[0,1])

    cumulative_plot(df[feature], axes[1,0])

    quantile_quantile_plot(df[feature], axes[1,1])

    plt.tight_layout()
    fig.savefig('Images/Visualizing_single_distribution.png')


    # 2 Figure: Comparing Histogram parameters
    histogram_parameter_comparison(df[feature], 'Images/Visualizing_single_distribution_hist_params.png')


def visualizing_multiple_distributions():
    def box_plot(distributions, ax):
        # Plot the boxplot with the data
        ax.boxplot(distributions, tick_labels=[distribution.name for distribution in distributions],
                   meanline=None, # If showmeans=True, show mean as a line spanning the full width of the box
                   showmeans=None, # Show the arithmetic means
                   showcaps=None, # Caps at the end of the whiskers
                   showbox=None, # Show the central box
                   showfliers=None) # Show the outliers beyond the caps
        # Rotate labels
        ax.set_xticks([i + 1 for i in range(len(distributions))], [distribution.name for distribution in distributions], rotation=45)

        # Add title and labels
        ax.set_title('Boxplot of Multiple Distributions')
        ax.set_ylabel('Values')

    def violin_plot(distributions, ax):
        # Plot the violin plot
        ax.violinplot(distributions,
                      showmeans=False,
                      showextrema=True,
                      showmedians=False,
                      quantiles=None)

        # Set the labels for the x-axis
        ax.set_xticks([i + 1 for i in range(len(distributions))], [distribution.name for distribution in distributions], rotation=45)

        # Add title and labels
        ax.set_title('Violin Plot of Multiple Distribution')
        ax.set_ylabel('Values')

    def strip_chart(distributions, ax, jittering_strength=0.1):

        # Plot each of the distributions
        for i, distribution in enumerate(distributions):
            jittering = np.random.uniform(-jittering_strength, jittering_strength, size=len(distribution))
            ax.scatter(np.ones(len(distribution)) * (i+1) + jittering, distribution, color='blue', alpha=0.7, s=40)

        # Set the labels for the x-axis
        ax.set_xticks([i+1 for i in range(len(distributions))], [distribution.name for distribution in distributions], rotation=45)

        # Add title and labels
        ax.set_title('Strip Chart of Multiple Distributions')
        ax.set_ylabel('Values')

    def sina_plot(distributions, ax, jittering_strength=0.1):
        def density_based_jitter(data):
            # Obtain density
            kde = gaussian_kde(data)
            densities = kde(data)
            # Normalize densities to range between 0 and 0.3
            densities = (densities - densities.min()) / (densities.max() - densities.min())
            # Jittering inversely proportional to density (higher density -> larger jitter)
            jitter = np.random.uniform(-1, 1, size=len(data)) * densities * jittering_strength
            return jitter

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
        ax.set_xticks([i + 1 for i in range(len(distributions))], [distribution.name for distribution in distributions],
                      rotation=45)

        ax.set_title('Sina Plot with Density-Based Jitter')
        ax.set_ylabel('Values')

    def desity_plot_comparison(distrib1, distrib2, ax1, ax2):
        # Join the yaxis
        # Optionally share y-axis between more subplots
        ax1.sharey(ax2)

        # Obtain general distribution
        distribution = pd.concat([distrib1, distrib2])

        # Plot general distribution in both axes:
        # Create density plot (With Seaborn!)
        sns.kdeplot(distribution, ax=ax1, fill=True, color='lightgrey', linewidth=2, label='Full distribution')
        sns.kdeplot(distribution, ax=ax2, fill=True, color='lightgrey', linewidth=2, label='Full distribution')

        # Obtain density scaling factor for each distribution
        scale_factor1 = len(distrib1) / len(distribution)
        scale_factor2 = 1 - scale_factor1

        # For distribution1:
        kde_distrib1 = gaussian_kde(distrib1)

        # Generate the X values for the kde plot
        x_vals = np.linspace(distrib1.min()-1, distrib1.max()+1, 100)
        y_vals = kde_distrib1(x_vals)

        # Plot the distribution 1, scaled down by the factor
        ax1.fill_between(x_vals, y_vals * scale_factor1, color='blue', alpha=0.6, linewidth=3)
        ax1.set_title(f'{distrib1.name} Distribution')

        # For distribution2:
        kde_distrib2 = gaussian_kde(distrib2)

        # Generate the X values for the kde plot
        x_vals = np.linspace(distrib2.min() -1, distrib2.max()+1, 100)
        y_vals = kde_distrib2(x_vals)

        # Plot the distribution 2, scaled down by the factor
        ax2.fill_between(x_vals, y_vals * scale_factor2, color='blue', alpha=0.6, linewidth=3)
        ax2.set_title(f'{distrib2.name} Distribution')

        # Customize the plot
        # Add a shared title for the subplots
        fig.text(0.275, 0.48, "Distribution comparison", ha='center', va='center', fontsize=14)
        ax1.set_ylabel(f'Scaled density')
        ax1.set_xlabel('Values')
        ax2.set_ylabel('Scaled Density')
        ax2.set_xlabel('Values')


    def overlapping_density_plot(distributions, ax):

        # Plot both distributions in ax:
        # Create density plot (With Seaborn!)
        for distribution in distributions:
            sns.kdeplot(distribution, ax=ax, fill=True, color=type_to_color[distribution.name], linewidth=2, label=distribution.name)

        ax.legend(fontsize='small', loc='upper left')
        ax.set_title('Overlapping density plot')
        ax.set_ylabel('Density')
        ax.set_xlabel('Values')

    def age_pyramid_plot(data1, data2, number_of_groups, ax):
        def separate_groups():
            # Create the different bins
            bins = np.linspace(min(data1.min(), data2.min()), max(data1.max(), data2.max()), number_of_groups + 1)
            # Bin data1 and order by bin (index)
            separated_data1 = pd.cut(data1, bins).value_counts().sort_index()
            # Bin data2 and order by bin (index)
            separated_data2 = pd.cut(data2, bins).value_counts().sort_index()

            # Change bin index name for clarity
            index_names = [f'({int(interval.left)}, {int(interval.right)}]' for interval in separated_data1.index]
            separated_data1. index = index_names
            separated_data2.index = index_names

            return separated_data1, separated_data2

        # Separate data into bins
        separated_data1, separated_data2 = separate_groups()

        ax.barh(separated_data1.index, -1 * separated_data1, color='blue', label='<140 kg')

        # Plot female data with positive values (right side)
        ax.barh(separated_data2.index, separated_data2, color='red', label='>=140 kg')

        # Add labels and title
        ax.set_title('Age Pyramid')
        ax.set_xlabel('Price range Count')

        # Add grid and legend
        ax.grid(True, which='major', axis='x', linestyle='--')
        ax.legend()

        # Remove x-ticks on the negative side and format labels correctly
        ax.set_xticks([int(abs(tick)) for tick in ax.get_xticks()])
        ax.set_xlabel('Values')

    full_df = obtain_dataset()
    # Pivot data so bike types are column names, and consumption are values
    df = full_df.pivot(columns='bike type', values='consumption')

    # Clean nulls from the data
    cleaned_data = [df[col].dropna() for col in df.columns]

    fig, axes = plt.subplots(2,4, figsize=(16,8))

    box_plot(cleaned_data, axes[0,0])

    violin_plot(cleaned_data, axes[0,1])

    strip_chart(cleaned_data, axes[0,2], jittering_strength=0.125)

    sina_plot(cleaned_data, axes[0,3], jittering_strength=0.25)

    desity_plot_comparison(cleaned_data[1], cleaned_data[4], axes[1,0], axes[1,1])

    overlapping_density_plot(cleaned_data, axes[1, 2])

    age_pyramid_plot(full_df.loc[full_df['weight full'] < 140, 'price'], full_df.loc[full_df['weight full'] >= 140, 'price'], 10, axes[1,3])

    fig.tight_layout()

    fig.savefig('Images/Visualizing_multiple_distributions.png')

    # Clean plot
    plt.clf()

    # Create the ridgeline plot
    plt.figure(figsize=(10, 6))
    joyplot(cleaned_data, overlap=0.4, figsize=(8, 6), colormap=plt.cm.Spectral)

    # Add a legend manually
    legend_elements = [Line2D([], [], color=type_to_color[type.name], label=type.name, markersize=5, lw=8) for type in cleaned_data]

    # Customize and show plot
    plt.title('Ridgeline Plot')
    plt.tight_layout()
    plt.savefig('Visualizing_multiple_distributions_ridgelineplot.png')


def proportions_visualization():

    def piechart(data, ax):
        # Plot the pie chart
        _, _, autotexts = ax.pie(
            data,
            labels=data.index,
            colors = [type_to_color[type] for type in data.index],
            autopct=lambda pct: f'{int(pct * sum(data) / 100)}',  # Shows exact amounts
            startangle=90
        )

        # Display the quantities in white and bold font
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
            autotext.set_weight('bold')

        ax.set_title('Category piechart')

    def vertical_barplot(data, ax):

        # Make the plot
        ax.bar(data.index, data.values, orientation='vertical', 
               color = [type_to_color[type] for type in data.index])

        # Customize xticks
        ax.set_xticklabels(data.index, rotation=45, ha='right')

        # Customize labels and title
        ax.set_xlabel('Categories')
        ax.set_ylabel('Value Count')
        ax.set_title('Vertical barplot')


    def horizontal_barplot(data, ax):

        # Make the plot
        ax.barh(data.index, data.values, orientation='horizontal',
                color = [type_to_color[type] for type in data.index])

        # Customize labels and title
        ax.set_xlabel('Value Count')
        ax.set_ylabel('Categories')
        ax.set_title('Horizontal barplot')

    def stackedbar_plot(data, ax):

        # Store the different categories names
        categories = data.index

        bottom = 0

        # Create the stacked bar chart
        for i, value in enumerate(data.values):
            ax.bar(1.5, value, bottom=bottom, label=categories[i],color=type_to_color[data.index[i]], width=1.5)

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



    # obtain data
    df = obtain_dataset()
    # Transform data for visualization
    df = df['bike type'].value_counts()

    # Create the figure
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12,4))

    piechart(df, axes[0])

    vertical_barplot(df, axes[1])

    horizontal_barplot(df, axes[2])

    stackedbar_plot(df, axes[3])
    # Set Layout
    fig.tight_layout()

    # Save plot
    fig.savefig('Images/Visualizing proportions.png')


def multiple_group_proportions_visualizations():
    
    def mosaic_plot(data):
        # Create plot
        fig, ax = plt.subplots(figsize=(16, 6))

        # Store all feature names in a list
        features = data.columns.tolist()

        # Obtain how many times each combination apears 
        mosaic_data = data.groupby(features).size()

        # Custom labelizer: do not show anything if it has 0 values
        def custom_labelizer(x):
            if mosaic_data[x] == 0:
                return ''
            else:
                return ('\n ').join(x)
        
        # Make the plot
        mosaic(mosaic_data,  title='Mosaic Plot', labelizer=custom_labelizer, ax=ax)

        # Save plot
        fig.savefig('Images/Visualizing_mosaic_plot.png')


    def tree_map_plot(data):

        # Store all feature names in a list
        features = data.columns.tolist()

        # Obtain how many times each combination apears
        tree_map_data = data.groupby(features).size().reset_index(name='values')

        # Create a treemap
        fig = px.treemap(
            tree_map_data,
            path=features,  # Hierarchical paths
            values='values',  # Values determine the size
            title="Treemap visualization",
        )

        # Show the plot
        fig.write_image("Images/Visualizing_treemap_plot.png")


    def paralel_set_plot(data):

        features = data.columns.to_list()

        # You need a column indication color of the sample
        data['color'] = df['bike type'].map(type_to_color)

        # Create the parallel sets plot
        fig = px.parallel_categories(
            df,
            dimensions=features,
            color='color',  # Optional: assign a constant color for all links
            labels=features,
            title='Parallel set visualization'
        )

        # Show the plot
        fig.write_image('Images/Visualizing_paralel_set_plot.png')


    def density_comparison_matrix(data, plot_range=25):
        '''

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

        axes[0,0].set_xlabel('a')
        # Set a single x and y label for the entire figure
        fig.supylabel(category_name, fontsize=14)
        fig.supxlabel(subcategory_name, fontsize=14)
        fig.suptitle('Density comparation matrix')

        # Save the figure
        fig.tight_layout()
        fig.savefig('Images/Visualizing_Density_Comparison.png')

    
    df = obtain_dataset()
    df = df[['bike type', 'weight full', 'acceleration']]
    df['acceleration'] = pd.cut(df['acceleration'], bins=(0, 25, 1000), labels=['fast', 'slow']).fillna('slow') # Nulls are slow bikes
    weight_continous = df[['acceleration', 'bike type', 'weight full']].copy()
    df['weight full'] = pd.cut(df['weight full'], bins=(0,150,1000), labels=['light', 'heavy'])


    # Any bike that does not reach 100 km/h (null values) is considered slow
    df.loc[df['acceleration'].isnull(), 'acceleration'] = 'slow'

    mosaic_plot(df)

    tree_map_plot(df)

    paralel_set_plot(df)

    density_comparison_matrix(weight_continous, plot_range=25)


def visualizing_uncertainty():
    def frequency_plot(probability, grid_size = 10):
        '''

        :param prob: probability (0.1-1)
        :return:
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
        fig, ax = plt.subplots()
        sns.heatmap(grid, ax=ax, cmap=["white", "blue"], linewidths=0.5, linecolor='gray',
                    cbar=False, square=True, xticklabels=False, yticklabels=False)

        # Set plot limits to ensure the entire grid is shown
        ax.set_xlim(0, grid_size+0.1)
        ax.set_ylim(-0.1, grid_size)

        # Title with probability
        ax.set_title(f"Probability: {probability * 100:.1f}%")
        # Save the image
        fig.tight_layout()
        fig.savefig('Images/Frequency_plot.png')

    def error_bar_plot(data, ax):
        
        # Obtain mean from the data for the bar plot
        data_average = data.mean(axis=0)

        # Obtain sample standard error as a measure of error
        bar_error = data.std(axis=0)

        # Make the barplot
        ax.bar(data_average.index, data_average, yerr=bar_error, capsize=5)

        # Rotate labels
        ax.set_xticklabels(data_average.index, rotation=45, ha='right')

        ax.set_title('Error bar plot')


    def graded_error_bar_plot(data, ax):
 
        def calculate_confidence_interval(data, confidence=0.95):
            mean_std = data.sem()
            critical_value = stats.t.ppf(1 - (1- confidence) / 2, len(data)-1)
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
            ax.barh(i, 2 * ci99, left=mean - ci99, color='skyblue', edgecolor='black', height=0.05, label='99% CI' if i == 0 else "")
            ax.plot([mean - ci99,mean - ci99], [i - 0.05, i + 0.05], color='black', linewidth=1.5)
            ax.plot([mean + ci99,mean + ci99], [i - 0.05, i + 0.05], color='black', linewidth=1.5)
            # Plot the 95% confidence interval and plot limit lines for every bar
            ax.barh(i, 2 * ci95, left=mean - ci95, color='cornflowerblue', edgecolor='black', height=0.1, label='95% CI' if i == 0 else "")
            ax.plot([mean - ci95,mean - ci95], [i - 0.075, i + 0.075], color='black', linewidth=1.5)
            ax.plot([mean + ci95,mean + ci95], [i - 0.075, i + 0.075], color='black', linewidth=1.5)
            # Plot the 80% confidence interval and plot limit lines for every bar
            ax.barh(i, 2 * ci80, left=mean - ci80, color='royalblue', edgecolor='black', height=0.15, label='80% CI' if i == 0 else "")
            ax.plot([mean - ci80,mean - ci80], [i - 0.1, i + 0.1], color='black', linewidth=1.5)
            ax.plot([mean + ci80,mean + ci80], [i - 0.1, i + 0.1], color='black', linewidth=1.5)
            
            # Plot the mean value as a vertical line
            ax.plot([mean, mean], [i - 0.15, i + 0.15], color='black', linewidth=1.5)

        # Plot mean values using a scatterplot
        mean_values = data.mean(axis=0)
        ax.scatter(mean_values, np.arange(len(mean_values)), color = 'orange', marker='o',s=150)

        # Label and ticks
        ax.set_yticks(range(len(data.columns)))
        ax.set_yticklabels(data.columns)
        ax.set_xlabel('Value')
        ax.set_title('Graded Error Bars with Confidence Intervals')

        # Add a legend
        ax.legend(loc='upper right')


    def quantile_dot_plot(data, ax, total_dots, cols):
        '''
            Total_dots: Total dots to represent the entire distribution
            cols: # Number of columns of dots
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
                y_position = diameter * aspect_ratio * (2*j + 1)

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


    df = obtain_dataset()
    # Only obtain data within the same value range 
    error_bar_data = df[['seat height', 'price', 'length','width','height']]

    graded_error_bar_data = pd.DataFrame({
        'Acc < 28': df.loc[df['acceleration'] < 28, 'max speed'],
        'Acc >= 28': df.loc[df['acceleration'] >= 28, 'max speed']
    })

    quantile_dot_plot_data = df['price']

    # Define figure axis
    fig, axes = plt.subplots(1, 3, figsize=(12,4))

    error_bar_plot(error_bar_data, axes[0])

    quantile_dot_plot(quantile_dot_plot_data, axes[1], 40, 10)

    graded_error_bar_plot(graded_error_bar_data, axes[2])

    frequency_plot(0.1)

    fig.tight_layout()

    fig.savefig('Images/visualizing_uncertainty.png')


def visualizing_probability():
    def hyphothetical_outcome_plot(data1, data2, sample_size):

        # Create plot
        fig, ax = plt.subplots()

        # Set vertical line positions
        y1 = 1
        y2 = 2

        ax.set_ylim(0,3)

        # Set vertical bar size
        ybar_size = 0.25
        
        # Set ax ticks
        ax.set_yticks([y1,y2], [data1.name, data2.name])

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
        anim.save("Images/hypothetical_outcome_plot.gif", writer=PillowWriter(fps=2))
        plt.close()


    df = obtain_dataset()

    df = df[['bike type', 'weight full']]

    df = [pd.Series(values, name=category) for category, values in df.groupby("bike type")["weight full"]]

    hyphothetical_outcome_plot(df[-2],df[-1], 5)

def when_barplots_fail():
    def ordered_scatter_plot(data, ax, colors, legend):
        # Order the data
        ordered_data = data.sort_values()

        # Obtain y values for every point and set max y limit
        y_values = np.arange(len(data))
        ax.set_ylim(-1, len(ordered_data))

        # Set grid lines and delete ylabels for everypoint
        ax.set_yticks(y_values, [])
        ax.grid(which='major', axis='y', zorder=0)

        # make the scatterplot
        ax.scatter(ordered_data, y_values, s=50, color=colors, zorder=2)

        # Add the legend with color information
        ax.legend(handles=legend, title="Legend")

        # Set plot aesthetics
        ax.set_xlabel('Feature values')
        ax.set_title('Ordered scatter plot')

    def ordered_heatmap(data, ax, columns):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Obtain data range
        xmin = min(chain(*data))
        xmax = max(chain(*data))

        # Step 1: Define intervals and bin data
        bins = np.linspace(xmin, xmax, columns)  # Adjust range and bins as necessary
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




    # Create subplot
    fig, axes = plt.subplots(1,2, figsize=(10,8))
    
    # Obtain data
    df = obtain_dataset()
    df = df[['weight full', 'bike type']]

    ordered_heatmap_data = [pd.Series(values, name=category) for category, values in df.groupby("bike type")["weight full"]]

    # Custom legend creation
    legend = [
        Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10)
        for label, color in type_to_color.items()
    ]

    ordered_scatter_plot(df['weight full'], axes[0], df['bike type'].map(type_to_color), legend)

    ordered_heatmap(ordered_heatmap_data, axes[1], 10)

    # Save the figure
    plt.tight_layout()
    fig.savefig('Images/when_barplots_fail.png')



visualizing_uncertainty()