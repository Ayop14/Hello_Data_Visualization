
import pandas as pd
from read_data import obtain_dataset
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
from scipy.stats import probplot, gaussian_kde
from joypy import joyplot
from statsmodels.graphics.mosaicplot import mosaic
from itertools import product

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
    
    def mosaic_plot(data, ax):

        # Store all feature names in a list
        features = data.columns.tolist()

        # Obtain how many times each combination apears 
        mosaic_data = data.groupby(features).size().reset_index()
        
        # Make the plot
        mosaic(mosaic_data, features, title='Mosaic Plot', labelizer=lambda x: x, ax=ax)

    
    df = obtain_dataset()
    df = df[['bike type', 'weight full', 'acceleration']]
    df['weight full'] = pd.cut(df['weight full'], bins=(0,140,160,1000), labels=['light', 'medium weight', 'heavy'])
    df['acceleration'] = pd.cut(df['acceleration'], bins=(0,20,35,1000), labels=['fast', 'normal speed', 'slow'])

    # Any bike that does not reach 100 km/h (null values) is considered slow
    df.loc[df['acceleration'].isnull(), 'acceleration'] = 'slow'

    fig, ax = plt.subplots(figsize=(16,6))

    print(df)
    
    mosaic_plot(df, ax)

    fig.savefig('Images/Visualizing_multiple_proportions.png')


multiple_group_proportions_visualizations()