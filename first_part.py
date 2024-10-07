
import pandas as pd
from read_data import obtain_dataset
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
from scipy.stats import probplot


type_to_shape_dict = {
        'Naked':'^',
        'Sport':'o',
        'Custom':'s',
        'Scrambler':'p',
        'Scooter':'*',
        'Trail': 'P'
    }

# Turn weight discrete. What are the different weight values? Lets see with a simple histogram
def weight_visualization():
    # Obtain data
    data = obtain_dataset()
    weight = data['weight full']
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
    df = obtain_dataset()
    acceleration = df['acceleration']

    # log transformed acceleration
    log_transformed_acceleration = np.log10(acceleration)

    # Define figure
    fig, axes = plt.subplots(3,1, )


    # deactivate y values
    plt.tick_params(left=False)

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
    axes[0].set_title('Original Acceleration, linear scale')
    axes[0].scatter(acceleration, [0] * len(acceleration))

    # Plot Log transformed values
    axes[1].set_title('Log transformed Acceleration, linear scale')
    axes[1].scatter(log_transformed_acceleration, [0] * len(acceleration))

    # Plot Log scale values
    axes[2].set_xscale('log')
    axes[2].set_title('Original Acceleration, logarithmic scale')
    axes[2].scatter(acceleration, [0] * len(acceleration))


    # Adjust vertical padding
    fig.subplots_adjust(hspace=1)  # adjust padding
    # Save fig
    fig.savefig('Images/Visualizing log acceleration initial')


def polar_coordinate_visualization():
    # obtain data
    df = obtain_dataset()
    capacity = df[['bike type', 'gasoline capacity']]

    # Create a figure and a polar subplot
    fig, axes = plt.subplots(2,2,figsize=(8,8), subplot_kw={'projection': 'polar'})

    for x in range(2):
        for y in range(2):
            # Set label position
            axes[x,y].set_rlabel_position(45)
            # Turn on xticks, but without labels
            axes[x,y].set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2], [])

            # Set height values
            axes[x,y].tick_params(axis='y', pad=-1)
            axes[x,y].set_yticks([5, 10, 15, 20])

            # Optionally add gridlines and a title
            axes[x,y].grid(True)

    # First plot: One lap around the polar coordinate system
    # Create angles from 0 to 4π (two times around)
    angles = np.linspace(0, 2 * np.pi, len(capacity))
    axes[0,0].plot(angles, capacity['gasoline capacity'], marker='o', markersize=4)
    axes[0,0].set_title("Gasoline capacity - One lap")

    # Second Plot: Two laps around the polar coordinate system
    angles = np.linspace(0, 4 * np.pi, len(capacity))
    axes[0,1].plot(angles, capacity['gasoline capacity'], marker='o', markersize=4)
    axes[0,1].set_title("Gasoline capacity - Two laps")

    # Third plot: Comparation against ech other bike type using line plot
    for bike_type in capacity['bike type'].unique():
        bike_type_capacities = capacity.loc[capacity['bike type'] == bike_type, 'gasoline capacity']
        angles = np.linspace(0, 2 * np.pi, len(bike_type_capacities))
        axes[1,0].plot(angles, bike_type_capacities, marker=type_to_shape_dict[bike_type], markersize=4, label=bike_type)

    axes[1,0].set_title("Gasoline capacity - Linear plot")
    axes[1,0].legend(loc='lower right', bbox_to_anchor=(1.25, -0.2))

    # Forth plot: Comparation against ech other bike type using line plot
    for bike_type in capacity['bike type'].unique():
        bike_type_capacities = capacity.loc[capacity['bike type'] == bike_type, 'gasoline capacity']
        angles = np.linspace(0, 2 * np.pi, len(bike_type_capacities))
        axes[1, 1].scatter(angles, bike_type_capacities, marker=type_to_shape_dict[bike_type], s=40, label=bike_type)

    axes[1, 1].set_title("Gasoline capacity - Scatterplot")
    axes[1, 1].legend(loc='lower right', bbox_to_anchor=(1.25, -0.2))

    fig.savefig('Images/Visualizing polar coordinates')


def color_powerpoint():
    # Obtain data
    df = obtain_dataset()

    # Create Scatterplot
    fig, ax = plt.subplots()

    # Scatterplot with torque, weight and seat height
    scatter = ax.scatter(df['torque'], df['weight full'], c=df['seat height'], s=50, edgecolor='k' )

    # Add a colorbar to map colors to values
    colorbar = plt.colorbar(scatter, ax=ax)
    colorbar.set_label('Seat height (mm)')

    # Add axis labels and title
    ax.set_xlabel('Torque (Nm)')
    ax.set_ylabel('Weight (kg)')
    ax.set_title('Scatterplot with color-encoded continous variable')

    # Save the figure
    fig.savefig('Images/Visualizing color importance - scatterplot')


    # Create a histogram comparison
    fig, axes = plt.subplots(1,2, figsize=(14,4))

    # Obtain/Filter data
    df = df.loc[df['bike type']=='Trail', ['name', 'price']]

    # Plot regular barplot
    x_values = np.arange(len(df))
    axes[0].bar(x_values, df['price'], color='blue', edgecolor='k')

    # Set Xtick label values
    axes[0].set_xticks(x_values, df['name'], rotation=20, fontsize=10)
    # Set title
    axes[0].set_title('Barplot comparing all Trail bikes prices')
    axes[0].set_ylabel('Price (Euros)')

    # Plot color variation barplot
    df.loc[:,'colors'] = 'lightsteelblue'
    df.loc[df['name'] == 'Montana XR1','colors'] = 'blue'
    x_values = np.arange(len(df))
    axes[1].bar(x_values, df['price'], color=df['colors'], edgecolor='k')

    # Set Xtick label values
    axes[1].set_xticks(x_values, df['name'], rotation=20, fontsize=10)
    # Set title
    axes[1].set_title('Barplot comparing Montana XR1 price against te others')


    plt.tight_layout()
    #plt.subplots_adjust(bottom=0.2)

    # Save figure
    fig.savefig('Images/Visualizing histogram color effect')


def different_amounts_visualization():
    # obtain data
    df = obtain_dataset()

    # Create plot
    fig, axes = plt.subplots(1,3,figsize=(16,6))

    # Set the positions for the bars on the x-axis
    bar_width = 0.2  # Width of the bars

    # Transform data for visualization
    df = df.pivot_table(index='brand', columns='bike type', aggfunc='size', fill_value=0)
    df = df.loc[['Honda', 'Kawasaki', 'Suzuki', 'Yamaha']]

    n_categories = len(df.index)
    n_subcategories = len(df.columns)


    # Create grouped bars plot
    x = np.arange(n_categories)  # Positions for the groups (categories)

    # Plot each subgroup in the bars, with slight shifts for each bar within a group
    for i in range(n_subcategories):
        axes[0].bar(x + i * bar_width, df.iloc[:, i], width=bar_width, label=df.columns[i])

    # Set the x-axis labels and ticks
    axes[0].set_xticks(x + bar_width * (n_subcategories - 1) / 2, df.index)  # Center the ticks

    # Add labels and title
    axes[0].set_xlabel('Bike Brand')
    axes[0].set_ylabel('Quantity')
    axes[0].set_title('Bike types divided by brands')


    # Create stacked bars plot
    # Initialize the bottom of the bars to zero (stacking from the bottom)
    bottom = np.zeros(n_categories)

    # Create the stacked bar chart
    for i in range(n_subcategories):
        axes[1].bar(x, df.iloc[:, i], bottom=bottom, label=df.columns[i])
        bottom += df.iloc[:, i]  # Update the bottom position for stacking

    # Set the x-axis ticks and labels
    axes[1].set_xticks(x, df.index)

    # Add labels and title
    axes[1].set_xlabel('Bike Brand')
    axes[1].set_ylabel('Quantity')
    axes[1].set_title('Bike types divided by brand')


    # Create Heatmap for last visualization
    # Create the heatmap using imshow
    cax = axes[2].imshow(df, cmap='viridis', aspect='auto')

    # Set row and column labels
    axes[2].set_xticks(np.arange(len(df.columns)), df.columns)
    axes[2].set_yticks(np.arange(len(df.index)), df.index)

    # Rotate the x-axis labels for better readability
    plt.setp(axes[2].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add a color bar to show the intensity scale
    fig.colorbar(cax, ticks=np.arange(df.max(axis=None)+1))

    # Set labels and title
    axes[2].set_xlabel('Bike Brand')
    axes[2].set_ylabel('Quantity')
    axes[2].set_title('Bike types divided by brands')

    # Show the plot
    plt.tight_layout()

    plt.savefig('Images/Visualizing_different_amounts.png')

def visualizing_single_distribution(feature):
    # Obtain dataset
    df = obtain_dataset()

    # Create a 2,2 subplot
    fig, axes = plt.subplots(2,2, figsize=(8,8))

    # Create a histogram
    axes[0,0].hist(df[feature], bins=15)

    axes[0,0].set_title(f'Histogram of {feature}')
    axes[0, 0].set_xlabel(f'{feature} values')
    axes[0,0].set_ylabel('Value counts')


    # Create density plot (With Seaborn!)
    sns.kdeplot(df[feature], ax=axes[0,1], fill=True, color='steelblue', linewidth=2)

    # Customize the plot
    axes[0,1].set_title(f'Density Plot of {feature}', fontsize=14)
    axes[0, 0].set_xlabel(f'{feature} values')
    axes[0,1].set_ylabel('Density')



    # Make a cumulative density plot
    sorted_feature = np.sort(df[feature])


    # Compute the cumulative values (normalized to 1)
    cumulative_values = np.arange(1, len(sorted_feature) + 1) / len(sorted_feature)

    # Plot the cumulative density function (CDF)
    axes[1,0].step(sorted_feature, cumulative_values, color='steelblue')

    # Set yticks manually
    axes[1,0].set_yticks(np.arange(0.2,1.2,0.2))

    # Customize the plot with title and labels
    axes[1,0].set_title(f'Cumulative Density Plot of {feature}')
    axes[1,0].set_xlabel(f'{feature} values')
    axes[1,0].set_ylabel('Cumulative Density')



    # Make a quantile quantile plot
    # Remove null values
    clean_feature = df.loc[df[feature].notnull(), feature]
    # Generate Q-Q plot data using scipy's probplot function
    (theoretical_quantiles, sample_quantiles), (slope, intercept, _) = probplot(clean_feature, dist="norm")

    # Plot the Q-Q plot (theoretical quantiles vs. sample quantiles)
    axes[1,1].scatter(theoretical_quantiles, sample_quantiles, color='blue', label='Sample Quantiles')

    # Plot the 45-degree line using the slope and intercept from probplot
    axes[1,1].plot(theoretical_quantiles, slope * theoretical_quantiles + intercept, color='red', linestyle='--', label='y = x (reference)')

    # Set a grid for clearer visualization
    axes[1,1].grid(True)

    # Customize the plot with title and labels
    axes[1,1].set_title(f'Q-Q Plot of {feature}')
    axes[1,1].set_xlabel('Theoretical Quantiles')
    axes[1,1].set_ylabel('Sample Quantiles')
    

    plt.tight_layout()
    fig.savefig('Images/Visualizing_single_distribution.png')


    # 2 Figure: Comparing Histogram parameters
    fig, axes = plt.subplots(4,4, figsize=(10,10), sharey=True)

    for i, ax in enumerate(axes.flatten()):
        # Create a histogram
        ax.hist(df[feature], bins=2 * i + 1)

        # Set title
        ax.set_title(f'Bins: {2*i + 1}')

        # Turn off axis labels to declutter the grid
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f'Histogram bin size comparison for {feature}', fontsize=14)

    fig.tight_layout()
    fig.savefig('Images/Visualizing_single_distribution_hist_params.png')

    # 3 Figure: Comparing Density plot parameters



visualizing_single_distribution('seat height')