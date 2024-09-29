
import pandas as pd
from read_data import obtain_dataset
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

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
    type_to_shape_dict = {
        'Naked':'^',
        'Sport':'o',
        'Custom':'s',
        'Scrambler':'p',
        'Scooter':'*',
        'Trail': 'P'
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


log_info()