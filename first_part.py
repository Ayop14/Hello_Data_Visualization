
import pandas as pd
from read_data import obtain_dataset
import matplotlib.pyplot as plt

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
                consumption: continous
                acceleration: continous
                max speed: continous
                bike type: discrete
                weight : continous - turn discrete
    :return:
    '''
    # import data
    df = obtain_dataset()
    # Turn discrete the weight



weight_visualization()