# Import data processing libraries
import pandas as pd
import numpy as np
from itertools import chain

# Import visualization libraries
import seaborn as sns
from joypy import joyplot
import plotly.express as px

import matplotlib.pyplot as plt
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
        data: array of values to plot (series, list, numpy array)
        bins: number of bins to make the histogram with
        ax: pyplot ax to plot the visualization in
    '''
    series = pd.Series(data)
    # Visualize weights using a histogram
    ax.hist(series, bins = bins)
    # Plot 0.25, 0.75 quartiles as vertical bars
    ax.axvline(series.quantile(0.75), color='red')
    ax.axvline(series.quantile(0.25), color='red')

    # Format the axes
    ax.set_xlabel('Variable value')
    ax.set_ylabel('Value counts')
    ax.set_title('Histogram plot')


def polar(data, times_around):
    '''
    Highly customizable plot, this just works as a base
        data: array of values to plot (series, list, numpy array)
        times_around: number of times to make the data go around in the polar axes
        ax: Not included because it needs aditional parameters in creation
    '''
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    # Create angles from 0 to 2π (one time around)
    angles = np.linspace(0, 2 * np.pi * times_around, len(data))
    ax.plot(angles, data, marker='o', markersize=4)
    ax.set_title("Polar plot")

    return fig, ax


def scatterplot(x, y, ax):
    '''
        x: array of values to plot (series, list, numpy array)
        y: array of values to plot (series, list, numpy array)
        ax: pyplot axes to make the scatterplot
    '''
    # Make the scatterplot. Simple
    ax.scatter(x, y, s=50, edgecolor='k')

    # Add axis labels and title
    ax.set_xlabel('X units')
    ax.set_ylabel('Y units')
    ax.set_title('Scatterplot')
    

def color_scatterplot(x, y, color_variable, ax):
    '''
        x: array of values to plot (series, list, numpy array)
        y: array of values to plot (series, list, numpy array)
        color_variable: array of values to plot (series, numpy array)
        ax: pyplot ax to plot the visualization in
    '''
    # Obtain min and max for scailing
    color_min = color_variable.min()
    color_max = color_variable.max()
    
    # Obtain colors from continous variable
    colors = plt.cm.coolwarm((color_variable- color_min)/(color_max-color_min)) # Normalize values

    # Scatterplot with torque, weight and seat height
    scatter = ax.scatter(x, y, c=colors, s=50, edgecolor='k')

    # Add a colorbar to map colors to values
    colorbar = plt.colorbar(scatter, ax=ax)
    colorbar.set_label(df_3.name)

    # Add axis labels and title
    ax.set_xlabel(df_1.name)
    ax.set_ylabel(df_2.name)
    ax.set_title('Scatterplot with color-encoded continous variable')

    # Save the figure
    fig.savefig(store)

def buble_chart(x, y, size_variable, ax):
    '''
        x: array of values to plot (series, list, numpy array)
        y: array of values to plot (series, list, numpy array)
        color_variable: array of values to plot (series, list, numpy array)
        ax: pyplot ax to plot the visualization in
    '''


    


