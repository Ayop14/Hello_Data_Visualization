# Visualization repository

This file is meant to be a summary of all visualizations created, as well as their general use and location.

# ***General useful plots***

## Quantile-quantile plot

![Quantile dot plot](Images/Quantile_dot_plot.png)

It can be used to assess wether or not data follows a certain distribution or not.  

Parameters are a pandas series to plot and ax to make the plot in. 

[**Code**](Visualization_repository.py#L274-301)

# ***Bar plots***
## Histogram 

![Heatmap](Images/Histogram.png)

Good to visualize continous values. Only that it requires visualization parameters (Bins)

Parameters are a pandas Series to split, number of bins, and ax to make the plot.

[**Code**](Visualization_repository.py#L24-L40)

## Grouped bars plot

![Grouped bars plot](Images/Grouped_bars_plot.png)

Great way to show a small set of features from independent entities. Do not use if any of the bars is 0, better use a stacked bar plot. 

Parameters are a matrix where every row is a category, and column a subcategory. Pivot_table + aggfunc=Size will be useful. AQnd ax to plot.

[**Code**](Visualization_repository.py#L140-L171)

## Stacked bars plot

![Stacked bars plot](Images/Stacked_bars_plot.png)

Great way to show a small set of features from independent entities. Do not use if all bars are > 0, better use a grouped bar plot. 

Parameters are a matrix where every row is a category, and column a subcategory. Pivot_table + aggfunc=Size will be useful. AQnd ax to plot.

[**Code**](Visualization_repository.py#L140-L171)

## Vertical plot

![Vertical bar plot](Images/Vertical_bar_plot.png)

Provides details of specific amounts

Parameters are a pandas categorical series to plot. (Doesnt have to be category type) 

[**Code**](Visualization_repository.py#L304-L335)

## Horizontal plot

![Horizontal bar plot](Images/Horizontal_bar_plot.png)

Provides details of specific amounts. Alternative to vertical when labels are a problem to visualize

Parameters are a pandas categorical series to plot. (Doesnt have to be category type) 

[**Code**](Visualization_repository.py#L644-L659)

## Piechart plot

![Piechart plot](Images/Piechart.png)

Compares the proportions of data

Parameters are a pandas categorical series to plot. (Doesnt have to be category type) 

[**Code**](Visualization_repository.py#L597-L617)

# ***Scatterplots***

## Scatterplot visualization

![Scatterplot visualization](Images/Scatterplot.png)

Good to compare variable relationships.

Parameters are two independent x and y pandas Series to plot and an ax to make the visualization

[**Code**](Visualization_repository.py#L61-L73)

## Color Scatterplot visualization

![Color Scatterplot](Images/Color_Scatterplot.png)

Good to compare even more variable relationships. 

Parameters are three independent x,y and color variable pandas Series to plot and an ax to make the visualization

[**Code**](Visualization_repository.py#L76-L102)

## Bubble chart (Size Scatterplot) visualization

![Bublechart](Images/Bubblechart_plot.png)

Good to compare even more variable relationships. 

Parameters are three independent x,y and size variable pandas Series to plot and an ax to make the visualization

[**Code**](Visualization_repository.py#L106-L136)

## Ordered Scatterplot visualization

![Ordered Scatterplot](Images/Ordered_scatter_plot.png)

Good to compare categorical with continous data. It solves the problem of very high, similar data visualized with histograms. 

Parameters are three independent x,y and color variable pandas Series to plot and an ax to make the visualization

[**Code**](Visualization_repository.py#L1061-L1110)

## Scatterplot Matrix visualization

![Scatterplot Matrix](Images/Scatterplot_matrix.png)

Detailed way to display the relations between a bunch of variables. 

Parameters are a dataframe with continous values, where every column will be a variable. And a file path to be stored. 

[**Code**](Visualization_repository.py#L1208-L1242)

## Connected Scatterplot visualization

![Connected scatterplot](Images/Connected_scatterplot.png)

Compare the temporal relations between two variables

Parameters are a dataframe with 2 columns, first will be x second y. An ax to make the plot, and optionally a tuple to display label offset.

[**Code**](Visualization_repository.py#L1272-L1295)

# ***Heatmaps***

## Heatmap plot

![Heatmap plot](Images/Heatmap_plot.png)

Great way to show a huge amount of data. It avoids overplotting with color/density information. 

Parameters are a matrix where every row is a category (Row of the heatmap), and column a subcategory (Column of the heatmap). Pivot_table + aggfunc=Size will be useful. AQnd ax to plot.

[**Code**](Visualization_repository.py#L208-L232)

## Ordered Heatmap

![Ordered heatmap](Images/Polar_plot.png)

Great way to show a huge amount of data, with an ordered axis and the number of columns/intervals the data is goinng to be divided by. 

Parameters are a pandas series to plot and the number of times to go around the axis

[**Code**](Visualization_repository.py#L1112-L1151)


# ***Density plots***

## Density plot

![Density plot](Images/Density_plot.png)

Great way to show information about a distribution as a whole. 

Parameters are a pandas series to plot and ax to make the plot in. 

[**Code**](Visualization_repository.py#L235-L246)

## Cumulative Density plot

![Cumulative Density plot](Images/Cumulative_density_plot.png)

Provides helpful, specific information about a distribution.  

Parameters are a pandas series to plot and ax to make the plot in. 

[**Code**](Visualization_repository.py#L248-L271)

## Box plot

![Box plot](Images/Box_plot.png)

Compares very specific details of distributions among one another.

Parameters are a pandas series or Series to plot and ax to make the plot in. Every column will be one distribution to plot. It can contain na that will be ignored, in case they are diferent sizes. 

[**Code**](Visualization_repository.py#L304-L335)

## Violin plot

![Violin plot](Images/Violin_plot.png)

Compares general distribution shapes against one another 

Parameters are a pandas series or Series to plot and ax to make the plot in. Every column will be one distribution to plot. It can contain na that will be ignored, in case they are diferent sizes. 

[**Code**](Visualization_repository.py#L338-L372)

## Strip chart

![Strip chart plot](Images/Strip_chart_plot.png)

Compares Distribution with specific data points distribution. Works better with smaller amounts of data.

Parameters are a pandas series or Series to plot and ax to make the plot in. Every column will be one distribution to plot. It can contain na that will be ignored, in case they are diferent sizes.
There is the option to add jitter with jittering_strengh parameter. It measures how much points deviate from the vertical line that is its distributon. 

[**Code**](Visualization_repository.py#L374-L414)

## Sina plot

![Sina plot](Images/Sina_plot.png)

Combines Violin with stripchart. It implements by default density based jitter (More jitter the closer to the mean)

Parameters are a pandas series or Series to plot and ax to make the plot in. Every column will be one distribution to plot. It can contain na that will be ignored, in case they are diferent sizes. 

[**Code**](Visualization_repository.py#L417-L465)

## Overlapping density plot

![Overlapping Density plot](Images/Overlapping_density_plot.png)

Compares directly distributions one on top of the other

Parameters are a pandas series or Series to plot and ax to make the plot in. Every column will be one distribution to plot. It can contain na that will be ignored, in case they are diferent sizes. 

[**Code**](Visualization_repository.py#L468-L490)

## Density plot Comparison

![Density plot comparison](Images/Density_comparison_plot.png)

Compares two segments of a distribution as a whole, providing insight of the differences between the data and their contributions to the whole

Parameters are a two pandas series or Series to plot and the two axes to plot them in. Flexibility adds margin to the x limits of the plot. 

[**Code**](Visualization_repository.py#L492-548)

# ***Other case specific visualizations***
## Age pyramid plot

![Age pyramid plot](Images/Age_pyramid_plot.png)

Divides a distribution by a variable. commonly used for age

Parameters are two pandas series or Series to plot an ax to make the plot in and the number of groups (divisions) to make on the y axis. 

[**Code**](Visualization_repository.py#L552-L597)
