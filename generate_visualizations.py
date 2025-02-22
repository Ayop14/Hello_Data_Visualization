
# Generate image examples of all visualization types in the visualization repository
from matplotlib import pyplot as plt
from Visualization_repository import  *
from project.read_data import obtain_dataset

df = obtain_dataset()


# Histogram plot ----------------------------------------------------
fig, ax = plt.subplots()

# Make the plot
bins = 10
histogram(df['price'], bins, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Histogram.png')
plt.close()


# Polar plot ----------------------------------------------------
# Make the plot (AX not required)
times_around = 1
fig, ax = polar(df['price'], times_around)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Polar_plot.png')
plt.close()


# Scatterplot ----------------------------------------------------
fig, ax = plt.subplots()

# Make the plot
x = df['price']
y = df['max speed']
scatterplot(x,y,ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Scatterplot.png')
plt.close()



# Color Scatterplot ----------------------------------------------------
fig, ax = plt.subplots()

# Make the plot
x = df['price']
y = df['max speed']
color = df['acceleration']
color_scatterplot(x, y, color, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Color_Scatterplot.png')
plt.close()


# Bubblechart plot ----------------------------------------------------
fig, ax = plt.subplots()

# Make the plot
x = df['price']
y = df['max speed']
size = df['acceleration']
bubble_chart(x, y, size, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Bubblechart_plot.png')
plt.close()


# Grouped bars plot ----------------------------------------------------
fig, ax = plt.subplots()

# Make the plot
matrix_data = df[['brand', 'bike type']].pivot_table(index='brand', columns='bike type', aggfunc='size', fill_value=0)
matrix_data = matrix_data.loc[['Honda', 'Kawasaki', 'Suzuki', 'Yamaha']]
grouped_bars_plot(matrix_data, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Grouped_bars_plot.png')
plt.close()


# Stacked bars plot ----------------------------------------------------
fig, ax = plt.subplots()

# Make the plot
matrix_data = df[['brand', 'bike type']].pivot_table(index='brand', columns='bike type', aggfunc='size', fill_value=0)
matrix_data = matrix_data.loc[['Honda', 'Kawasaki', 'Suzuki', 'Yamaha']]
stackedbars_plot(matrix_data, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Stacked_bars_plot.png')
plt.close()


# Heatmap plot ----------------------------------------------------
fig, ax = plt.subplots()

# Make the plot
matrix_data = df[['brand', 'bike type']].pivot_table(index='brand', columns='bike type', aggfunc='size', fill_value=0)
matrix_data = matrix_data.loc[['Honda', 'Kawasaki', 'Suzuki', 'Yamaha']]
heatmap_plot(matrix_data, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Heatmap_plot.png')
plt.close()


# Density plot ----------------------------------------------------
fig, ax = plt.subplots()

# Make the plot
density_plot(df['price'], ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Density_plot.png')
plt.close()


# Cumulative density plot ----------------------------------------------------
fig, ax = plt.subplots()

# Make the plot
cumulative_density_plot(df['price'], ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Cumulative_density_plot.png')
plt.close()


# Quantile-quantile plot ----------------------------------------------------
fig, ax = plt.subplots()

# Make the plot
quantile_quantile_plot(df['price'], ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Quantile_quantile_plot.png')
plt.close()


# Box plot ----------------------------------------------------
fig, ax = plt.subplots()

# Make the plot
aux = df.pivot(columns='bike type', values='consumption')
box_plot(aux, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Box_plot.png')
plt.close()


# Violin plot ----------------------------------------------------
fig, ax = plt.subplots()

# Make the plot
aux = df.pivot(columns='bike type', values='consumption')
violin_plot(aux, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Violin_plot.png')
plt.close()


# Strip chart plot ----------------------------------------------------
fig, ax = plt.subplots()

# Make the plot
aux = df.pivot(columns='bike type', values='consumption')
strip_chart(aux, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Strip_chart_plot.png')
plt.close()


# Sina plot ----------------------------------------------------
fig, ax = plt.subplots()

# Make the plot
aux = df.pivot(columns='bike type', values='consumption')
sina_plot(aux, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Sina_plot.png')
plt.close()


# Overlapping density plot ----------------------------------------------------
fig, ax = plt.subplots()

# Make the plot

overlapping_density_plot(aux, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Overlapping_density_plot.png')
plt.close()


# Density plot comparison ----------------------------------------------------
fig, axes = plt.subplots(2,1)

# Make the plot
aux = df.pivot(columns='bike type', values='consumption')
data1 = aux.loc[:, 'Scooter']
data2 = aux.loc[:, 'Naked']
density_plot_comparison(data1, data2, axes[0], axes[1])

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Density_comparison_plot.png')
plt.close()


# Age pyramid plot ----------------------------------------------------
fig, ax = plt.subplots()

# Make the plot
aux = df.pivot(columns='bike type', values='consumption')
data1 = aux.loc[:, 'Scooter']
data2 = aux.loc[:, 'Naked']
n_groups = 10
age_pyramid_plot(data1, data2, n_groups, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Age_pyramid_plot.png')
plt.close()


# Piechart visualization ----------------------------------
fig, ax = plt.subplots()

# Make the plot
aux = df['bike type']
piechart(aux, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Piechart.png')
plt.close()


# Vertical bar plot visualization ----------------------------------
fig, ax = plt.subplots()

# Make the plot
aux = df['bike type']
vertical_barplot(aux, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Vertical_bar_plot.png')
plt.close()


# Horizontal visualization ----------------------------------
fig, ax = plt.subplots()

# Make the plot
aux = df['bike type']
horizontal_barplot(aux, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Horizontal_bar_plot.png')
plt.close()


# Single stacked bar visualization ----------------------------------
fig, ax = plt.subplots()

# Make the plot
aux = df['bike type']
single_stackedbar_plot(aux,ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Single_stacked_bar_plot.png')
plt.close()


# Mosaic plot visualization -----------------------------
fig, ax = plt.subplots()

# Make the plot
aux = df[['bike type', 'max speed']].copy()
aux['max speed'] = pd.cut(aux['max speed'], bins = [0,105,100000], labels = ['slow', 'fast'])
mosaic_plot(aux, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Mosaic_plot.png')
plt.close()


# Tree map plot visualization -----------------------------
# (Cant use pyplot. Saved directly)
# Make the plot
aux = df[['bike type', 'max speed']].copy()
aux['max speed'] = pd.cut(aux['max speed'], bins = [0,105,100000], labels = ['slow', 'fast'])
tree_map_plot(aux, 'Images/Tree_map_plot.png')


# Paralel set plot visualization -----------------------------
# (Cant use pyplot. Saved directly)
# Make the plot
aux = df[['bike type', 'max speed']].copy()
aux['max speed'] = pd.cut(aux['max speed'], bins = [0,105,100000], labels = ['slow', 'fast'])
# Replace NaNs with "unknown"
aux['max speed'] = aux['max speed'].cat.add_categories('unknown').fillna('unknown')
paralel_set_plot(aux, 'Images/Parallel_set_plot.png')


# Density comparison matrix visualization -----------------------------

# Make the plot
aux = df[['bike type', 'max speed', 'weight full']].copy()
aux['max speed'] = pd.cut(aux['max speed'], bins = [0,105,100000], labels = ['slow', 'fast'])
density_comparison_matrix(aux,'Images/Density_comparison_matrix.png', plot_range=25)
plt.close()


# Frequency plot -----------------------------
fig, ax = plt.subplots()

# Make the plot
frequency_plot(0.3, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Frequency_plot.png')
plt.close()


# Error bar plot -----------------------------
fig, ax = plt.subplots()

# Make the plot
aux = df[['weight full', 'acceleration', 'max speed']]
error_bar_plot(aux, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Error_bar_plot.png')
plt.close()


# Graded error bar plot -----------------------------
fig, ax = plt.subplots()

# Make the plot
aux = df[['weight full', 'acceleration', 'max speed']]
graded_error_bar_plot(aux, ax, 0.5)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Graded_error_bar_plot.png')
plt.close()


# Quantile dot plot -----------------------------
fig, ax = plt.subplots()

# Make the plot
aux = df['weight full']
quantile_dot_plot(aux, ax, 80, 15)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Quantile_dot_plot.png')
plt.close()


# Hypothetical outcome plot -----------------------------
fig, ax = plt.subplots()

# Make the plot
mask = df['max speed'] >= 105
aux1 = df.loc[mask, 'weight full'].copy()
aux1.name = 'Fast weight'
aux2 = df.loc[~mask, 'weight full'].copy()
aux2.name = 'Slow weight'
hyphothetical_outcome_plot(aux1, aux2, 10, 'Images/Hyphothetical_outcome_plot.gif')


# Ordered scatter plot -----------------------------
fig, ax = plt.subplots()

# Make the plot
aux = df[['weight full', 'bike type']]
ordered_scatter_plot(aux, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Ordered_scatter_plot.png')
plt.close()


# Ordered heatmap plot -----------------------------
fig, ax = plt.subplots()

# Make the plot
aux = df[['weight full', 'bike type']]
ordered_heatmap(aux, ax, 6)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Ordered_heatmap_plot.png')
plt.close()


# correlogram plot -----------------------------
fig, ax = plt.subplots()

# Make the plot
aux = df[['weight full', 'price', 'acceleration', 'max speed', 'gasoline capacity', 'torque']]
correlogram(aux, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Correlogram_plot.png')
plt.close()


# Scatterplot matrix plot -----------------------------

# Make the plot
aux = df[['weight full', 'price', 'acceleration', 'max speed', 'gasoline capacity', 'torque']]
scatterplot_matrix(aux, 'Images/Scatterplot_matrix.png')


# Slopegraph plot -----------------------------
fig, ax = plt.subplots()

# Make the plot
aux = df.loc[:10,['weight full', 'acceleration', 'max speed']]
slopegraph(aux, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Slopegraph.png')
plt.close()


# connected scatterplot -----------------------------
fig, ax = plt.subplots()

# Make the plot
aux = df[['price', 'acceleration']]
connected_scatterplot(aux, ax)

# Format and store the image
fig.tight_layout()
fig.savefig('Images/Connected_scatterplot.png')
plt.close()


# Divide time series into components -----------------------------

# Make the plot
aux = df['price']
divide_time_series_into_components(aux, 'Images/Time_series_into_components.png')
































