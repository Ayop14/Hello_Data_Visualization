# Project diary
the idea is to write down things that acme to mind while making certain parts of the project. This will make easier returning to the project about how did I acomplished certain requirement.

As a Junior, Im also experimenting with diferent types of codings. In the middle of the project, I decided to start making the visualization code more appealing, easier to read, or to reuse the code in some other project, isolating the visualization code inside another method. That will allow the repository to be a visualization repository as well. 

## First part
### Five feature visualization
First thing that called my atention was the fact 5 fetures could be visualized at the same time in a scatterplot. They are often not useful, but it can be helpful to know how could they be made.

First define each of the 5 variables as qualitative or continous. 

The turn weight and max speed into categories to make the difference more notable. However, continous variables are less burden for the eye. And lets you use the right hand y axis for a color bar, resembling more a bubblechart directly.

To turn from continous to discrete, use pandas function cut. Then a dictionary to map each feature value to a *aesthetic*. 

there is no posibility to provide a marker list, so using a scatter plot each indpendent shape feature value. 

There is the posibility to work with plt, or fig,ax using plt.subplots(). plt works great for quick visualization, but ax/fig is way more customizable, specially when working with multiple ilustrations in one figure.

you can use multiple legends with their own tittle. Just create the legend, store the result and add the artist (result) to the figure. 

Figure legends are just invisible line with markers form a different color. Use color='w' to make invisible the line, and markerfacecolor for the color you want that part of the legend.
Legend titles are useful as well. 

Overall, five feature visualizations are slow to make and not very informative. There are lots of information and the same time, but specially no point about the figure can be made. 

This is the first actionable result I got. I can see how I could improve the visualization in future steps:
- Rearange features to actually make a point
- Use the continous max speed variable. The information needed to define the buckets is excesive, and it ocludes the visualization, when it could be a simple colorbar in the third axis. 
- Label sizes and datapoints should be bigger
- Make the figure memorable

### Log scale or Log transformed?
Next step the book makes a point about, is correct scailing. Frequently, data can have a wide array of value range. For those cases, log scale or log values are often useful. More scale than values though.

The idea is to make an image as similar as posible as one in the book. A feature visualized In one dimensional plot, normal, log transformed, in a log scale. Log scales (Changing the axis so values are exponentially bigger as they move right, rather than only linearly bigger) are more informative as the real value is still shown. 

First of all, simple 1D plots like this one do not exist directly in pyplot. I made this one up by disabling all bounding boxes but bottom one, drag it up, and add some padding between labels and spine/axis. The padding is managed using fig.subplots_adjust(hspace=1), if you dont use it every tittle is evenly spaced and you dont know which figure it belongs to, above or below one. 

An alternative for log scale, is root scale. Its mostly useful when data includes 0 values, since you cant transform them.

The point of the figure is just displaying the comparation between the different transformation and scaled used, alongside showing different acceleration values. One simple conclusion we can get from the figure is the fact that values dont vary wildly, only one outlier. A bike that accelerates really slowly compared to the others. 



### Polar Scale
Using Polar coordinate system for time series visualization was new for me. Really simple way to visualize seasonality in the data.  I want to try my hand at a symple polar coordinate visualization. 

Since I dont have any time-series related series, I will just use gasoline capacity feature, looking for any peaks in the "fake time series" to see if its possible to locate independent outliers this way.

Working on polar visualizations, it surprised me how it has a different way to work on it. Its not a "Special" visualization, you only change the coordinates system, rather than making a specific "Polar" visualization. 

It has everything a plot has, labels, x,y axis (Changed, but they are there!)... The only thing I couldnt manage to do is moving tick labels. It is really prepared to just put them at a certain angle. I'd like to have them paralel to y axis, outside the polar plot, like in the book's figures. 

Related to visualization details, tick label position is defined using set_rlabel_position. Then, to define a polar plot you need a radian list for each of the points x axis, and height values (the ones you actually want to plot). Another thing I learned with many different subplots, is that axis have a legend feature, you can change it from there rather than from figure. 

Final conclusion around the figure, its that is not a great way to visualize non-tabular data. It looks cool and complex, but past that is just visual noise. A simple histogram, or 1D scatterplot is far simpler and makes a point with way less visual noise. So, the point of this figure, is to show to actually NOT use the figure, haha!

The point of polar plots is to show **seasonal** data. For that matter I'd use a line plot like the first one, but make every interval (lap) from a different color, and label it. Wether it be year or monthly data.  


### A powerpoint about color
Color is interesting. It can be used for both continous (graded scale) and discrete data. Aditionally, it can help making the point of the visualization easier. 

For this section I want to make 2 visualizations. One where color shows a continous variable (I already made a discrete color variable in the five feature visualization), and one where it actually helps making a point around a visualization. 

First visualization is a scatterplot of 3 different dimmensions, using color for the third one. 

Second visualization will use color to make a point. The idea is to make a histogram of different values for a certain bike type. But, the point is to make how a certain bike compares to the others, not making a simple display about what the market has to offer. To see an after & before, I will make a simple histogram about a feature using discrete, random colors. Then the same histogram, but using a intense color only for the bike Im interested in comparing.

One important detail I found is the color bar item for the scatterplot, as well as the parameter "edgecolor" for both visualizations. It makes it stand on its own better, more pleasing to see. Also, do not use set_xticklabels method. Works BAD. adjust_subplplots has amazing functionalities for overlapping features and changing overall spaces inside the visualization.  

Final thoughts about these visualizations. The barplot should be sorted, but I will leave that for a future visualization error corrections section in this project. Regarding the bar plot comparison, it surprised me then (first time I read it) and it surprises me now once I implement it. Such a tiny detail changes the point of the visualizations from "Look at all these bikes" to "Look at how does this bike against the others".


# Visulizing different amounts
The idea is to make simple, yet informative plots that display similarly equally ranged features, using a grouped bar plot, stacked bars. The point is to compare them against each other.

For all plots, the idea is to show how many bikes of each type each brand sells. And then compare how do they dliver information.

First thing that called my atention. There is no such thing as a "grouped bar" visualization in matplotlib. you just plot different bars manually. Same with stacked bars. There is a parameter "bottom" where you just move upwards the remaining bars. Same with the heatmap, there is no such thing as a heatmap. You use the imshow function, which takes a matrix as a input. 

At the start of the project, I wondered where could stacked bars be better than grouped bars. By mere chance, I found it: whenever any of the values is 0. Heatmaps, although they are as quick to deliver information, I'd treat them more as a density plot. To see how a big group of variables relate among each other, rather than for showing a small amount of data. 

The conclusion of the visualization is that... I should use grouped bars always for visualizing multiple amounts, but if in any case one variable is 0, then use stacked bars (Whenever the groups are small enough)

# Visualizing a single distribution

There are 4 ways to visualize a distribution in detail: Histogram, Density plots, quantile-quantile plots and cumulative density. The idea is to make a side by side visualization for all these types of visualizations. 

Aditionally, I want to experiment with the shortcommings of each of these visualizations. Histograms and density plots are easy to understand, but they depend on parameters (bin width, and kernel density estimator respectively) that might cause distrtion on the data. On the latter two, they are harder to understand, but they do not depend on any parameters. 

While working on visualizations, I learned a few things. Histograms are as easy to make as they look. Hist named method, give data, and number of bins. Give labels a value and thats it. 

On the other hand, matplotlib does not have native support for density plots though. You can calculate them on your own, or use **seaborn**! This makes it easy to do and has native support to work with matplotlib. 

Next, for a cumulative plot its not as easy as making the acum operation. As the values in between, will be filled with a slope. You need the "Step" function. Once that is clear, the visualization is easy as matplotlib has native support for this visualization.

Quantile Quantile plots do not have native support in matplotlib. Seaborn offers a similar function, but looks more like a regular scatterplot. Its better to use scipy for that matter. With the math covered, its easier to make manually the plot. 

Finally, the histogram parameter comaprison. The number of bins shown. As the number increases, the bins are progresively thinner to a point where they look sparce. Too low and it gives the impression there is data where there truly isnt. Too little bins and the sparcity prevents from obtaining useful information. Aditionally, There is also a kernel estimator parameter for density plots. However, gaussian plot is so commonly used many libraries (Like seaborn) does not offer a kernel alternative. I could implement it on my own, but there is truly no necessity.
Regarding histogram parameters, in this case I'd use 19 bins. It allows for the balance previously mentioned. 


# Visualizing multiple distributions

There are multiple ways to visualize and compare distributions that relate two variable values:
- Boxplots
- Violin plots
- Strip charts
- Sina plots (Combination of strip and sina plots. You can find a variation where the jittering depends on density. So points in higher density points have increasingly more jitter than outliers, imitating the violin plot outline)
- Stacked histograms (Bad for comparation. Not recomended its use, and wont be implemented in)
- Overlapping densities
- Comparing density plots (Plot each individual distribution, in front of overall distribution that resumes both variables)
- Age pyramids
- Ridgeline plots

The idea is to make a plot with all of them to train in data visualization, and comparing results.

The distributions will be about how consumption changes related to the type of bike. Except for the age pyramid. To make it more visually appealing and use more data, analyse price depending on weight value. 

Doing these visualizations I learned a lot. I want to divide them individually:

The boxplot was surprisingly easy to make. Its directly implemented in matplotlib, and its highly customizable. Only worry about label rotation.

Similar to boxplot, violinplot has direct implementation and thus it was easy to use. It has lots of parameters to customize its appearence. 

Stripcharts kinda surprised me, since they dont have direct support. To create them, you use a scatter plot containing the x values. To add jitter, just add a tiny random value to the x axis. 

Sina plots are truly a mix of strip charts and violinplots. However, I read there is a variation where you can use density-based jittering, where points have more jittering depending on how dense the function is at that point value. To implement that, only estimate density using gausian kde, and obtain the densities at the point. Using a minmax scaled density, a uniform distribution between -1 and 1, and a jitter strength value you have it.

For the density plot comparison the most difficult thing is scaling each density now. Since all distribution areas must be equal to one, if you plot the overall distribution with the subset, they will have similar sizes. You have to scale down the densities of the subset, by multiplying their real densities by the percentage they represent in the whole distribution. 

Overlapping density plot only works well for smaller subsets of distributions, and shine only when the distributions are very different from each other.

Age pyramids are made by plotting horizontal bar plots. Then, only make sure to change all label labels to their absolute value. 

Ridgeplots are curious. I have not found a way to make them using matplotlib or seaborn. The closest thing is a axis on top of each other trying to simulate it, bu it doesnt do the trick. 
I found a library called "joypy" (joyplot is alias for ridgeline plot) that implements it directly. It uses pyplot as its backend, but poorly. It has its problems, but exists as the best option currently. 
Ridgeline plots are **speacially** useful to show trends over time. Since none of my data 

An interesting detail I want to share before I forget. Custom shared. It is possible to share x and y axis by row or column, but you can share in a customized way pretty easily:
# Manually share the y-axis
ax1.sharey(ax2, ax3)

Also, to make a custom plot (occupies multiples subplots) You  gotta define a figure and a gridspec independently, and then create each of the axis:
        def create_custom_subplot():
            # Create a grid with 2 rows and 4 columns
            fig = plt.figure(figsize=(8, 6))
            gs = fig.add_gridspec(2, 4)

            axes = []

            for i in range(2):
                for j in range(4):
                    # First subplot (occupies the top left cell)
                    axes.append(fig.add_subplot(gs[0, 0:2]))

I learned this while trying to make the density plot comparison in one single axes. This was not the case for this to be useful, but it could be nice in the future to have that tool at hand.


# Visualizing proportions
Similar to visualizaing amounts, but time all adds to a whole. As such, the visualizations frequently used are similar, only including piecharts. This part will compare all proportion visualizations:
- Piechart
- Vertical bar plot
- Horizontal barplot
- Stacked bar plot

I will make the same division as before: bike types in the market.

The piechart is easy to do, as it has a direct implementation. Only thing to take into account is the fact that it wants to show quantity values in perecntages. that can be useful sometimes, but real values sometimes might be better to enhance the dimmension of the subset.

Vertical and horizontal bar plots are simple to do as well. Vertical bar plots need rotation on its horizontal labels to avoid them to colision into themselves. Horizontal bar plots do not suffer from this problem, but too long labels might be too long and distracting, providing huge margins.

A single stacked bar have one more problem in adition to the ones seen previously. Pyplot will adjust the plot so it ocupies everything widely speacking. So Its necessary to set bar width, and plot axis limits for the bar to not look too big, or not leave enough space for the legend. 


# Visualizing proportions with multiple grouping variables

Sometimes grouping can be quite more complex than just one variable. We might be interested to divide bikes not only by types, but acceleration and weight as well. I will turn weight to a discrete variable (light < 150, heavy >= 150 ), and acceleration (slow >  25, fast <= 25)

There are three main ways to visualize this type of data:
- tree map plots
- mosaic plots
- parallel sets

There is another option, density plots comparison. We used them previously. Regular ones only allow for two grouping variables. But I want to explore the posibility of using 3 using a matrix. I will use a different plot for each case. 

- Density plot comparison (Only for 2 levels of anidation, or more using a matrix. But)
