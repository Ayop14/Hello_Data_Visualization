# Project diary
the idea is to write down things that acme to mind while making certain parts of the project. This will make easier returning to the project about how did I acomplished certain requirement.

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


# Visuaizing different amounts
The idea is to make simple, yet informative plots that display similarly equally ranged features, using a grouped bar plot, stacked bars. The point is to compare them against each other.  