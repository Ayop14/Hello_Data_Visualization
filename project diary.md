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

