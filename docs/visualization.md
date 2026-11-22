# Visualization

## plot_mutable

*`def torchdish.plot_mutable(dish: torchdish.Dish, input: any, framerate: number=100) -> matplotlib.animation.FuncAnimation`*

Produces an interactive animation of the mutable state created by `input` up until step `dish.steps + dish.noise`, with `framerate` determined in milliseconds. 

Colouring: 
- Colours are normalized on all channels within each step. 
- If the mutable state has 1 channels, the animation will display in greyscale. 
- If the mutable state has 3 channels, the animation will use RGB. 
- For any other number of channels, the top 3 principal components of the entire run are used to display RGB. 

Important: the return value of this function should be stored in order to prevent automatic garbage collection affecting visualization results. 

## plot_mutable

*`def torchdish.plot_immutable(dish: torchdish.Dish) -> matplotlib.animation.FuncAnimation`*

Plots the immutable state of `dish` in RGB. Colours are treated the same as in `plot_mutable`. 