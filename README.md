# Classification
Single-feature classification algorithm using gradient descent. 

Using python libraries: Matplotlib, csv, random, and math. 

Initalised to use testing data. Datapoints are plotted in blue, with the orange line being the predictions of the algorithm for the given x-value. The cost (mean squared error between predictions and given data) through interations is also graphed to track the improvement of the algorithm (the lower the better). 

## Adjustable Parameters
- `order`: user may set the order of the polynomial used to fit the datapoints
- `batch`: the number of datapoints fit by the algorithm per iteration (setting it to `size` works well for small datasets. Consider smaller values for large datasets)
- `lrate`: learning rate. Larger values mean each iteration affects the hypothesis function to a greater extent. Values too small can result in slow convergence, values too large can overshoot the optimal hypothesis.
- `iters`: the number of interations 
- `detail`: used for plotting. Determines the number of sample points of which the orange line is constructed. More points result in a smoother curve.

## Acknowlegements

DATA.csv is a section of [this](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) breast cancer dataset. 

Thanks to Andrew Ng's [course](https://www.coursera.org/learn/machine-learning/home/welcome) on Coursera, where I learnt these techniques
