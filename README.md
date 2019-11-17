# MachineLearning
A collection of various algorithms under ML coded in Python.

## Python Libraries:
* Scikit-learn [(sklearn)](https://en.wikipedia.org/wiki/Scikit-learn) is a free software machine learning library for the Python programming language.

1. numPy: NumPy is the fundamental package(library) needed for scientific computing with Python.It provides:

* a powerful N-dimensional array object
* sophisticated (broadcasting) functions
* tools for integrating C/C++ and Fortran code
* useful linear algebra, Fourier transform, and random number capabilities

source: https://github.com/numpy/numpy

2. matplotlib.pyplot: It is a state-based interface to matplotlib. It provides a MATLAB-like way of plotting.
pyplot is mainly intended for interactive plots and simple cases of programmatic plot generation. 
Matplotlib is a python library inspired by MATLAB whereas pyplot is a shell-like interface to matplotlib, to make it easier to use for people used to MATLAB.

source: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.html

3. pandas: Pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with "relational" or "labeled" data both easy and intuitive. It aims to be the fundamental high-level building block for doing practical, real world data analysis in Python. Additionally, it has the broader goal of becoming the most powerful and flexible open source data analysis/manipulation tool available in any language.

source: https://github.com/pandas-dev/pandas

## Data Files:
* .csv files: A comma-separated values file is a delimited text file that uses a comma to separate values. A CSV file stores tabular data in plain text. Each line of the file is a data record. Each record consists of one or more fields, separated by commas.

## Classes
* LabelEncoder: It is used to convert categorical text data into model-understandable numerical data.

* OneHotEncoder: Used to solve the problem of categorical data converting into hierarchial data that label Encoding creates. It takes a column which has categorical data, which has been label encoded, and then splits the column into multiple columns.
[[labelencoder vs onehotencoder]](https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621)

## Functions
* Fit(): calculates the value of parameters.
* Transform(): Fits data according to the parameter values.
* Fit_transform(): Calculates paramteres according to data and then transforms it.

## Visualizing Results
* scatter(): Matplot has a built-in function to create scatterplots called scatter(). A scatter plot is a type of plot that shows the data as a collection of points.<br>
      eg- plt.scatter(x_coordinates, y_coordinates, color = 'any_color', alpha = {0-1..opacity})

* plot(): Plots y versus x as lines and/or markers.<br>
      eg- plt.plot(x_coordinates, y_coordinates, color = 'any_color')

* show():  Function displays the current figure that you are working on. [[show() vs draw()]](https://stackoverflow.com/questions/23141452/difference-between-plt-draw-and-plt-show-in-matplotlib)<br>
      eg- plt.show()

* draw(): This will re-draw the figure. This allows you to work in interactive mode and, should you have changed your data or formatting, allow the graph itself to change.<br>
      eg- plt.draw();




