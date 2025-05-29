import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

iris_path ='C:\\Codes\\Machine Learning\\Assets\\iris.csv'
iris = pd.read_csv(iris_path, header=None, names=columns)

unique_species = iris['species'].unique()
print("Unique species:\n", unique_species)

count_species = iris['species'].value_counts()
print("Count of each species:\n", count_species)