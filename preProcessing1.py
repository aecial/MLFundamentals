import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # : Select all columns except the last one
# .values converts the DataFrame to a NumPy array
y = dataset.iloc[:, -1].values # Select the last column as the target variable 

print(dataset)
print(X)
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)