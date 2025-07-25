# Importing The Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Importing the Dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  # Independent Variables
y = dataset.iloc[:, -1].values  # Dependent Variable

# When data not in order
# X = dataset.drop(columns='Purchased').values
# y = dataset['Purchased'].values

# Identifying how many missing data per column
# missing_data = dataset.isnull().sum()
# print("Missing data in each column:")
# print(missing_data)


# Taking care of missing data
# Find the missing values in the dataset and use mean strategy
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# Show where the missing values are be sure to impute only the columns with numeric data
imputer.fit(X[:, 1:3])
# Finally, transform the data where the missing values are located
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data

# Encoding the Independent Variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
# Encoding the Dependent Variable
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)
print("X_train:", X_train)
print("X_test:", X_test)
print("y_train:", y_train)
print("y_test:", y_test)

# Feature Scaling
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print("Scaled X_train:", X_train)
print("Scaled X_test:", X_test)