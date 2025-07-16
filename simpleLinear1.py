import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print(y_pred)

# plt.scatter(X_train, y_train, color='red')
# plt.plot(X_train, regressor.predict(X_train), color='blue')
# plt.title('Salary vs Experience (Training set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()

# plt.scatter(X_test, y_test, color='green')
# plt.plot(X_train, regressor.predict(X_train), color='yellow')
# plt.title('Salary vs Experience (Test set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()

print('For 5 years of experience:')
salary_pred = regressor.predict([[5]])[0]  # Predicting salary for 5 years of experience
formatted_salary = f"₱{salary_pred:,.2f}"
print(formatted_salary)
print('For a salary of ₱40,000:')
years_exp = (40000 - regressor.intercept_) / regressor.coef_[0]
print(f"Estimated years of experience: {years_exp:.2f}")
