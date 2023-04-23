import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('example_dataset.csv')

# Split the dataset into input and output variables
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a logistic regression model object
model = LogisticRegression()

# Fit the model to the training data
model.fit(x_train, y_train)

# Make predictions using the model
y_pred = model.predict(x_test)

# Print the model accuracy
print('Accuracy: ', accuracy_score(y_test, y_pred))
