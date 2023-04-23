import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Load the dataset
data = pd.read_csv('example_dataset.csv')

# Split the dataset into input and output variables
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create individual models
model1 = LogisticRegression()
model2 = DecisionTreeClassifier()
model3 = SVC(kernel='linear', probability=True)

# Create an ensemble voting classifier
model = VotingClassifier(estimators=[('lr', model1), ('dt', model2), ('svm', model3)], voting='soft')

# Fit the model to the training data
model.fit(x_train, y_train)

# Make predictions using the model
y_pred = model.predict(x_test)
y_pred_prob = model.predict_proba(x_test)[:, 1]

# Evaluate the model
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred))
print('F1 Score: ', f1_score(y_test, y_pred))
print('ROC AUC Score: ', roc_auc_score(y_test, y_pred_prob))
print('Confusion Matrix: ')
print(confusion_matrix(y_test, y_pred))
