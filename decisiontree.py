import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree

# Load the dataset
df = pd.read_csv("path/to/dataset.csv")

# Exploratory Data Analysis (EDA)
print("Dataset shape:", df.shape)
print("Columns:", df.columns)
print("Data types:\n", df.dtypes)
print("Head:\n", df.head())
print("Describe:\n", df.describe())

# Preprocessing
# Drop any unnecessary columns
df.drop(columns=["column1", "column2"], inplace=True)

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Handle missing values
df.fillna(value=df.mean(), inplace=True)

# Separate the target variable from the features
X = df.drop(columns=["target_variable"])
y = df["target_variable"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training set
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=["class1", "class2"])
plt.show()
