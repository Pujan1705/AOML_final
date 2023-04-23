import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load the dataset
data = pd.read_csv('example_dataset.csv')

# Perform EDA
print(data.head()) # Print first 5 rows of data
print(data.describe()) # Print summary statistics of data
print(data.isnull().sum()) # Print number of missing values in each column

# Split the dataset into input variable (X)
X = data.iloc[:, :-1].values

# Create a KMeans model
model = KMeans(n_clusters=3)

# Fit the model to the data
model.fit(X)

# Get the predicted labels for each data point
labels = model.predict(X)

# Evaluate the model using silhouette score and Davies-Bouldin score
silhouette = silhouette_score(X, labels)
db_score = davies_bouldin_score(X, labels)

print('Silhouette Score: ', silhouette)
print('Davies-Bouldin Score: ', db_score)
