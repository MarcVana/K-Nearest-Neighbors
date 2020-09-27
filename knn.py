"""
Created on Sun Sep 27 13:07:00 2020

KNN - K NEAREST NEIGHBORS CLASSIFICATION PROJECT

@author: Marc
"""
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading the dataset
df = pd.read_csv('KNN_Project_Data.csv')

# Scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis = 1))
df_scaled = scaler.transform(df.drop('TARGET CLASS', axis = 1))

# Splitting the data into Train and Test
from sklearn.model_selection import train_test_split
X = df_scaled
Y = df['TARGET CLASS']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 101)

# Chossing the best K Value from 1 to 100 (building a KNN for each K)
from sklearn.neighbors import KNeighborsClassifier
error_rate = []
minimum = 1.0
best_k = 1
for k in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, Y_train)
    pred_k = knn.predict(X_test)
    error_rate.append(np.mean(pred_k != Y_test))
    error = np.mean(pred_k != Y_test)
    if error < minimum:
        minimum = error
        best_k = k

# Plotting the error rates
print('> BEST K VALUE = ', best_k)
plt.figure(figsize = (10, 6))
plt.plot(range(1, 100), error_rate, color = 'blue', linestyle = '--', markersize = 5, marker = 'o', markerfacecolor = 'red')
plt.title('Error rate VS K Value')
plt.xlabel('K Value')
plt.ylabel('Error rate')
plt.savefig('Plots/k_values_error_rate.png')

# Building the final KNN
final_knn = KNeighborsClassifier(n_neighbors = best_k)
final_knn.fit(X_train, Y_train)
pred = final_knn.predict(X_test)

# Printing final metrics
from sklearn.metrics import confusion_matrix, classification_report
print('Confusion matrix')
print(confusion_matrix(Y_test, pred))
print('-------------------------------------------------------')
print('Classification report')
print(classification_report(Y_test, pred))