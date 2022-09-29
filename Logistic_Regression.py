# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:25:20 2022

@author: 47483
"""

# Logistic Regression

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Customer_Data.csv')

#Matrix of features (Independent Variables) 
X = dataset.iloc[:, [2, 3]].values
#Vector of the dependent variable
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
""" Defining the model/ classifier and Specify a number for random_state
 to ensure same results for each run """
classifier_model = LogisticRegression(random_state = 1)
#fit = Capturing the patterns from the provided data
classifier_model.fit(X_train, y_train)


# Predicting the Test set results
predicted_y = classifier_model.predict(X_test)


# Making the Confusion Matrix for visualization of the performance 
#Determining how accurate the model's predictions are.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_y)
print(cm)
