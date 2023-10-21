import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from numpy import random

wine = pd.read_csv('winequality-white.csv', sep=';', header = 'infer')
y = wine.iloc[:,11]
x = wine.drop(wine.columns[11], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#print(len(y))

#idea for evaluating accuracy this way comes from Michael Elgin and his code at https://github.com/COSC5557/ml-algorithm-selection-Mikey-E/blob/main/Alg_Selection.ipynb
def accuracy(y_pred:np.ndarray, y_test:np.ndarray) -> float:
    correct = (np.round(y_pred) == y_test).astype(int)
    correct = correct.sum()
    return (correct/len(guess)) * 100

guess = random.randint(4, 9, size=(len(x_test))) #randomly guessing 0-10 yields terrible results compared to guessing 4-9 or even 3-9 (min to max)
acc = accuracy(guess, y_test)
##########################################################################
#correct = (np.round(guess) == y_test).astype(int)
#correct = correct.sum()
#acc = (correct/len(guess)) * 100
##########################################################################
print("Percent accuracy for Randomly Guessing:{0:.2f}%\n".format(acc))

#after linear regression i pulled some different algorithms off of https://scikit-learn.org/stable/supervised_learning.html to use here and the ones recommended in github
#some are left out like the guassian regressor since it seemed like it needs more tinkering or maybe cant be easily applied to this problem
############################################################################################################################################################################
#linear regression
############################################################################################################################################################################
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
acc = accuracy(y_pred, y_test)

#print("Mean Squared Error for Linear Regression:", mean_squared_error(y_test, y_pred)) #between 0.35 and 0.52
print("\nCoefficient of determination for Linear Regression:", r2_score(y_test, y_pred)) #between 0.28 and 0.48
print("Percent accuracy for Linear Regression: {0:.2f}%".format(acc))
from sklearn.metrics import d2_absolute_error_score
print("Absolute error score for Linear Regression:", d2_absolute_error_score(y_test, y_pred))

############################################################################################################################################################################
#random forest
############################################################################################################################################################################
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
acc = accuracy(y_pred, y_test)

#print("\nMean Squared Error for Random Forest:", mean_squared_error(y_test, y_pred))
print("\nCoefficient of determination for Random Forest:", r2_score(y_test, y_pred))
print("Percent accuracy for Random Forest: {0:.2f}%".format(acc))
print("Absolute error score for Random Forest:", d2_absolute_error_score(y_test, y_pred))

############################################################################################################################################################################
#decision tree
############################################################################################################################################################################
#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor
#why 
from sklearn.tree import DecisionTreeRegressor
dtm = DecisionTreeRegressor()
dtm.fit(x_train, y_train)
y_pred = dtm.predict(x_test)
acc = accuracy(y_pred, y_test)

#print("\nMean Squared Error for Decision Tree:", mean_squared_error(y_test, y_pred))
print("\nCoefficient of determination for Decision Tree:", r2_score(y_test, y_pred))
print("Percent accuracy for Decision Tree: {0:.2f}%".format(acc))
print("Absolute error score for Decision Tree:", d2_absolute_error_score(y_test, y_pred))

############################################################################################################################################################################
#nearest neighbor regression
############################################################################################################################################################################
#code from https://scikit-learn.org/stable/auto_examples/neighbors/plot_regression.html#sphx-glr-auto-examples-neighbors-plot-regression-py
#info from https://scikit-learn.org/stable/modules/neighbors.html
from sklearn import neighbors
#quality is between 0-10 but as tested with the random guesses, 4-9 seems to make the result better
#this is one ill definitely want to look at for hyperparemeter optimization
knn = neighbors.KNeighborsRegressor() #6 neighbors for better results
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc = accuracy(y_pred, y_test)

#print("\nMean Squared Error for Nearest Neighbor Regression:", mean_squared_error(y_test, y_pred))
print("\nCoefficient of determination for Nearest Neighbor Regression:", r2_score(y_test, y_pred))
print("Percent accuracy for Nearest Neighbor Regression: {0:.2f}%".format(acc))
print("Absolute error score for Nearest Neighbor Regression:", d2_absolute_error_score(y_test, y_pred))

############################################################################################################################################################################
#support vector machine applied to regression
############################################################################################################################################################################
#code/info from https://scikit-learn.org/stable/modules/svm.html#regression
from sklearn import svm
supportvm = svm.SVC()
supportvm.fit(x_train, y_train)
y_pred = supportvm.predict(x_test)
acc = accuracy(y_pred, y_test)

#print("\nMean Squared Error for Support Vector Machine:", mean_squared_error(y_test, y_pred))
print("\nCoefficient of determination for Support Vector Machine:", r2_score(y_test, y_pred))
print("Percent accuracy for Support Vector Machine: {0:.2f}%".format(acc))
print("Absolute error score for Support Vector Machine:", d2_absolute_error_score(y_test, y_pred))

############################################################################################################################################################################
#gradient boosting regression
############################################################################################################################################################################
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train)
y_pred = gbr.predict(x_test)
acc = accuracy(y_pred, y_test)

#print("\nMean Squared Error for Gradient Boosting Regression:", mean_squared_error(y_test, y_pred))
print("\nCoefficient of determination for Gradient Boosting Regression:", r2_score(y_test, y_pred))
print("Percent accuracy for Gradient Boosting Regression: {0:.2f}%".format(acc))
print("Absolute error score for Gradient Boosting Regression:", d2_absolute_error_score(y_test, y_pred))