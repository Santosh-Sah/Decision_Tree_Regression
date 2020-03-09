# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:26:05 2020

@author: Santosh Sah
"""
from sklearn.tree import DecisionTreeRegressor
from DecisionTreeRegressionUtils import (saveDecisionTreeRegressionModel, readIndepentDataset, readDependentDataset)

"""
Train DecisionTree regression model 
"""
def trainDecisionTreeRegressionModel():
    
    X = readIndepentDataset()
    y = readDependentDataset()
    
    # Fitting DecisionTree Regression to the dataset
    decisionTreeRegressor = DecisionTreeRegressor(random_state = 0)
    decisionTreeRegressor.fit(X, y)
    
    saveDecisionTreeRegressionModel(decisionTreeRegressor)

if __name__ == "__main__":
    trainDecisionTreeRegressionModel()
