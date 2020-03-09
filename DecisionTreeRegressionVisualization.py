# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:27:31 2020

@author: Santosh Sah
"""

import matplotlib.pyplot as plt
import numpy as np
from DecisionTreeRegressionUtils import (readDecisionTreeRegressionModel, readIndepentDataset, readDependentDataset)

"""
Visualising the DecisionTree Regression results (for higher resolution and smoother curve)

"""
def visualisingDecisionTreeRegressionInHighResolution():
    
    X = readIndepentDataset()
    y = readDependentDataset()
    decisionTreeRegressionModel = readDecisionTreeRegressionModel()
    
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))    

    # Visualising the Polynomial Regression results
    plt.scatter(X, y, color = 'red')
    plt.plot(X_grid, decisionTreeRegressionModel.predict(X_grid), color = 'blue')
    plt.title('Truth or Bluff (Polynomial Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    
    plt.savefig("decision_tree__trainingsetresult_high_resolution.png")
    
    plt.show()
    
if __name__ == "__main__":
    visualisingDecisionTreeRegressionInHighResolution()