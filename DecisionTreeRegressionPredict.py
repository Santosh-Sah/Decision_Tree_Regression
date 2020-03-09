# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:24:42 2020

@author: Santosh Sah
"""

import pandas as pd
from DecisionTreeRegressionUtils import readDecisionTreeRegressionModel

def predictDecisionTreeRegression():
    
    decisionTreeRegressionModel = readDecisionTreeRegressionModel()
    
    inputValue = [6.5]
    inputValueDataframe = pd.DataFrame(inputValue)
    
    predictedValue = decisionTreeRegressionModel.predict(inputValueDataframe.values)
    
    print(predictedValue)

if __name__ == "__main__":
    predictDecisionTreeRegression()

