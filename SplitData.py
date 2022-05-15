# -*- coding: utf-8 -*-
#%%
import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import *
from math import sqrt
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import time

df = pd.read_excel('LJ-TahminModel-2018.xlsx' )
 
print("Column headings:")
print(df.columns)

X  = df[["V1","V2","Vloss","Age"]].values
y  = df["Distance"].values

scaler = StandardScaler().fit(X)
X = scaler.transform(X)
#%%
fold = 1
kf = KFold(n_splits=5, random_state=1, shuffle=True )
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    dfTrain = pd.DataFrame({
        "V1":X_train[:,0],
        "V2":X_train[:,1],
        "Vloss":X_train[:,2],
        "Age":X_train[:,3],
        "Distance":y_train        
        })
    
    dfTest = pd.DataFrame({
        "V1":X_test[:,0],
        "V2":X_test[:,1],
        "Vloss":X_test[:,2],
        "Age":X_test[:,3],
        "Distance":y_test        
        })
    
    break
    dfTrain.to_csv(f"Train_{fold}.csv",index=False)
    dfTest.to_csv(f"Test_{fold}.csv",index=False)
    fold += 1