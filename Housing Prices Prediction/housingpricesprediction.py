# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 12:33:08 2018

@author: iGuest
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.head()
#taken from Alexandre Papiu's kernel
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="mean_squared_error", cv = 10))
    return(rmse)


totaldata = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

#taken from Alexandre Papiu's kernel
#used since most of the variables are skewed
train["SalePrice"] = np.log1p(train["SalePrice"])
numerical_f = all_data.dtypes[all_data.dtypes != "object"].index

skewed_f = train[numerical_f].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_f = skewed_f[skewed_feats > 0.75]
skewed_f = skewed_f.index

totaldata[skewed_f] = np.log1p(totaldata[skewed_f])

totaldata = pd.get_dummies(totaldata)

totaldata = all_data.fillna(totaldata.median())

X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score

model_lasso = LinearRegression().fit(X_train, y)

print(rmse_cv(model_lasso).mean())
#print(rmse_cv(modelLR).mean())

lasso_preds = np.expm1(model_lasso.predict(X_test))
lasso_preds = model_lasso.predict(X_test)
solution = pd.DataFrame({"id":test.Id, "SalePrice":lasso_preds})
solution.to_csv("solutions.csv", index = False)
