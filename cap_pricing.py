"""
Cap Pricing

Source Code

Author : William Carpenter
Date   : April 2024

Objective: Leverage a binomial tree rate model to price caps.

"""
import sys
import os
import math 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Custom module
import model_ho_lee as model

#%%
    
zero_coupons = pd.read_csv('https://raw.githubusercontent.com/wrcarpenter/Interest-Rate-Models/main/Data/zcbs.csv')
zcbs  = zero_coupons.loc[zero_coupons['Date']=='3/8/2024']
zcbs  = zcbs.drop("Date", axis=1)
# small example for calibration

notion = 1000000
sigma  = 0.009
strike = 5.50 
delta  = 1/12
cpn    = 0
prob   = 1/2

zeros = np.array(zcbs.iloc[:,0:60])
x     = model.build(zeros, sigma, delta)
tree  = model.rateTree(x[0], x[2], sigma, delta)
cap   = model.cf_cap(tree, strike, delta, notion, cpn)
p     = model.priceTree(tree, prob, cap, delta, "cap", notion)

px    = p[0]
ptree = p[1]





