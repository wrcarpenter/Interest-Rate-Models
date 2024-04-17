# -*- coding: utf-8 -*-
"""
Testing
"""
import sys
import os
import math 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
# Custom module
import rate_model_engine as model

zero_coupons = pd.read_csv('https://raw.githubusercontent.com/wrcarpenter/Interest-Rate-Models/main/Data/zcbs.csv')

zcbs = zero_coupons.loc[zero_coupons['Date']=='3/8/2024']
zcbs = zcbs.drop("Date", axis=1)
# small example for calibration
zeros  = np.array(zcbs.iloc[:,0:60])
cal    = model.build(zeros, 0.009, 1/12)
tree   = model.rateTree(x[0], x[2], 0.01, 1/12, 'HL')
cf     = model.cf_bond(tr, 5.00, 1/12, 1, 0.00)
px     = model.priceTree(tr, 1/2, cf, 1/12, bond, 1) 

#%%


# Reading in interest rate tree data for testing 
one_period_tree = np.array([[0.0168, 0.0433, np.nan], 
                            [np.nan, 0.0120, np.nan], 
                            [np.nan, np.nan, np.nan]]) 

two_period_tree = np.array([[0.0168, 0.0433, 0.0638, np.nan], 
                            [np.nan, 0.0120, 0.0361, np.nan], 
                            [np.nan, np.nan, 0.0083, np.nan], 
                            [np.nan, np.nan, np.nan, np.nan]])

largeTree = pd.read_csv("https://raw.githubusercontent.com/wrcarpenter/Fixed-Income-Valuation/main/Data/testTree.csv", header=0).values

# Testing a cash flow generation
# 1/2 delta is semi annual 
flr = cf_floor(two_period_tree, 1.00, 1/2, 100, 5.00)
cp  = cf_cap(two_period_tree, 1.00, 1/2, 100, 5.00)
bd  = cf_bond(two_period_tree, 1.00, 1/2, 100, 5.00)

#%%
    
# Test cases
# Slide 5 
# prob  = probTree(4)
# pu    = prob[1,1]
# cf    = cfTree(one_period_tree, 5.00, delta, 1, ) 
# delta = 1/2

# price = priceTree(one_period_tree, prob, cf, delta, bond, 1)

# # Slide 7
# prob  = probTree(5) 
# delta = 1/2

# price = priceTree(two_period_tree, prob, delta, cf, bond, 1)

# prob  = probTree(len(rateTree))
# cf = cfTree(rateTree, 0, 1/2, 100, 0.02, bond_cf)
# delta = 1/2

# price = priceTree(rateTree, prob, cf, delta, bond, 100)

