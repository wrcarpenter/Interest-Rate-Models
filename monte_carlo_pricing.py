# -*- coding: utf-8 -*-
"""
Monte Carlo - Rate Models

"""

# monte carlo 
# take a rate model and simulate through various paths

import sys
import os
import math 
import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy.optimize import newton
# Custom module
import rate_model_engine as model

#%%
# Read in zero coupon data 
zero_coupons = pd.read_csv('https://raw.githubusercontent.com/wrcarpenter/Interest-Rate-Models/main/Data/zcbs.csv')
zcbs = zero_coupons.loc[zero_coupons['Date']=='3/8/2024']
zcbs = zcbs.drop("Date", axis=1)

#%%
zeros  = np.array(zcbs.iloc[:,0:48])
cal    = model.build(zeros, 0.012, 1/12)
tree   = model.rateTree(cal[0], cal[2], 0.012, 1/12, 'HL')
cf     = model.cf_bond(tree, 5.00, 1/12, 1, 0.00)
out    = model.priceTree(tree, 1/2, cf, 1/12, bond, 1)
px     = out[0]
ptree  = out[1]

#%%

 

def tree_monte_carlo(tree, paths):    
    # return a dataframe
    # same rows as the rate tree, columns is the number of simulations
    monte = np.zeros([len(tree)-1, paths])

    monte[0,:] = tree[0,0] # assign initial interest rate then simulate
    
    # initialize 

    # the first 
    # return monte
    # the array is 48 cells 
    
    for col in range(0,monte.shape[1]):
        
        r = 0
        c = 0
        p = 0
        
        for row in range(1, monte.shape[0]):
        
            p = random.rand()
            
            if p > 0.5:
                # move through rate tree
                monte[row, col] = tree[r, c+1]
                # update position on rate tree
                r = r 
                c = c+1
            else:
                # move through rate tree
                monte[row, col] = tree[r+1, c+1]
                # update position on rate tree
                r = r + 1
                c = c + 1

    
    periods = np.arange(1, len(tree), 1)
    monte   = pd.DataFrame(monte)
    monte.insert(0, 'Period', periods)
    
    return monte

arr = tree_monte_carlo(tree, 15)    


    
def chart_monte_carlo(monte):
    pass
    # chart graph with monte carlo simulations of interest rate tree
     