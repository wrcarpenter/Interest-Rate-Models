"""
Interest Rate Model - Calibrating Theta
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

#%%
# Read in zero coupon data 
zero_coupons = pd.read_csv('https://raw.githubusercontent.com/wrcarpenter/Interest-Rate-Models/main/Data/zcbs.csv')
zcbs = zero_coupons.loc[zero_coupons['Date']=='3/8/2024']
zcbs = zcbs.drop("Date", axis=1)

#%%
# Build a tree
zeros     = np.array(zcbs.iloc[:,0:24])
calibrate = model.build(zeros, 0.007, 1/12)
tree      = model.rateTree(calibrate[0], calibrate[2], 0.007, 1/12)

arr  = np.zeros([zeros.shape[1], 3])

for i in range(0,len(arr)):
    
    # month
    arr[i,0] = i+1
    # Zero coupon price
    arr[i,1] = zeros[0,i]
    # use model
    rate_tree = tree[:i+1, :i+1]
    cash_flow = model.cf_bond(rate_tree, 0.00, 1/12, 1, 0.00) 
    result    = model.priceTree(rate_tree, 1/2, cash_flow, 1/12, "bond", 1)
    # Model zcb price    
    arr[i, 2] = result[0]


prices = pd.DataFrame(arr, columns=["Period", "ZCB Price", "Model Price"])
chrt   = chart_zcb_calibration(prices, 10, 5, "Zero Coupon Bond (ZCB) Calibration - 24 Period Tree")

#%%
# Charting example       
def chart_zcb_calibration(arr, w, l, title):
    
    x1 = np.array(arr['Period'])
    y1 = np.array(arr['ZCB Price'])
    y2 = np.array(arr['Model Price'])
    
    fig,ax = plt.subplots(figsize=(w,l))
    ax.set_xticks(np.arange(1, len(arr)+15, 5))
    ax.set_yticks(np.arange(0,1,0.01))
    
    
    ax.set_title(title, fontsize="large")
    ax.set_ylabel('Price', fontsize="large")
    ax.set_xlabel('Months', fontsize="large")
    
   
    plt.scatter(x1, y1, color='blue', marker='x', label="Market ZCB Price")
    plt.plot(x1, y2, color='green',label='Model ZCB Price')
    
    plt.legend(loc='upper right', fontsize='large')


