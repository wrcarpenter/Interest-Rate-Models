"""
Bond Pricing
Source Code

Author : William Carpenter
Date   : April 2024

Objective: Use a binomial tree rate model to price swaps.

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

#%%
zeros = np.array(zcbs.iloc[:,0:120])
x     = model.build(zeros, sigma, delta)
tree  = model.rateTree(x[0], x[2], sigma, delta)

#%%
# Varying rate strike for cap pricing 
# Create cap pricing array 

notion = 100
sigma  = 0.009
strike = 0.00
delta  = 1/12
coupon = 1.00
prob   = 1/2

pricing = np.zeros([20,5])
coupon  = 1.00 

for row in range(0, pricing.shape[0]):
    
    pricing[row, 0] = notion
    pricing[row, 1] = coupon
    pricing[row, 2] = zeros.shape[1] # periods
    pricing[row, 3] = sigma
    
    cashflow = model.cf_bond(tree, strike, delta, notion, coupon)
    output   = model.priceTree(tree, prob, cashflow, delta, "bond", notion)
    
    px = output[0]
    
    pricing[row, 4] = px
    coupon = coupon + 0.25

pricing = pd.DataFrame(pricing, columns=["Notional", "Coupon", "Periods", "Volatility", "Price"])


#%%
# Chart the given array 

def chart_bond_pricing(arr, w, l, title):
    
    x1 = np.array(arr['Coupon'])
    y1 = np.array(arr['Price'])
    
    plt.ylim(0, 800000)
    fig,ax = plt.subplots(figsize=(w,l))
    ax.set_xticks(np.arange(0, len(arr)+1, 0.5))
    ax.set_yticks(np.arange(70,120, 2))
    
    ax.set_title(title, fontsize="large")
    ax.set_ylabel('Price', fontsize="large")
    ax.set_xlabel('Coupon', fontsize="large")
    
    plt.plot(x1, y1, color='blue', label="Bond Price")
    
    plt.legend(loc='upper right', fontsize='large')

chart_bond_pricing(pricing, 10, 5, "Bond Prices with Ho-Lee Model: Vary by Coupon Rate (%)") 



