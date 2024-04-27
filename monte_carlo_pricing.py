# -*- coding: utf-8 -*-
"""
Monte Carlo - Rate Models

"""
import sys
import os
import math 
import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy.optimize import newton
# Custom module
import model_ho_lee as model

# Monte carlo simluation 
def tree_monte_carlo(tree, paths):    

    monte = np.zeros([len(tree)-1, paths])

    monte[0,:] = tree[0,0] # assign initial interest rate 
    
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

# Charting example       
def chart_monte_carlo(monte, spots, w, l, title):
    
    x1 = np.array(monte['Period'])
    s1 = np.array(spots[0,:]/100)
    fig,ax = plt.subplots(figsize=(w,l))
    ax.set_xticks(np.arange(1, len(monte)+5, 20))
    ax.set_yticks(np.arange(round(np.min(monte.iloc[:,1:].values)-0.50,2), round(np.max(monte.iloc[:,1:].values)+.50,2), 0.02))
    ax.set_title(title, fontsize="large")
    
    
    ax.set_ylabel('Interest Rate (%)', fontsize="large")
    ax.set_xlabel('Months', fontsize="large")
    
    for col in range(1, monte.shape[1]):
        
        y = np.array(monte.iloc[:, col])
        if col==1:
            plt.plot(x1, y, linewidth=0.9, label="Simulations")
        plt.plot(x1, y, linewidth=0.9)    
    
    # plt.plot(x1, s1, color="blue", label="Spot Rates")
    # plt.legend(loc='upper right', fontsize='large')

# Charting example       
def chart_mc(monte, spots, w, l, title):
    
    x1 = np.array(monte['Period'])
    s1 = np.array(spots[0,:]/100)
    s1 = s1[:len(s1)-1]
    fig,ax = plt.subplots(figsize=(w,l))
    ax.set_xticks(np.arange(1, len(monte)+5, 20))
    ax.set_yticks(np.arange(round(np.min(monte.iloc[:,1:].values)-0.50,2), round(np.max(monte.iloc[:,1:].values)+.50,2), 0.02))
    ax.set_title(title, fontsize="large")
    
    ax.set_ylabel('Interest Rate (%)', fontsize="large")
    ax.set_xlabel('Months', fontsize="large")
    
    for col in range(1, monte.shape[1]):
        
        y = np.array(monte.iloc[:, col])
        if col==1:
            plt.plot(x1, y, linewidth=0.9, label="Simulations", color="grey")
        plt.plot(x1, y, linewidth=0.9, color="grey")    
    
    plt.plot(x1, s1, color="blue", label="Spot Rates")
    plt.legend(loc='upper right', fontsize='large')

           
if __name__ == "__main__":

    # Read in zero coupon data 
    zero_coupons = pd.read_csv('https://raw.githubusercontent.com/wrcarpenter/Interest-Rate-Models/main/Data/zcbs.csv')
    spot_rates   = pd.read_csv("https://raw.githubusercontent.com/wrcarpenter/Interest-Rate-Models/main/Data/spots-monthly.csv", header=0)
    zcbs  = zero_coupons.loc[zero_coupons['Date']=='3/8/2024']
    spots = spot_rates.loc[spot_rates['Date']=='3/8/2024']
    zcbs = zcbs.drop("Date", axis=1)
    spots = spots.drop("Date", axis=1)
    
    zeros  = np.array(zcbs.iloc[:,0:120])
    spots  = np.array(spots.iloc[:,0:120])
    cal    = model.build(zeros, 0.012, 1/12)
    tree   = model.rateTree(cal[0], cal[2], 0.012, 1/12)
    cf     = model.cf_bond(tree, 5.00, 1/12, 1, 0.00)
    out    = model.priceTree(tree, 1/2, cf, 1/12, "bond", 1)
    px     = out[0]
    ptree  = out[1]

    monte  = tree_monte_carlo(tree, 500)
    # chart_monte_carlo(monte, spots, 12,5.0, "Ho-Lee Binomial Tree Monte Carlo: 1,000 Simulations")
    chart_mc(monte, spots, 12, 5.0, "Ho-Lee Binomial Tree Monte Carlo: 500 Simulations")
    

     