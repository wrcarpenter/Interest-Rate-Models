"""
Interest Rate Models

Source Code

Author : William Carpenter
Date   : April 2024

Objective: Create a binomial tree interest rate model that takes as arguments
todays forward curve and volatilities. Use the tree to price various bonds and 
other fixed income derivatives (caps, floors, swaps, etc.).

"""
import sys
import os
import math 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton


def payoff(x, typ):
    if typ == "bond":
        return x
    else:
        return 0
        
def cf_floor(rates, strike, delta, notion, cpn):
    
    '''
    Floor Cash Flows
    '''
    cf  = np.zeros([len(rates)+1, len(rates)+1])

    for col in range(0, len(cf)-1):
        for row in range(0, col+1):
            rate = rates[row,col]
            cf[row, col] = delta*notion*max(strike/100-rate, 0)
            
    return cf 

def cf_cap(rates, strike, delta, notion, cpn):

    cf  = np.zeros([len(rates)+1, len(rates)+1])

    for col in range(0, len(cf)-1):
        for row in range(0, col+1):
            rate = rates[row,col]
            cf[row, col] = delta*notion*max(rate-strike/100, 0)

    return cf

def cf_bond(rates, strike, delta, notion, cpn):
    
    cf  = np.zeros([len(rates)+1, len(rates)+1])

    for col in range(0, len(cf)-1):
        for row in range(0, col+1):
            cf[row, col] = delta*notion*cpn/100  
    
    return cf

def cf_swap(rates, strike, delta, notion, cpn):
    
    cf  = np.zeros([len(rates)+1, len(rates)+1])

    for col in range(0, len(cf)-1):
        for row in range(0, col+1):
            rate = rates[row,col]
            cf[row, col] = delta*notion*(rate-strike/100)    
            
    return cf


def display(arr):
  for i in arr:
    for j in i:
        print("{:8.4f}".format(j), end="  ")
    print() 
  print("\n")


def probTree(length):

    prob = np.zeros((length, length))
    prob[np.triu_indices(length, 0)] = 0.5
    return(prob)

def solver(theta, tree, zcb, i, sigma, delta):    
        
    # Create pricing matrix for ZCBs
    price = np.zeros([i+2, i+2])
    
    # assign the last row to be payoff of ZCB
    price[:,len(price)-1] = 1
    
    # Assign new rates to tree 
    for row in range(0, i+1):
        if row == 0: 
            tree[row, i] = tree[row, i-1] + theta*delta + sigma*math.sqrt(delta)
        else:
            tree[row, i] = tree[row-1, i-1] + theta*delta - sigma*math.sqrt(delta)
    
    # need pricing tree?    
    for col in reversed(range(0, i+1)):
        for row in range(0, col+1):
            node = np.exp(-1*tree[row, col]*(delta))
            price[row, col] = node*(1/2*price[row, col+1] + 1/2*price[row+1, col+1])     
    
    return price[0,0] - zcb    
    
def calibrate(tree, zcb, i, sigma, delta):
    
    t0    = 0.5
    miter = 1000

    theta = newton(solver, t0, args=(tree, zcb, i, sigma, delta))

    for row in range(0, i+1):
        if row == 0: 
            tree[row, i] = tree[row, i-1] + theta*delta + sigma*math.sqrt(delta)
        else:
            tree[row, i] = tree[row-1, i-1] + theta*delta - sigma*math.sqrt(delta)
    
    
    return [theta, tree]
            
def build(zcb, sigma, delta):
    
    # empty rates tree
    tree  = np.zeros([zcb.shape[1], zcb.shape[1]])
    # empty theta tree
    theta = np.zeros([zcb.shape[1]]) 
    
    # Initial Zero Coupon rate
    tree[0,0] = np.log(zcb[0,0])*-1/delta

    r0 = tree[0,0]
    
    for i in range(1, len(theta)):
        
        solved   = calibrate(tree, zcb[0,i], i, sigma, delta)
        
        # update theta array
        theta[i] = solved[0]
        tree     = solved[1]
    
    return [r0, tree, theta]
    

def rateTree(r0, theta, sigma, delta):

    tree = np.zeros([len(theta), len(theta)])
    
    tree[0,0] = r0
       
    for col in range(1, len(tree)):
        
        tree[0, col] = tree[0, col-1] + theta[col]*delta+sigma*math.sqrt(delta)
   
    
    for col in range(1, len(tree)):
        for row in range(1, col+1):
            tree[row, col] = tree[row-1, col] - 2*sigma*math.sqrt(delta)

    return tree
         
                                   
def priceTree(rates, prob, cf, delta, typ, notion):
        
    # include extra column for payoff         
    tree = np.zeros([len(rates)+1, len(rates)+1])
    
    # assign security payoff
    tree[:,len(tree)-1] = payoff(notion, typ)
    
    # interate through the price tree    
    for col in reversed(range(0,len(tree)-1)):  
        
        for row in range(0, col+1):
            
            rate = rates[row,col]
            pu = pd = 1/2 
            tree[row, col] = np.exp(-1*rate*delta)* \
                             (pu*(tree[row, col+1]+cf[row,col+1]) + pd*(tree[row+1, col+1]+cf[row+1, col+1]))      
    
    return (tree[0,0], tree) 


# Unit testing        
if __name__ == "__main__":
        
    # theta = [0.021145, 0.013807] 
    # small ho-lee tree
    # ho_lee = rateTree(0.0169, [0.021145, 0.013807], 0.015, 0.5, 'BDT')
    
    zero_coupons = pd.read_csv('https://raw.githubusercontent.com/wrcarpenter/Interest-Rate-Models/main/Data/zcbs.csv') 
    zcbs = zero_coupons.loc[zero_coupons['Date']=='3/8/2024']
    zcbs = zcbs.drop("Date", axis=1)
    
    # small example for calibration
    zeros   = np.array(zcbs.iloc[:,0:60])
    x       = build(zeros, 0.011, 1/12) 
    tree_hl = rateTree(x[0], x[2], 0.011, 1/12)
     
    cashfl  = cf_bond(tree_hl, 5.00, 1/12, 1, 0.00)
    pricing = priceTree(tree_hl, 1/2, cashfl, 1/12, "bond", 1)
    
    print(zeros[0,zeros.shape[1]-1])
    print(pricing[0])
    print(pricing[0] - zeros[0,zeros.shape[1]-1])
    
    # pd.DataFrame(tr).to_clipboard()
    short_tree = tree_hl[:5,:5]
    cashfl  = cf_bond(short_tree, 5.00, 1/12, 1, 0.00)
    short   = priceTree(short_tree, 1/2, cashfl, 1/12, "bond", 1)
    
    

   
    
                    






  