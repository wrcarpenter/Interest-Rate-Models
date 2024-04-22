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
    cf  = np.zeros([len(rates), len(rates)])

    for col in range(0, len(cf)-1):
        for row in range(0, col+1):
            rate = rates[row,col]
            cf[row, col] = delta*notion*max(strike/100-rate, 0)
            
    return cf 

def cf_cap(rates, strike, delta, notion, cpn):
    '''
    Cap Cash Flows
    '''
    cf  = np.zeros([len(rates), len(rates)])

    for col in range(0, len(cf)-1):
        for row in range(0, col+1):
            rate = rates[row,col]
            cf[row, col] = delta*notion*max(rate-strike/100, 0)

    return cf


def cf_bond(rates, strike, delta, notion, cpn):
    
    '''
    Bond Cashflows
    
    Strike input is not important.
    
    '''
    cf  = np.zeros([len(rates), len(rates)])

    for col in range(0, len(cf)-1):
        for row in range(0, col+1):
            cf[row, col] = delta*notion*cpn/100  
    
    return cf

def cf_swap(rates, strike, delta, notion, cpn):
    
    f  = np.zeros([len(rates), len(rates)])

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
    '''
    Generating a probability tree - assumed to be up/down 50%/50% for this project

    '''
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
    
            
    # now we need to handle BDT tree
    
    # need pricing tree?    
    for col in reversed(range(0, i+1)):
        for row in range(0, col+1):
            node = np.exp(-1*tree[row, col]*(delta))
            price[row, col] = node*(1/2*price[row, col+1] + 1/2*price[row+1, col+1])     
    
    return price[0,0] - zcb    
    
def calibrate(tree, zcb, i, sigma, delta):

    '''
    Calibrated a rate tree - solving for sigma to match ZCB prices. 
    '''
    
    t0    = 0.5
    miter = 1000

    theta = newton(solver, t0, args=(tree, zcb, i, sigma, delta))

    for row in range(0, i+1):
        if row == 0: 
            tree[row, i] = tree[row, i-1] + theta*delta + sigma*math.sqrt(delta)
        else:
            tree[row, i] = tree[row-1, i-1] + theta*delta - sigma*math.sqrt(delta)
    
    
    return [theta, tree]
            
def build(zcb, sigma, delta, model):
    
    # empty rates tree
    tree  = np.zeros([zcb.shape[1]+1, zcb.shape[1]+1])
    # empty theta tree
    theta = np.zeros([zcb.shape[1]]) 
    
    # Initial Zero Coupon rate
    tree[0,0] = np.log(zcb[0,0])*-1/delta
    
    # Add model consideration here too
    if model == "BDT":
        r0 = np.log(tree[0,0])
    else:
        r0 = tree[0,0]
    
    for i in range(1, len(theta)):
        
        solved   = calibrate(tree, zcb[0,i], i, sigma, delta)
        
        # update theta array
        theta[i] = solved[0]
        tree     = solved[1]
    
    if model == "BDT":
        return [r0, np.exp(tree), theta]
    else:
        return [r0, tree, theta]
    
def rateTree(r0, theta, sigma, delta, model):

    '''
    General Rate Model Tree Function
    
    Sigma (volatility) can be multi-dimentional for the full BDT model
    Theta is multi-dimentional
    '''

    tree = np.zeros([len(theta)+1, len(theta)+1])
    
    # BDT model
    if model == "BDT":
        tree[0,0] = np.log(r0)
    # Ho-Lee model
    else:
        tree[0,0] = r0
       
    for col in range(1, len(tree)-1):
        
        tree[0, col] = tree[0, col-1] + theta[col]*delta+sigma*math.sqrt(delta)
   
    
    for col in range(1, len(tree)-1):
        for row in range(1, col+1):
            tree[row, col] = tree[row-1, col] - 2*sigma*math.sqrt(delta)
    
    if model == "BDT":
        return np.exp(tree)
    else:
        return tree
                                            
def priceTree(rates, prob, cf, delta, typ, notion):
    
    '''
    General Tree Pricing Function 
    
    Returns final asset price and tree of price evolution
    
    rates    : N+1 x N+1 tree of interest rates 
    prob     : N+1 x N+1 tree of up/down probabilities
    delta    : time delta, with a value of 1 being annual 
    payoff   : payoff funtion 
    notion   : notional value of the security being priced
    coupon   : coupon of the security being priced, if relevant 
    strike   : strike rate of the security being priced if relevant
    cashflow : cashflow function
    
    '''
            
    tree = np.zeros([len(rates), len(rates)])

    tree[:,len(tree)-1] = payoff(notion, typ)
        
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
    zeros = np.array(zcbs.iloc[:,0:60])
    x     = build(zeros, 0.011, 1/12)
    tree    = rateTree(x[0], x[2], 0.011, 1/12, 'HL')
    c     = cf_bond(tr, 5.00, 1/12, 1, 0.00)
    p     = priceTree(tr, 1/2, c, 1/12, "bond", 1)
    
    
    print(zeros[0,zeros.shape[1]-1])
    print(p[0])
    print(p[0] - zeros[0,zeros.shape[1]-1])
    
    pd.DataFrame(tr).to_clipboard()
    

   
    
                    






  