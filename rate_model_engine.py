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


# Asset Payoff Lamda Functions 
call    = lambda x: max(x-100, 0)
put     = lambda x: max(100-x, 0)
forward = lambda x: x - 100
cap     = lambda x: 0 
floor   = lambda x: 0
swap    = lambda x: 0
collar  = lambda x: 0
bond    = lambda x: x


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
    Bond Cash flows
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
            cf[row, col] = strike/100 - rate     
            
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
    price = np.zeros([i+1, i+1])
    
    # Assign new rates
    for row in range(0, len(tree)):
        if row == 0: 
            tree[row, i] = tree[row, i-1] + theta*delta + sigma*math.sqrt(delta)
        else:
            tree[row, i] = tree[row-1, i-1] + theta*delta - sigma*math.sqrt(delta)
            
            
    # fill in the pricing tree
    for row in range(0, len(price)):
        # rate
        price[row, i] = 1/((1+tree[row, i])**((i+1)/12))

    # need pricing tree?    
    for col in reversed(range(0, i)):
       
        for row in range(0, col+1):
            
            node = 1/((1+tree[col, row])**(col/12)) # get that discount factor 
            price[row, col] = node*(1/2*price[row, col+1] + 1/2*price[row+1, col+1])
           
    return price[0,0] - zcb    
    
def calibrate(tree, zcb, i, sigma, delta):

    '''
    Calibrated a rate tree - solving for sigma to match ZCB prices. 
    '''
    
    # add argument into a solver function
    t0    = -0.20
    miter = 10000
    
    # this should be a loop that assembles all theta and returns
    theta = newton(solver, t0, args=(tree, zcb, i, sigma, delta))
    
    # update rate tree with theta here
    for row in range(0, i+1):
        if row == 0: 
            tree[row, i] = tree[row, i-1] + theta*delta + sigma*math.sqrt(delta)
        else:
            tree[row, i] = tree[row-1, i-1] + theta*delta - sigma*math.sqrt(delta)
    
    
    return [theta, tree]
    
        
def build(zcb, sigma, delta):
    
    # empty rates tree
    tree  = np.zeros([zcb.shape[1]+1, zcb.shape[1]+1])
    # empty theta tree
    theta = np.zeros([zcb.shape[1]]) 
    
    # Initial Zero Coupon rate (monthly)
    tree[0,0] = ((1/zcb[0,0])**(1/(delta)))-1
    r0        = tree[0,0]
    
    for i in range(1, len(theta)):
        
        # here you need to solve for theta and also get an updated tree        
        solved   = calibrate(tree, zcb[0,i], i, sigma, delta)
        
        # update theta array ... it does not need previous theta but it needs updated rates
        theta[i] = solved[0]
        tree     = solved[1]
        
        
            
    # you have effectivley calibrated and already created the tree
    # return [r0, rateTree, theta]
    display(tree)
    return [r0, tree, theta]
    
def rateTree(r0, theta, sigma, delta, model):

    '''
    General Rate Model Tree Function
    
    Sigma (volatility) can be multi-dimentional for the full BDT model
    Theta is multi-dimentional
    '''

    tree = np.zeros([len(theta)+1, len(theta)+1])
    # initialize tree
    tree[0,0] = r0
       
    # fill in first row 
    for col in range(1, len(tree)-1):
        
        tree[0, col] = tree[0, col-1] + theta[col]*delta+sigma*math.sqrt(delta)
   
    
    for col in range(1, len(tree)-1):
        for row in range(1, col+1):
            tree[row, col] = tree[row-1, col] - 2*sigma*math.sqrt(delta)
    
    return tree
                                            
def priceTree(rates, prob, cf, delta, payoff, notion):
    
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

    tree[:,len(tree)-1] = payoff(notion)
        
    for col in reversed(range(0,len(tree)-1)):  
        
        for row in range(0, col+1):
            
            rate = rates[row,col]
            # cf_d = cashflow(rate, )
            # cf_u = cashflow(rate, strike, delta, notional, cpn)
            pu = pd = 1/2 # always equal to 1/2 from prob tree
            
            tree[row, col] = np.exp(-1*rate*delta)* \
                             (pu*(tree[row, col+1]+cf[row,col+1]) + pd*(tree[row+1, col+1]+cf[row+1, col+1]))      

    return tree[0,0]  



# Unit testing        
if __name__ == "__main__":
        
    # theta = [0.021145, 0.013807] 
    # small ho-lee tree
    # ho_lee = rateTree(0.0169, [0.021145, 0.013807], 0.015, 0.5, 'BDT')
    
    zero_coupons = pd.read_csv('https://raw.githubusercontent.com/wrcarpenter/Interest-Rate-Models/main/Data/zcbs.csv')
    
    zcbs = zero_coupons.loc[zero_coupons['Date']=='3/8/2024']
    zcbs = zcbs.drop("Date", axis=1)
    
    # zcbs = np.array(zcbs.iloc[:,0:4])  # these are my zero coupon bonds
    
    zeros = np.array(zcbs.iloc[:,0:4])
    x     = build(zeros, 0.009, 1/12)
    tr    = rateTree(x[0], x[2], 0.009, 1/12, 'HL')
    c     = cf_bond(tr, 5.00, 1/12, 1, 0.00)
    p     = priceTree(tr, 1/2, c, 1/12, bond, 1)
    
    
    
    
     
    # Calibrating the tree
    result = build(zcbs, 0.009, 1/12)
    
    holee = rateTree(result[0], result[2], 0.009, 1/12, 'HL')
    
    zcbcf = cf_bond(holee, 5.00, 1/12, 1, 0.00) # zero coupon bond
    x    = priceTree(holee, 1/2, zcbcf, 1/12, bond, 1)
    
    
    # rate tree builder function works correctly  
    # test  = rateTree(0.045, [0.02, 0.02, 0.02, 0.02, 0.02], 0.001, 1/12, "HL")
    # zcbcf = cf_bond(holee, 5.00, 1/12, 1, 0.00) # zero coupon bond
    # px    = priceTree(holee, 1/2, zcbcf, 1/12, bond, 1)
    
    
    
                    






  