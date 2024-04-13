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

# Asset Payoff Lamda Functions 
call    = lambda x: max(x-100, 0)
put     = lambda x: max(100-x, 0)
forward = lambda x: x - 100
cap     = lambda x: 0 
floor   = lambda x: 0
swap    = lambda x: 0
collar  = lambda x: 0
bond    = lambda x: x


# Create payoff functions then multiply them by a triangle array with zero and ones
# the rates tree is already created/given for pricing 

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
    # interatively fill up the array
    for col in range(0, len(cf)-1):
        for row in range(0, col+1):
            cf[row, col] = delta*notion*cpn/100  
    
    return cf

def cf_swap(rates, strike, delta, notion, cpn):
    cf  = np.zeros([len(rates), len(rates)])
    # interatively fill up the array
    for col in range(0, len(cf)-1):
        for row in range(0, col+1):
            rate = rates[row,col]
            cf[row, col] = strike/100 - rate     
            
    return cf

# Print helper function 
def display(arr):
  for i in arr:
    for j in i:
        print("{:8.4f}".format(j), end="  ")
    print() 
  print("\n")

# Defining tree probabilities 
def probTree(length):
    '''
    Generating a probability tree - assumed to be up/down 50%/50% for this project

    '''
    prob = np.zeros((length, length))
    prob[np.triu_indices(length, 0)] = 0.5
    return(prob)


def rateTree(r0, theta, sigma, delta, model):
    # create an interest rate tree based on a defined model for pricing 
    '''
    General Rate Model Tree Function
    
    Sigma (volatility) can be multi-dimentional for the full BDT model
    Theta is multi-dimentional
    '''

    # generate an interest rate tree of size N+1 because the last period is payoff

    tree = np.zeros([len(theta)+2, len(theta)+2])
    # initialize tree
    tree[0,0] = r0
       
    # fill in first row 
    for col in range(1, len(tree)-1):
        
        tree[0, col] = tree[0, col-1] + theta[col-1]*delta+sigma*math.sqrt(delta)
   
    
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
            pu = pd = prob[row, col] # always equal to 1/2 from prob tree
            
            tree[row, col] = np.exp(-1*rate*delta)* \
                             (pu*(tree[row, col+1]+cf[row,col+1]) + pd*(tree[row+1, col+1]+cf[row+1, col+1]))      

    return tree[0,0]  


# Unit testing        
if __name__ == "__main__":
    
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
    
    theta = [0.021145, 0.013807]
    
    # small ho-lee tree
    ho_lee = rateTree(0.0169, [0.021145, 0.013807], 0.015, 0.5, 'BDT')
    
    
    
    
        
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



  