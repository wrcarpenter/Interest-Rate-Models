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
    # interatively fill up the array
    for col in range(0, len(cf)-1):
        for row in range(0, col+1):
            rate = rates[row,col]
            cf[row, col] = math.exp(rate*-1*delta)*delta*notion*max(strike/100-rate, 0)
            
    return cf 


def cf_cap(rates, strike, delta, notion, cpn):
    pass


def cf_bond(rates, strike, delta, notion, cpn):
    pass
    

def cf_swap(rates, strike, delta, notion, cpn):
    pass    

# floor_cf = lambda rate, strike, delta, notion, cpn : math.exp(-1*rate*delta)*delta*notion*max(strike-rate,0)
# swap_cf  = lambda rate, strike, delta, notion, cpn : math.exp(-1*rate*delta)*delta*notion*(rate - strike)
# bond_cf  = lambda rate, strike, delta, notion, cpn : notion*delta*cpn
# zcb_cf   = lambda rate, strike, delta, notion, cpn : 0 

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

def rateTree():
    # create an interest rate tree based on a defined model for pricing 
    pass

def cfTree(rates, strike, delta, notion, cpn, cf_type):
    
    '''
    Cash Flow Tree Function
    
    Returns periodic cash flow given a certain type of asset.
    
    '''
    
    cf = np.zeros([len(rates), len(rates)])
    cf[np.triu_indices(len(cf),0)] = cf_type(rates, strike, delta, notion, cpn)
    
    return cf             
                                       
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

    rateTree = pd.read_csv("https://raw.githubusercontent.com/wrcarpenter/Fixed-Income-Valuation/main/Data/testTree.csv", header=0).values

    # Testing a cash flow generation
    
    # Bond?
    print(one_period_tree.shape)
    
    
    res = cf_floor(two_period_tree, 1.00, 1/2, 100, 5.00)
        
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



  