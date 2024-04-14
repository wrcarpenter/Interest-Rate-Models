# -*- coding: utf-8 -*-
"""
Generate Zero Coupon Bond Prices

Helper Fuction 

"""
import sys
import os
import math 
import pandas as pd
import numpy as np

spot_rates = pd.read_csv("https://raw.githubusercontent.com/wrcarpenter/Interest-Rate-Models/main/Data/spots-monthly.csv", header=0)
cols = list(spot_rates.columns.values)

zcbs = pd.DataFrame(np.zeros((spot_rates.shape[0], spot_rates.shape[1]), dtype=float), columns=cols)
zcbs['Date'] = spot_rates['Date']

spot_rates.to_clipboard()

for row in range(0, zcbs.shape[0]):
    for col in range(1, zcbs.shape[1]):
        zcbs.iloc[row, col] = 1/((1+spot_rates.iloc[row, col]/100)**(col/12))


zcbs.to_csv('C:/Users/wcarp/OneDrive/Desktop/Interest Rate Model/Data/zcbs.csv', index=False)  