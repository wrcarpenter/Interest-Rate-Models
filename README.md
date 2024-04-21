# Interest Rate Models
Fixed income bonds and derivatives are complicated financial instruments whose pricing is often conditional on the path of interest rates in the future. The assumption that current rates will remain 'static' is overly simplistic, especially when one needs to price a security with embedded options (ex: swaption). 

This repository focuses on the implementation of two interest rate models in a binomial lattic framework, somewhat similar to valuation methods in the equity derivative market beyond Black-Scholes. The two models are: Ho-Lee and Black-Derman-Toy. The fixed income industry has become more complex since these were intially created but they have both been used in practice at some point to develop pricing models for banks and other market participants. 

## Table of Contents

## Objectives
* Use market data to calibrate a binomail interest tree model (Ho-Lee or Black-Derman-Toy)
* Price various interest rate securities and derivatives with a tree model: caps, floors, swaps, bonds, etc.
* Create Monte Carlo simluations from a given tree and compare pricing results to binomial pricing
* Use Monte Carlo interest rate simulations to price path-dependent securities, like mortgages 

## Construction 
The tree should be calibrated to price zero coupon bond prices given the market. Volatility is typically based of of forward volatility implied by caplets or other rate options like swaptions. This project does not have access to that kind of pricing data.

Construct tree interatively solving to calibrate a full rate tree. 

## Binomial Tree Pricing Method
Price bonds.

## Monte Carlo Pricing Method
Generate a monte carlo simulation. 


