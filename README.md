# Interest Rate Models
This project implements binomial trees to model interest rates, which can be calibrated to market data and used to price a variety of fixed income securities. 

## Objectives
* Use market data to calibrate a binomail interest tree model (Ho-Lee or Black-Derman-Toy)
* Price various interest rate securities and derivatives with a tree model: caps, floors, swaps, bonds, etc.
* Create Monte Carlo simluations from a given tree and compare pricing results to binomial pricing
* Use Monte Carlo interest rate simulations to price path-dependent securities, like mortgages 

## Construction 
The tree should be calibrated to price zero coupon bond prices given the market. Volatility is typically based of of forward volatility implied by caplets or other rate options like swaptions. This project does not have access to that kind of pricing data.

Construct tree interatively solving to calibrate a full rate tree. 

## Binomial Tree Pricing
Price bonds.

## Monte Carlo Pricing
Generate a monte carlo simulation. 


