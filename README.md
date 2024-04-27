# Interest Rate Models
Fixed income bonds and derivatives are complicated financial instruments whose pricing is often conditional on the path of interest rates in the future. The assumption that current rates will remain 'static' is overly simplistic, especially when one needs to price a security with embedded options (ex: swaption). 

This repository focuses on the implementation of the acclaimed [Ho-Lee interest rate model](https://en.wikipedia.org/wiki/Ho%E2%80%93Lee_model) in a binomial lattic framework, which is somewhat similar to valuation methods in the equity derivative market. The fixed income industry has become more complex since this model was initiall created but it has been used in practice at some point to develop pricing models for banks and other market participants.

## Table of Contents

## Objectives
* Use market pricing data to calibrate a binomial interest tree model (Ho-Lee)
* Price various interest rate securities and derivatives with a tree model: caps, floors, swaps, bonds, etc.
* Create Monte Carlo simluations from a given tree and compare pricing results to binomial pricing
* Use Monte Carlo interest rate simulations to price path-dependent securities, like mortgages 

## Ho-Lee Rate Model Construction 
The tree should be calibrated to price zero coupon bond prices given the market. Volatility is typically based of of forward volatility implied by caplets or other rate options like swaptions. This project does not have access to that kind of pricing data.

## Model Dynamics

## Determining Interest Rate Volatility $\sigma$

## Calibrating for Theta $\theta$ to Price Zero Coupon Bonds

Construct tree interatively solving to calibrate a full rate tree. 

## Binomial Tree Pricing Method
Price bonds.

## Monte Carlo Pricing Method
Generate a monte carlo simulation. 


