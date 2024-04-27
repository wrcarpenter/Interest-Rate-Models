# Interest Rate Models
Fixed income bonds and derivatives are complicated financial instruments whose pricing is often conditional on the path of interest rates in the future. The assumption that current rates will remain 'static' is overly simplistic, especially when one needs to price a security with embedded options (ex: swaption, callable bond, etc.). 

This repository focuses on the implementation of the acclaimed [Ho-Lee interest rate model](https://en.wikipedia.org/wiki/Ho%E2%80%93Lee_model) in a binomial lattic framework, which is somewhat similar to valuation methods in the equity derivative market. The fixed income industry has become more complex since this model was initially created but it has been used in practice at some point to develop pricing models for banks and other market participants. Once a model is created, one can price various securities traded daily in fixed income markets via tree discount pricing or via monte carlo simluation from rate paths generated from a given tree, such as:

![Image](https://github.com/wrcarpenter/Interest-Rate-Models/blob/main/Images/ho_lee_monte_carlo_500_simulations.png)

## Objectives
* Use market pricing data to calibrate a binomial interest tree model (Ho-Lee)
* Price various interest rate securities and derivatives with a tree model: caps, floors, swaps, bonds, etc.
* Create Monte Carlo simluations from a given tree and compare pricing results to binomial pricing
* Use Monte Carlo interest rate simulations to price path-dependent securities, like mortgages 

## Ho-Lee Rate Model Construction 
The Ho-Lee model was introduced in 1986 by Thomas Ho and Sang Bin Lee. Generally, it defines a short rate to follow a stochastic process:

```math
dr^* = \theta(t)dt + \sigma dZ^*
```
The drift term, $\theta(t)dt$, is time-varying which allows the model to be calibrated to match a given term structure of interest rates (by varying theta each period). Thus, the model will produce rates that can price a zero coupon bond in any given month to match market prices. One potential downside of this model is that it allows interest rates to become negative, which many practictioners sought to avoid because it seemed unlikely that would ever occur in real markets. However, recent negative interest rate environmetns in Japan and Europe over the past decade could make this model more plausible moving forward. 

### Model Dynamics
In order to use the Ho-Lee model to build a tree, the dynamics must be discretized so the short rate in the tree follows:

Move up the tree:
```math
r^{*}_{t+\Delta t} = r^{*}_t + \theta (t) \Delta t + \sigma \sqrt{\Delta t}
```
Move down the tree:
```math
r^{*}_{t+\Delta t} = r^{*}_t + \theta (t) \Delta t - \sigma \sqrt{\Delta t}
```
This will produce a tree model that can then be used for pricing via backwards discounting (martingale condition) or with Monte Carlo simulation where paths are sampled from the rates in the tree. 

### Determining Interest Rate Volatility $\sigma$
The Ho-Lee model assumes a constant volatility which means it cannot match a given term structure of volatility in the market. This is certainly one downside of the model because options typically have different volatilities at different maturities. Other models (such as the Black-Derman-Toy model) were subsequently created to handle a term structure of volatility. 

### Zero Coupon Bond Prices
Zero Coupon Bond (ZCB) prices are an input used for calibrating the model to market data. ZCBs for Treasury securities can be bootstrapped from market data. See [this project](https://github.com/wrcarpenter/Z-Spread) that goes into detail on how to implement a bootstapping method in python and subsequently use ZCBs to determine the yield-spreads of various bonds.  

### Calibrating for Theta $\theta$ to Price Zero Coupon Bonds

Construct tree interatively solving to calibrate a full rate tree. 

### Binomial Tree Pricing Method
Price bonds.

### Monte Carlo Pricing Method
Generate a monte carlo simulation. 

## Cap Pricing 

## Swap Pricing 

## Bond Pricing

