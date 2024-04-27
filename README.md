# Interest Rate Models
Fixed income bonds and derivatives are complicated financial instruments whose pricing is often conditional on the path of interest rates in the future. The assumption that current rates will remain 'static' is overly simplistic, especially when one needs to price a security with embedded options (ex: swaption, callable bond, etc.). 

This repository focuses on the implementation of the acclaimed [Ho-Lee interest rate model](https://en.wikipedia.org/wiki/Ho%E2%80%93Lee_model) in a binomial lattic framework, which is somewhat similar to valuation methods in the equity derivative market. The fixed income industry has become more complex since this model was initially created but it has been used in practice at some point to develop pricing models for banks and other market participants. Once a model is created, one can price various securities traded daily in fixed income markets via tree discount pricing or via monte carlo simluation from rate paths generated from a given tree, such as:

![Image](https://github.com/wrcarpenter/Interest-Rate-Models/blob/main/Images/ho_lee_monte_carlo_500_simulations.png)

## Objectives
* Use market pricing data to calibrate a Ho-Lee binomial interest tree model
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
This will produce a tree model that can then be used for pricing via backwards discounting (martingale condition) or with Monte Carlo simulation where paths are sampled from the rates in the tree. The following code is what implements construction of a tree once all relevant variables are provided:

```Python
def rateTree(r0, theta, sigma, delta):

    tree = np.zeros([len(theta), len(theta)])
    
    tree[0,0] = r0
       
    for col in range(1, len(tree)):
        
        tree[0, col] = tree[0, col-1] + theta[col]*delta+sigma*math.sqrt(delta)
   
    for col in range(1, len(tree)):
        for row in range(1, col+1):
            tree[row, col] = tree[row-1, col] - 2*sigma*math.sqrt(delta)

    return tree
```



### Determining Interest Rate Volatility $\sigma$
The Ho-Lee model assumes a constant volatility which means it cannot match a given term structure of volatility in the market. This is certainly one downside of the model because options typically have different volatilities at different maturities. Other models (such as the Black-Derman-Toy model) were subsequently created to handle a term structure of volatility. 

### Zero Coupon Bond Prices
Zero Coupon Bond (ZCB) prices are an input used for calibrating the model to market data. ZCBs for Treasury securities can be bootstrapped from market data. See [this project](https://github.com/wrcarpenter/Z-Spread) that goes into detail on how to implement a bootstapping method in python and subsequently use ZCBs to determine the yield-spreads of various bonds.  

### Calibrating for Theta $\theta$ to Price Zero Coupon Bonds

The Ho-Lee model has a time-varying $\theta$ that gives it the flexibilty to be calibrated to match a given term structure of interest rates. To accomplish this, market data of ZCB prices must be provided and then the tree is iteratively constructed, choosing a $\theta_i$ at each point to fit all given zero coupon bond prices. The following code is part of the functions that executes this calibration process:
```Python
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
```
The calibration process is arguably the most computationally intensive process of the tree creation process but it produces a $\theta$ array and a fully constructed tree that can immediately be used and subsequently reused for security pricing. Below is example of a calibrated model where prices of ZCBs generated from the tree are compared to the ZCB market data. The fit is extremely precise:

![Image](https://github.com/wrcarpenter/Interest-Rate-Models/blob/main/Images/zcb_calibration_24_period_tree.png)

### Binomial Tree Pricing Method
There are a few important conditions that must be met for tree pricing via the Ho-Lee model to be possible: risk neautral dymanics, martingale assumption, etc. The Ho-Lee trees created satisfy these conditions and allows backward discounting in the tree to provide valid prices for bonds and derivatives. The below algorithm is what is implemented in code that is essentially taking a weighted average of cash flows at each node in the tree and discounting them back to the present.  

```Python
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
```
This algorithm enables one to price securities like bonds, swaps, caps, floors, etc and can be expanded quickly to handle more complicated derivatives like swaptions or callable bonds with coupon payments. 

### Monte Carlo Pricing Method
Monte Carlo simulations are a popular methodology for pricing fixed income securities, especially when payoffs can be "path dependent." One common example of this can be found in mortgages, a multi-billion dollar market traded by all of Wall Street's largest banks. Assumptions about borrower prepayment often depend on interest rates they have seen in the past, thus the entire path of rates is important to simluation for future cash flows. 

With a Ho-Lee model, simulation can be introduced by randomly moving up or down the branches assuming an "up" or "down" probability of 50%. The following code performs this simulation for an arbitrary number of paths when given a rate tree:

```Python
# Monte carlo simluation 
def tree_monte_carlo(tree, paths):    

    monte = np.zeros([len(tree)-1, paths])

    monte[0,:] = tree[0,0] # assign initial interest rate 
    
    for col in range(0,monte.shape[1]):
        
        r = 0
        c = 0
        p = 0
        
        for row in range(1, monte.shape[0]):
        
            p = random.rand()
            
            if p > 0.5:
                # move through rate tree
                monte[row, col] = tree[r, c+1]
                # update position on rate tree
                r = r 
                c = c+1
            else:
                # move through rate tree
                monte[row, col] = tree[r+1, c+1]
                # update position on rate tree
                r = r + 1
                c = c + 1

    periods = np.arange(1, len(tree), 1)
    monte   = pd.DataFrame(monte)
    monte.insert(0, 'Period', periods)
    
    return monte
```
The following is how paths on simulated tree could appear and how they compare to provide ZCB pricing data used to calibrate the tree:

![Image](https://github.com/wrcarpenter/Interest-Rate-Models/blob/main/Images/monte_carlo_with_zcbs.png)

## Cap Pricing 

An interest rate cap is the following. The cap cash flows are:

```Python
def cf_cap(rates, strike, delta, notion, cpn):

    cf  = np.zeros([len(rates)+1, len(rates)+1])

    for col in range(0, len(cf)-1):
        for row in range(0, col+1):
            rate = rates[row,col]
            cf[row, col] = delta*notion*max(rate-strike/100, 0)

    return cf
```
Using the model, here is one example of how a cap price varies based on a strike rate (holding volatility, cap notional amount, etc constant). 

![Image](https://github.com/wrcarpenter/Interest-Rate-Models/blob/main/Images/cap_pricing_table.png)

![Image](https://github.com/wrcarpenter/Interest-Rate-Models/blob/main/Images/cap_pricing_chart.png)

This is sensible considering a cap with a very low strike (ex: around 1%) should be "expensive" in the current rate environment, where the current short rate is about 5% and longer dated Treasury bonds are all 4% yield or higher. The same is also true for the extremely low cost of a high-strike cap, considering that the market is not pricing in future rates as high as 9-12%. 

## Swap Pricing 

An interest rate swap is the following. The swap cash flows are:

```Python
def cf_swap(rates, strike, delta, notion, cpn):
    
    f  = np.zeros([len(rates)+1, len(rates)+1])

    for col in range(0, len(cf)-1):
        for row in range(0, col+1):
            rate = rates[row,col]
            cf[row, col] = delta*notion*(rate-strike/100)    
            
    return cf
```
![Image](https://github.com/wrcarpenter/Interest-Rate-Models/blob/main/Images/swap_pricing_table.png)

![Image](https://github.com/wrcarpenter/Interest-Rate-Models/blob/main/Images/swap_pricing_chart.png)

## Bond Pricing

A bond is the following. The bond cash flows are. We assume here that the bond is not callable and does not have embedded borrower options.

```Python
def cf_bond(rates, strike, delta, notion, cpn):
    
    cf  = np.zeros([len(rates)+1, len(rates)+1])

    for col in range(0, len(cf)-1):
        for row in range(0, col+1):
            cf[row, col] = delta*notion*cpn/100  
    
    return cf
```



