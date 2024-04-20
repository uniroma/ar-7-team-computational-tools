#This is the Code for assignment 2

import pandas as pd
import numpy as np
import scipy 
import scipy.optimize
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt 

#Read Data
df = pd.read_csv('~/Downloads/current.csv')
#Select INDPRO
INDPRO = df['INDPRO']
#Drop first Row
INDPRO = INDPRO.drop(index=0)
#transform INDPRO using log differences
INDPRO = np.log(INDPRO).diff().dropna()

#implement Starter Code from the assignment
## Lagged Matrix Function
def lagged_matrix(Y, max_lag=7):
    n = len(Y)
    lagged_matrix = np.full((n, max_lag), np.nan)    
    # Fill each column with the appropriately lagged data
    for lag in range(1, max_lag + 1):
        lagged_matrix[lag:, lag - 1] = Y[:-lag]
    return lagged_matrix

## Mean- Variance - Stationarity Function
def unconditional_ar_mean_variance(c, phis, sigma2):
    ## The length of phis is p
    p = len(phis)
    A = np.zeros((p, p))
    A[0, :] = phis
    A[1:, 0:(p-1)] = np.eye(p-1)
    ## Check for stationarity
    eigA = np.linalg.eig(A)
    if all(np.abs(eigA.eigenvalues)<1):
        stationary = True
    else:
        stationary = False
    # Create the vector b
    b = np.zeros((p, 1))
    b[0, 0] = c
    
    # Compute the mean using matrix algebra
    I = np.eye(p)
    mu = np.linalg.inv(I - A) @ b
    
    # Solve the discrete Lyapunov equation
    Q = np.zeros((p, p))
    Q[0, 0] = sigma2
    #Sigma = np.linalg.solve(I - np.kron(A, A), Q.flatten()).reshape(7, 7)
    Sigma = scipy.linalg.solve_discrete_lyapunov(A, Q)
    
    return mu.ravel(), Sigma, stationary

## Conditional Likelihood
def cond_loglikelihood_ar7(params, y):
    c = params[0] 
    phi = params[1:8]
    sigma2 = params[8]
    mu, Sigma, stationary = unconditional_ar_mean_variance(c, phi, sigma2)
    ## We could check that at phis the process is stationary and return -Inf if it is not
    if not(stationary):
        return -np.inf
    ## The distribution of 
    # y_t|y_{t-1}, ..., y_{t-7} ~ N(c+\phi_{1}*y_{t-1}+...+\phi_{7}y_{t-7}, sigma2)
    ## Create lagged matrix
    X = lagged_matrix(y, 7)
    yf = y[7:]
    Xf = X[7:,:]
    loglik = np.sum(norm.logpdf(yf, loc=(c + Xf@phi), scale=np.sqrt(sigma2)))
    return loglik

## Unconditional Likelihood
def uncond_loglikelihood_ar7(params, y):
    ## The unconditional loglikelihood
    ## is the unconditional "plus" the density of the
    ## first p (7 in our case) observations
    cloglik = cond_loglikelihood_ar7(params, y)

    ## Calculate initial
    # y_1, ..., y_7 ~ N(mu, sigma_y)
    c = params[0] 
    phi = params[1:8]
    sigma2 = params[8]
    mu, Sigma, stationary = unconditional_ar_mean_variance(c, phi, sigma2)
    if not(stationary):
        return -np.inf
    mvn = multivariate_normal(mean=mu, cov=Sigma, allow_singular=True)
    uloglik = cloglik + mvn.logpdf(y[0:7])
    return uloglik
    
# Using INDPRO as the target variable.
## Computing OLS
X = lagged_matrix(INDPRO, 7)
yf = INDPRO[7:]
Xf = np.hstack((np.ones((len(INDPRO)-7,1)), X[7:,:]))
beta = np.linalg.solve(Xf.T@Xf, Xf.T@yf)
sigma2_hat = np.mean((yf - Xf@beta)**2)
params= np.hstack((beta, sigma2_hat))
#print("The parameters of the OLS model are", params)

# to maximize likelihood a function of the negative likelihood is defined to be minimized
def cobj(params, y): 
    return - cond_loglikelihood_ar7(params,y)

#Same Procedure for unconditional likelihood
def uobj(params, y): 
    return - uncond_loglikelihood_ar7(params,y)

# First we estimate multiple unconstrained models:
## OLS model
modOLS = np.hstack((beta, sigma2_hat))
## Conditional Likelihood
## 1. Using the OLS parameters as the initial guess
### L-BFGS-B
mod1 = scipy.optimize.minimize(fun = cobj, x0 =  modOLS, args = INDPRO, method='L-BFGS-B').x
### Nelder-Mead
mod2 = scipy.optimize.minimize(fun = cobj, x0 =  modOLS, args = INDPRO, method='Nelder-Mead').x
## 2. Using a slightly different initial guess
Initial_Guess = np.array([
    0.0012, ## c
    0.0291, 0.07, 0.059, 0.04, 0.04, 0.02, 0.06, ## phi
    0.009 ## sigma2 
])
### L-BFGS-B
mod3 = scipy.optimize.minimize(fun = cobj, x0 =  Initial_Guess, args = INDPRO, method='L-BFGS-B').x
### Nelder-Mead
mod4 = scipy.optimize.minimize(fun = cobj, x0 =  Initial_Guess, args = INDPRO, method='Nelder-Mead').x

# Compiling models to print to LateX table
def as_latex_table(rows, rownames, colnames, caption):
    df = np.array(rows)
    df = pd.DataFrame(df)
    df = df.T
    df.insert(0, colnames[1], rownames)
    df.columns = column_names
    tabular = df.to_latex(index=False,
                           caption=caption)
    return tabular

column_names = ("Coefficients", "OLS", "Model 1", "Model 2", "Model 3", "Model 4")
rownames = ["$c$", "$\\phi_1$", "$\\phi_2$", "$\\phi_3$", "$\\phi_4$", 
            "$\\phi_5$", "$\\phi_6$", "$\\phi_7$", "$\\sigma^2$"]
caption="Results of the unconstrained maximization of the conditional likelihood"
#print(as_latex_table([modOLS, mod1, mod2, mod3, mod4], rownames=rownames, colnames=column_names, caption=caption))

## We then reestimate these models unsing constrained maximization:
bounds_constant = tuple((-np.inf, np.inf) for _ in range(1))
bounds_phi = tuple((-1, 1) for _ in range(7))
bounds_sigma = tuple((0.000001,np.inf) for _ in range(1))
bounds = bounds_constant + bounds_phi + bounds_sigma

## Conditional Likelihood
## 1. Using the OLS parameters as the initial guess
### L-BFGS-B
mod1 = scipy.optimize.minimize(fun = cobj, x0 =  modOLS, args = INDPRO, method='L-BFGS-B', bounds=bounds).x
### Nelder-Mead
mod2 = scipy.optimize.minimize(fun = cobj, x0 =  modOLS, args = INDPRO, method='Nelder-Mead', bounds=bounds).x
## 2. Using a slightly different initial guess
Initial_Guess = np.array([
    0.0012, ## c
    0.0291, 0.07, 0.059, 0.04, 0.04, 0.02, 0.06, ## phi
    0.009 ## sigma2 
])
### L-BFGS-B
mod3 = scipy.optimize.minimize(fun = cobj, x0 =  Initial_Guess, args = INDPRO, method='L-BFGS-B', bounds=bounds).x
### Nelder-Mead
mod4 = scipy.optimize.minimize(fun = cobj, x0 =  Initial_Guess, args = INDPRO, method='Nelder-Mead', bounds=bounds).x

column_names = ("Coefficients", "OLS", "Model 1", "Model 2", "Model 3", "Model 4")
rownames = ["$c$", "$\\phi_1$", "$\\phi_2$", "$\\phi_3$", "$\\phi_4$", 
            "$\\phi_5$", "$\\phi_6$", "$\\phi_7$", "$\\sigma^2$"]
caption="Results of the constrained maximization of the conditional likelihood"
#print(as_latex_table([modOLS, mod1, mod2, mod3, mod4], rownames=rownames, colnames=column_names, caption=caption))


## Unbounded Unconditional Likelihood
## 1. Using the OLS parameters as the initial guess
### L-BFGS-B
mod5 = scipy.optimize.minimize(fun = uobj, x0 =  modOLS, args = INDPRO, method='L-BFGS-B').x
### Nelder-Mead
mod6 = scipy.optimize.minimize(fun = uobj, x0 =  modOLS, args = INDPRO, method='Nelder-Mead').x
## 2. Using a slightly different initial guess
Initial_Guess = np.array([
    0.0012, ## c
    0.00291, 0.007, 0.0509, 0.0024, 0.0409, 0.012, 0.0601, ## phi
    0.009 ## sigma2 
])


column_names = ("Coefficients", "OLS", "Model 5", "Model 6")
rownames = ["$c$", "$\\phi_1$", "$\\phi_2$", "$\\phi_3$", "$\\phi_4$", 
            "$\\phi_5$", "$\\phi_6$", "$\\phi_7$", "$\\sigma^2$"]
caption="Results of the unconstrained maximization of the unconditional likelihood"
#print(as_latex_table([modOLS, mod5, mod6], rownames=rownames, colnames=column_names, caption=caption))

## Bounded Unconditional Likelihood
## 1. Using the OLS parameters as the initial guess
### L-BFGS-B
mod5 = scipy.optimize.minimize(fun = uobj, x0 =  modOLS, args = INDPRO, method='L-BFGS-B', bounds=bounds).x
### Nelder-Mead
mod6 = scipy.optimize.minimize(fun = uobj, x0 =  modOLS, args = INDPRO, method='Nelder-Mead', bounds=bounds).x
## 2. Using a slightly different initial guess
Initial_Guess = np.array([
    0.0012, ## c
    0.00291, 0.007, 0.0509, 0.0024, 0.0409, 0.012, 0.0601, ## phi
    0.009 ## sigma2 
])
mod7= scipy.optimize.minimize(fun = uobj, x0 =  Initial_Guess, args = INDPRO, method='L-BFGS-B', bounds=bounds).x
mod8= scipy.optimize.minimize(fun = uobj, x0 =  Initial_Guess, args = INDPRO, method='Nelder-Mead', bounds=bounds).x

caption="Results of the constrained maximization of the unconditional likelihood"
column_names = ("Coefficients", "OLS", "Model 5", "Model 6", "Model 7", "Model 8")

#print(as_latex_table([modOLS, mod5, mod6, mod7, mod8], rownames=rownames, colnames=column_names, caption=caption))

### Forecasting:
mods = np.array([modOLS, mod1, mod2, mod3, mod4, mod5, mod6, mod7, mod8])
def forecastAR(h=8, model = mods[0]): # defaults: h is the forecasting horizon, mods[0] the OLS model
    forecastArray = np.empty(h) # Empty array to store forecasts
    lastrow = np.array(INDPRO.tail(7)) #use last 7 rows of INDPRO for the first forecast
    lastrow = np.flip(lastrow) # invert array to align with order of estimated parameters in the model
    for i in range(1, h+1):
        forecast = model[0] + model[1:8] @ lastrow # mods[0] uses the OLS estimates
        lastrow = np.insert(lastrow, 0, forecast)
        lastrow = np.delete(lastrow, -1)
        forecastArray[i-1] = forecast

    return forecastArray

rownames = ("$y_{t+1}$", "$y_{t+2}$", "$y_{t+3}$", "$y_{t+4}$",
            "$y_{t+5}$","$y_{t+6}$", "$y_{t+7}$", "$y_{t+8}$")
results = {'Forecasts': rownames,'OLS': forecastAR(), 'Model 1': forecastAR(model = mods[1]),
           'Model 2': forecastAR(model = mods[2]),'Model 3': forecastAR(model = mods[3]),
           'Model 4': forecastAR(model = mods[4])}
results = pd.DataFrame(results)

caption="Forecasts using the estimated coefficients from OLS and the bounded conditional likelihood maximization."
#print(results.to_latex(index=False, caption=caption))

rownames = ("$y_{t+1}$", "$y_{t+2}$", "$y_{t+3}$", "$y_{t+4}$",
            "$y_{t+5}$","$y_{t+6}$", "$y_{t+7}$", "$y_{t+8}$")
results = {'Forecasts': rownames,'OLS': forecastAR(), 'Model 5': forecastAR(model = mods[5]),
           'Model 6': forecastAR(model = mods[6]),'Model 7': forecastAR(model = mods[7]),
           'Model 8': forecastAR(model = mods[8])}
results = pd.DataFrame(results)

caption="Forecasts using the estimated coefficients from OLS and the bounded unconditional likelihood maximization."
#print(results.to_latex(index=False, caption=caption))

# Comparing OLS, Model 3, Model 7:
rownames = ("$y_{t+1}$", "$y_{t+2}$", "$y_{t+3}$", "$y_{t+4}$",
            "$y_{t+5}$","$y_{t+6}$", "$y_{t+7}$", "$y_{t+8}$")
results = {'Forecasts': rownames,'OLS': forecastAR(), 'Model 3': forecastAR(model = mods[3]),
           'Model 7': forecastAR(model = mods[7])}
results = pd.DataFrame(results)
results['OLS - Model 3'] = results['OLS'] - results['Model 3']
results['OLS - Model 7'] = results['OLS'] - results['Model 7']
sum1 = sum(results['OLS - Model 3']**2)
sum2 = sum(results['OLS - Model 7']**2)
sum_of_deviations = ["$\\sum \\Delta^2$", " ", " ", " ", sum(results['OLS - Model 3']**2), sum(results['OLS - Model 7']**2) ]
results.loc[len(results.index)] = sum_of_deviations

caption= (
    "Forecasts using OLS, Model 3 (Bounded Conditional Likelihood, Different Initial Guess), Model 7(Bounded Unconditional Likelihood, Different Initial Guess)")
#print(results.to_latex(index=False, caption=caption, float_format="%.9f" ))

# Task 4: Comparing Forecasts by plotting them
# plotting forecasts:
# Using the data up to 2000 as training data and the rest of the data as testing data
# Creating a new dataframe containgin all the data up to a certain date, say for example 01/01/2020 and forecasted data from there onwards to compare
# actual to forecasted data.
df = df.drop(index=0)
df['sasdate'] = pd.to_datetime(df['sasdate'])
date = "01/01/2000"
lasttrain = df.index[df['sasdate']== date].to_list() 
train = INDPRO[:lasttrain[0]]
test = INDPRO[lasttrain[0]-1:]

#First, OLS Model:
X = lagged_matrix(train, 7)
yf = train[7:]
Xf = np.hstack((np.ones((len(train)-7,1)), X[7:,:]))
beta = np.linalg.solve(Xf.T@Xf, Xf.T@yf)
sigma2_hat = np.mean((yf - Xf@beta)**2)
params= np.hstack((beta, sigma2_hat))
modOLS = params

#Second, conditional likelihood:
modCon = scipy.optimize.minimize(fun = cobj, x0 =  Initial_Guess, args = train, method='L-BFGS-B', bounds=bounds).x

#Third, unconditional likelihood:
modUCon = scipy.optimize.minimize(fun = uobj, x0 =  Initial_Guess, args = train, method='L-BFGS-B', bounds=bounds).x

mods = np.array([modOLS, modCon, modUCon])
def forecastAR(h=8, model = mods[0], data=train): # defaults: h is the forecasting horizon, mods[0] the OLS model
    forecastArray = np.empty(h) # Empty array to store forecasts
    lastrow = np.array(data.tail(7)) #use last 7 rows of test data for the first forecast
    lastrow = np.flip(lastrow) # invert array to align with order of estimated parameters in the model
    for i in range(1, h+1):
        forecast = model[0] + model[1:8] @ lastrow # mods[0] uses the OLS estimates
        lastrow = np.insert(lastrow, 0, forecast)
        lastrow = np.delete(lastrow, -1)
        forecastArray[i-1] = forecast

    return forecastArray


h = 8 # forecast for h periods
forecast={'INDPRO': test[:h], 
          'Forecast OLS': forecastAR(h=h), 
          'Forecast Conditional': forecastAR(h=h, model = mods[1]),
          'Forecast Unconditional': forecastAR(h=h, model = mods[2])}

forecast = pd.DataFrame(forecast)

plotdata = {'INDPRO': train, 
            'Forecast OLS': train, 
            'Forecast Conditional': train,
            'Forecast Unconditional': train}

plotdata = pd.DataFrame(plotdata)
plotdata =  [plotdata, forecast]
plotdata = pd.concat(plotdata)
plotdata.insert(0, "sasdate",df.iloc[1:len(plotdata)+1,0])
plotdata = plotdata.tail(2*h) # seletct the h forecasted observations and the h previous observations for plotting.
print(plotdata)

plt.plot(plotdata['sasdate'], plotdata['Forecast OLS'], color = "blue")
plt.plot(plotdata['sasdate'], plotdata['Forecast Conditional'], color = "green")
plt.plot(plotdata['sasdate'], plotdata['Forecast Unconditional'], color = "red")
plt.plot(plotdata['sasdate'], plotdata['INDPRO'],color="black")
plt.legend(['Forecast OLS', 'Forecast Conditional', 'Forecast Unconditional', 'INDPRO'])
plt.show()
