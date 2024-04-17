import pandas as pd
import numpy as np
import scipy 
from scipy.stats import norm
from scipy.stats import multivariate_normal
#Read Data
df = df = pd.read_csv('~/Downloads/current.csv')
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
print("The parameters of the OLS model are", params)

# to maximize likelihood a function of the negative likelihood is defined to be minimized
def cobj(params, y): 
    return - cond_loglikelihood_ar7(params,y)

# Maximizing the likelihood
results = scipy.optimize.minimize(fun = cobj, x0 =  params, args = INDPRO, method='L-BFGS-B')
print("The parameters estimated by maximizing the conditional likelihood are:", results.x)

#Same Procedure for unconditional likelihood
def uobj(params, y): 
    return - uncond_loglikelihood_ar7(params,y)

results = scipy.optimize.minimize(uobj, params, args = INDPRO, method='Nelder-Mead')
print("The parameters estimated by maximizing the unconditional likelihood are:", results.x)

# Now we estimate multiple models:
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


mods = np.array([modOLS, mod1, mod2, mod3, mod4])
modsDF = pd.DataFrame(mods)
modsDF = modsDF.T
rownames = ["$c$", "$\phi_1$", "$\phi_2$", "$\phi_3$", "$\phi_4$", 
            "$\phi_5$", "$\phi_6$", "$\phi_7$", "$\sigma^2$"]
modsDF.insert(0, "Coefficients", rownames)
column_names = ("Coefficients", "OLS", "Model 1", "Model 2", "Model 3", "Model 4")
modsDF.columns = column_names
#print(modsDF.to_latex())

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
### L-BFGS-B
#mod7 = scipy.optimize.minimize(fun = uobj, x0 =  Initial_Guess, args = INDPRO, method='L-BFGS-B').x
### Nelder-Mead
#mod8 = scipy.optimize.minimize(fun = uobj, x0 =  Initial_Guess, args = INDPRO, method='Nelder-Mead').x

umods = np.array([modOLS, mod5, mod6])
umodsDF = pd.DataFrame(umods)
umodsDF = umodsDF.T
rownames = ["$c$", "$\phi_1$", "$\phi_2$", "$\phi_3$", "$\phi_4$", 
            "$\phi_5$", "$\phi_6$", "$\phi_7$", "$\sigma^2$"]
umodsDF.insert(0, "Coefficients", rownames)
column_names = ("Coefficients", "OLS", "Model 5", "Model 6")
umodsDF.columns = column_names
#print(umodsDF.to_latex())

### Forecasting:
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

#print(results.to_latex())

rownames = ("$y_{t+1}$", "$y_{t+2}$", "$y_{t+3}$", "$y_{t+4}$",
            "$y_{t+5}$","$y_{t+6}$", "$y_{t+7}$", "$y_{t+8}$")
results = {'Forecasts': rownames,'OLS': forecastAR(), 'Model 5': forecastAR(model = umods[1]),
           'Model 6': forecastAR(model = umods[2])}
results = pd.DataFrame(results)

print(results.to_latex())