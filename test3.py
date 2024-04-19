from scipy.stats import norm
from scipy.stats import multivariate_normal
import numpy as np

def lagged_matrix(Y, max_lag=7):
    n = len(Y)
    lagged_matrix = np.full((n, max_lag), np.nan)    
    # Fill each column with the appropriately lagged data
    for lag in range(1, max_lag + 1):
        lagged_matrix[lag:, lag - 1] = Y[:-lag] if lag > 1 else Y[:-1]
    return lagged_matrix

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

import scipy 

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

# Example usage:
phis = [0.2, -0.1, 0.05, -0.05, 0.02, -0.02, 0.01]
c = 0
sigma2 = 0.5
mu, Sigma, stationary = unconditional_ar_mean_variance(c, phis, sigma2)
print("The process is stationary:", stationary)
print("Mean vector (mu):", mu)
print("Variance-covariance matrix (Sigma);", Sigma)

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
    
import pandas as pd

#Read Data
df = df = pd.read_csv('C:/Users/Dell/Desktop/cours_S2/comput_tools/current_dataset.csv')
#Select INDPRO
INDPRO = df['INDPRO']
#Drop first Row
INDPRO = INDPRO.drop(index=0)
#transform INDPRO using log differences
INDPRO = np.log(INDPRO).diff()


X = lagged_matrix(INDPRO, 7)
yf = INDPRO[7:]
Xf = np.hstack((np.ones((len(INDPRO)-7,1)), X[7:,:]))
beta = np.linalg.solve(Xf.T@Xf, Xf.T@yf)
sigma2_hat = np.mean((yf - Xf@beta)**2)
params= np.hstack((beta, sigma2_hat))
print("The parameters of the OLS model are", params)


# Defining the function to minimize for maximizing likelihood
def cobj(params, yf):
    return -cond_loglikelihood_ar7(params, yf)

# Maximizing the likelihood
results = scipy.optimize.minimize(fun=cobj, x0=params, args=yf, method='L-BFGS-B')
print("The parameters estimated by maximizing the conditional likelihood are:", results.x)

# Forecasting using OLS parameters
def forecast_ols(params, yf, horizon=7):
    # Extract parameters
    beta = params[:-1]
    sigma2 = params[-1]
    
    # Construct lagged matrix for forecast period
    X_forecast = lagged_matrix(y[-7:], 7)
    X_forecast = np.hstack((np.ones((horizon, 1)), X_forecast))
    
    # Forecast future values
    forecast = X_forecast @ beta
    return forecast

ols_forecast = forecast_ols(params, yf, horizon=7)
print("Forecast using OLS parameters:", ols_forecast)


# Defining the function to minimize for maximizing likelihood
def uobj(params, yf): 
    return - uncond_loglikelihood_ar7(params,yf)

bounds_constant = tuple((-np.inf, np.inf) for _ in range(1))
bounds_phi = tuple((-1, 1) for _ in range(7))
bounds_sigma = tuple((0,np.inf) for _ in range(1))
bounds = bounds_constant + bounds_phi + bounds_sigma

## L-BFGS-B support bounds
results = scipy.optimize.minimize(uobj, results.x, args = yf, method='L-BFGS-B', bounds = bounds)
print("The parameters estimated by maximizing the unconditional likelihood are:", results.x)

# Forecasting using maximum likelihood parameters 

def forecast_max_likelihood(params, yf, horizon=7):
    # Extract parameters
    beta = params[:-1]
    sigma2 = params[-1]
    
    # Construct lagged matrix for forecast period
    X_forecast = lagged_matrix(yf[-7:], 7)
    X_forecast = np.hstack((np.ones((horizon, 1)), X_forecast))
    
    # Forecast future values
    forecast = X_forecast @ beta
    return forecast

# Extracted parameters from maximizing likelihood
max_likelihood_params = results.x

# Forecast using maximum likelihood parameters
max_likelihood_forecast = forecast_max_likelihood(max_likelihood_params, yf, horizon=7)

# Print the forecast
print("Forecast using maximum likelihood parameters:", max_likelihood_forecast)
