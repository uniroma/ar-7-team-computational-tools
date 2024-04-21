import numpy as np
import scipy 
from scipy.stats import norm, multivariate_normal

def unconditional_ar_mean_variance(c, phis, sigma2):
    p = len(phis)
    A = np.zeros((p, p))
    A[0, :] = phis
    A[1:, 0:(p-1)] = np.eye(p-1)
    
    eigA = np.linalg.eigvals(A)
    stationary = all(np.abs(eigA) < 1)
    
    b = np.zeros((p, 1))
    b[0, 0] = c
    
    I = np.eye(p)
    mu = np.linalg.inv(I - A) @ b
    
    Q = np.zeros((p, p))
    Q[0, 0] = sigma2
    Sigma = scipy.linalg.solve_discrete_lyapunov(A, Q)
    
    return mu.ravel(), Sigma, stationary

def lagged_matrix(Y, max_lag=7):
    n = len(Y)
    lagged_matrix = np.full((n, max_lag), np.nan)    
    for lag in range(1, max_lag + 1):
        lagged_matrix[lag:, lag - 1] = Y[:-lag]
    return lagged_matrix

def cond_loglikelihood_ar7(params, y):
    c = params[0] 
    phi = params[1:8]
    sigma2 = params[8]
    mu, Sigma, stationary = unconditional_ar_mean_variance(c, phi, sigma2)
    if not stationary:
        return -np.inf
    
    X = lagged_matrix(y, 7)
    yf = y[7:]
    Xf = X[7:,:]
    loglik = np.sum(norm.logpdf(yf, loc=(c + Xf @ phi), scale=np.sqrt(sigma2)))
    return loglik

def uncond_loglikelihood_ar7(params, y):
    cloglik = cond_loglikelihood_ar7(params, y)

    c = params[0] 
    phi = params[1:8]
    sigma2 = params[8]
    mu, Sigma, stationary = unconditional_ar_mean_variance(c, phi, sigma2)
    if not stationary:
        return -np.inf
    
    mvn = multivariate_normal(mean=mu, cov=Sigma, allow_singular=True)
    uloglik = cloglik + mvn.logpdf(y[0:7])
    return uloglik

# Example Usage:
params = np.array([
    0.0, ## c
    0.2, -0.1, 0.05, -0.05, 0.02, -0.02, 0.01, ## phi
    1.0 ## sigma2    
])

## Fake data
y = np.random.normal(size=100)

## The conditional distribution
cond_loglikelihood_ar7(params, y)
## The unconditional distribution
uncond_loglikelihood_ar7(params, y)

#second part
import numpy as np
from scipy.optimize import minimize

# We define the negative log-likelihood function that we want to maximize.
def neg_log_likelihood(params, y, likelihood_func):
    return -likelihood_func(params, y)

# Maximization of the conditional likelihood function.
result_cond = minimize(neg_log_likelihood, params, args=(y, cond_loglikelihood_ar7), method='L-BFGS-B')

# Maximization of the unconditional likelihood function.
result_uncond = minimize(neg_log_likelihood, params, args=(y, uncond_loglikelihood_ar7), method='L-BFGS-B')

# Output of results
print("Maximum of the conditional likelihood function:")
print(result_cond.x)

print("\nMaximum of the unconditional likelihood function:")
print(result_uncond.x)

#third part
import pandas as pd

# Loading data from the file current.csv
data = pd.read_csv('~/Downloads/current.csv')

# Displaying the first few rows of data for verification
print(data.head())

# Extracting data from the column INDPRO
indpro_data = data['INDPRO']

# Displaying the first few values of the column INDPRO for verification
print(indpro_data.head())
import numpy as np

# Calculating monthly logarithmic differences
indpro_data = indpro_data.drop(index=0)
log_diffs = np.log(indpro_data).diff().dropna()
indpro_data['log_diff']=log_diffs
# Displaying the first few values of logarithmic differences for verification
print('This is the data:', indpro_data.head())
print(indpro_data['log_diff'])
# Initial parameter values
initial_params = np.array([0.0, 0.2, -0.1, 0.05, -0.05, 0.02, -0.02, 0.01, 1.0])
initial_params = np.array([0.01, 0.2, -0.1, 0.05, -0.05, 0.02, -0.02, 0.01, 1.0])
bounds_constant = tuple((-np.inf, np.inf) for _ in range(1))
bounds_phi = tuple((-1, 1) for _ in range(7))
bounds_sigma = tuple((0.000001,np.inf) for _ in range(1))
bounds = bounds_constant + bounds_phi + bounds_sigma

# Maximization of the conditional likelihood function
result_cond = minimize(neg_log_likelihood, initial_params, args=(indpro_data['log_diff'], cond_loglikelihood_ar7), method='L-BFGS-B', bounds=bounds)

# Maximization of the unconditional likelihood function
result_uncond = minimize(neg_log_likelihood, initial_params, args=(indpro_data['log_diff'], uncond_loglikelihood_ar7), method='L-BFGS-B', bounds=bounds)

def cobj(params, y):
    return - cond_loglikelihood_ar7(params, y)

mod1 = scipy.optimize.minimize(fun = cobj, x0 =  initial_params, args = indpro_data['log_diff'], method='L-BFGS-B', bounds=bounds).x
# Output of results
print("Maximum of the conditional likelihood function:")
print(result_cond.x)
print('Second Model: ', mod1)

print("\nMaximum of the unconditional likelihood function:")
print(result_uncond.x)


#4 part
# Forecasting future values using estimated parameters of AR(7).

def ar7_forecast(params, initial_data, h=8):
    """
    Forecasting future values of logarithmic differences of INDPRO using the AR(7) model.
    Parameters:
    - `params`: estimated parameters of the AR(7) model (c, phi1, ..., phi7, sigma^2)
    - `initial_data`: initial data for the model (last 7 values of logarithmic differences)
    - `h`: number of periods for forecasting (default is 8 months)
    
    Returns:
    - `forecast`: array of forecasted values of logarithmic differences of INDPRO for h periods ahead.
    """
    # Extracting the estimated parameters
    c = params[0]
    phi = params[1:8]
    sigma2 = params[8]
    
    # Initial values for forecasting
    forecast = initial_data.copy()
    
    # Forecasting for h periods ahead.
    for i in range(h):
        # Calculating a new value based on previous values and model parameters
        new_value = c + np.dot(phi, forecast[-7:]) + np.random.normal(scale=np.sqrt(sigma2))
        
        # Adding the new value to the forecast
        forecast = np.append(forecast, new_value)
    
    return forecast

# Forecasting using the conditional model
conditional_forecast = ar7_forecast(result_cond.x, log_diffs[-7:])

# Forecasting using the unconditional model
unconditional_forecast = ar7_forecast(result_uncond.x, log_diffs[-7:])

# Printing the results
print("Forecast using the conditional AR(7) model:", conditional_forecast)
print("\nForecast using the unconditional AR(7) model:", unconditional_forecast)
