import numpy as np
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

import pandas as pd
from numpy.linalg import solve
import numpy as np

#Read Data
df = df = pd.read_csv('C:/Users/Dell/Desktop/cours_S2/comput_tools/current_dataset.csv')
#Select INDPRO
INDPRO = df['INDPRO']
#Drop first Row
INDPRO = INDPRO.drop(index=0)
#transform INDPRO using log differences
INDPRO = np.log(INDPRO).diff()

from scipy.stats import norm
from scipy.stats import multivariate_normal

def lagged_matrix(Y, max_lag=7):
    n = len(Y)
    lagged_matrix = np.full((n, max_lag), np.nan)    
    # Fill each column with the appropriately lagged data
    for lag in range(1, max_lag + 1):
        lagged_matrix[lag:, lag - 1] = Y[:-lag]
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
    

## Example
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

## Unconditional - define the negative loglikelihood

import numpy as np
import pandas as pd
import scipy.optimize

# Read Data
df = pd.read_csv('C:/Users/Dell/Desktop/cours_S2/comput_tools/current_dataset.csv')

# Clean the data: Select INDPRO, drop first row, and transform using log differences
INDPRO = df['INDPRO'].drop(index=0).apply(np.log).diff().dropna().values

# Define function to create lagged matrix
def lagged_matrix(Y, max_lag=7):
    n = len(Y)
    lagged_matrix = np.full((n, max_lag), np.nan)    
    # Fill each column with the appropriately lagged data
    for lag in range(1, max_lag + 1):
        lagged_matrix[lag:, lag - 1] = Y[:-lag]
    return lagged_matrix

# Create lagged matrix for INDPRO
X = lagged_matrix(INDPRO, 7)

# Extract lagged data and add intercept column
Xf = np.hstack((np.ones((X.shape[0] - 7, 1)), X[7:, :]))
yf = INDPRO[7:]

# Function to calculate conditional log-likelihood
def cond_loglikelihood_ar7(params, y):
    beta = params[:-1]
    sigma2 = params[-1]
    residuals = y - Xf @ beta
    log_likelihood = -0.5 * (np.log(2 * np.pi * sigma2) + (residuals ** 2 / sigma2)).sum()
    return -log_likelihood

# Define objective function for optimization
def cobj(params, y): 
    return -cond_loglikelihood_ar7(params, y)

# Initial parameters (beta and sigma2)
beta_initial = np.zeros(Xf.shape[1])
sigma2_initial = np.var(yf)
params_initial = np.hstack((beta_initial, sigma2_initial))

# Perform optimization
results = scipy.optimize.minimize(cobj, params_initial, args=(yf,), method='L-BFGS-B')

# Display results
print("Estimated parameters:", results.x)
