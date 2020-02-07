import numpy as np
import matplotlib.pyplot as plt
import random

def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2) / (2 * np.power(sigma, 2))) / np.sqrt( 2 * np.pi * np.power(sigma, 2) )

# Data
x = np.array([4.6, 6.0, 2.0, 5.8])
sigma = np.array([2.0, 1.5, 5.0, 1.0])

mu_array = np.linspace( 0, 10 , 1000 )
A = max(mu_array) - min(mu_array)

def ver( x_arr , mu_arr , sigma_arr ):
    ver_arr = []
    for m in mu_arr:
        ver_arr.append( np.prod( gaussian( x_arr, m, sigma_arr ) ) )
    return ver_arr

h = mu_array[1] - mu_array[0]
norm = sum( (ver( x , mu_array , sigma ) / A) * h )
posterior = ver( x , mu_array , sigma ) / ( norm * A )

# L function
log_posterior = np.log( posterior )

# Derivative
der_L = (log_posterior[2:-1] - log_posterior[0:-3])/(2*h)

# Find mu_0
pos_mu0 = np.argmin(abs(der_L))
mu_0 = mu_array[ pos_mu0 ]
mu_0

# Find sigma
second_der_L = (log_posterior[2:-1] - 2*log_posterior[1:-2] + log_posterior[0:-3])/(h**2)
devst_mu = 1 / np.sqrt(-second_der_L[ pos_mu0 ])
devst_mu

plt.figure()
plt.plot( mu_array , posterior )
# plt.axvline( x = mu_array[pos_mu0] )
plt.title( '$\\mu$ = ' + str(round(mu_0,5)) + r'$\pm$' + str(round(devst_mu,5)) )
plt.savefig('mean.png')