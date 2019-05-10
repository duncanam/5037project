######################################################################
#    ____            _           _     ____   ___  _____ 
#   |  _ \ _ __ ___ (_) ___  ___| |_  |___ \ / _ \|___ / 
#   | |_) | '__/ _ \| |/ _ \/ __| __|   __) | | | | |_ \ 
#   |  __/| | | (_) | |  __/ (__| |_   / __/| |_| |___) |
#   |_|   |_|  \___// |\___|\___|\__| |_____|\__\_\____/ 
#                 |__/                                   
#   
# Purpose: Main script for ASEN 5037 Project 2 Question 3
#
# Note: for best results, use an iPython interpreter. 
#
# Author: Duncan McGough
#
# Date Created:     5/9/19
# Date Edited:      5/10/19
#
######################################################################
# IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
#import cProfile
#from numba import jit
#from scipy.stats import norm
#from mpl_toolkits.mplot3d import Axes3D
#from skimage.measure import marching_cubes_lewiner as mcl
import sys
sys.path.insert(0, './generic/') # path to generic functions

# Import generic functions:
from read_data import read_dat
from importdata import dataimport

######################################################################
# PRINT WELCOME MESSAGE
print('\n'*100)
print('-'*60)
print('TURBULENCE PROJECT 2 Q1 SCRIPT')
print('Author: Duncan McGough')
print('-'*60)
print('\n')

######################################################################
# DEFINE VARIABLES
pi = np.pi
nx = np.array([256,129,256])
lx = np.array([2*pi, pi, 2*pi])
dx = lx/nx
datafolder = './project2data/'
filenames = [datafolder+'HIT_u.bin', datafolder+'HIT_v.bin',
            datafolder+'HIT_w.bin', datafolder+'HST_u.bin',
            datafolder+'HST_v.bin', datafolder+'HST_w.bin']
uvw = ['u','v','w']

######################################################################
# IMPORT THE DATA
[hit_u, hit_v, hit_w,
 hst_u, hst_v, hst_w] = dataimport(filenames, nx)

######################################################################
# CALCULATE U-VELOCITY AUTOCORRECTION AND SPATIAL INTEGRAL SCALE 
#   AND SPATIAL TAYLOR SCALE [3.1]
# Define autocorrelation function: 
def rho(r,up):
    # Make a u' array that is wrapped based on r:
    up_p1 = np.concatenate((up[r:,:,:],up[0:r,:,:]))
    # Return the autocorrelation:
    return np.mean(up*up_p1)/np.mean(up**2)

# Load the HIT u' data from Question 1:
hit_p = np.load('hit_up.npy')

# Housekeeping for next step:
inf = nx[0] # define infinity
r_inf = np.arange(inf) # Define vector
rho_i = np.zeros((inf)) # allocate memory 

# Now loop through infinity:
for i in range(inf):
    # Store rho() in vector for numeric integration:
    rho_i[i] = rho(i,hit_p) 

# Calculate the spatial integral scale:
llam = np.trapz(rho_i,r_inf)

# Calculate and evaluate derivatives of rho:
d2pdr2_0 = np.gradient(np.gradient(rho_i))[0]

# Calculate the spatial taylor scale:
lam2 = -2*d2pdr2_0**(-1)

# Print results:
print('Spatial Integral Scale: \n')
print(llam)
print('\n')

print('Spatial Taylor Scale: \n')
print(lam2)
print('\n')

# Plot results:
plt.figure(figsize=(8,6), dpi=160)
plt.plot(r_inf,rho_i,label='HIT u\' Autocorrelation')
plt.plot(r_inf, np.exp(-r_inf/llam),label='Exponential')
plt.plot(r_inf, np.exp(-(np.pi/4)*(r_inf/llam)**2),label='Gaussian')
plt.legend()

######################################################################
# JOINT PDFS [3.2]


######################################################################
# PRINT LINE END
print('\n')
print('-'*60)
print('\n')

######################################################################
# SHOW FIGURES
#plt.close('all')
plt.show()
