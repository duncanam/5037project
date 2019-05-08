######################################################################
#    ____            _           _     ____   ___  ____  
#   |  _ \ _ __ ___ (_) ___  ___| |_  |___ \ / _ \|___ \ 
#   | |_) | '__/ _ \| |/ _ \/ __| __|   __) | | |   __) |
#   |  __/| | | (_) | |  __/ (__| |_   / __/| |_|  / __/
#   |_|   |_|  \___// |\___|\___|\__| |_____|\__\_|_____|
#                 |__/                               
#   
# Purpose: Main script for ASEN 5037 Project 2 Question 2
#
# Note: for best results, use an iPython interpreter. 
#
# Author: Duncan McGough
#
# Date Created:     5/7/19
# Date Edited:      5/7/19
#
######################################################################
# IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
#import cProfile
#from numba import jit
from scipy.stats import norm
import sys
sys.path.insert(0, './generic/') # path to generic functions

# Import generic functions:
from read_data import read_dat
from importdata import dataimport

######################################################################
# PRINT WELCOME MESSAGE
print('\n'*100)
print('-'*60)
print('TURBULENCE PROJECT 2 Q2 SCRIPT')
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

# Pack for iterating:
hit_n = [hit_u, hit_v, hit_w]

######################################################################
# CALCULATE Aij AND <Aij>xyz: [2.1]
# Preallocate:
A = np.ones((3,3,nx[0],nx[1],nx[2]))
Ap = np.ones((3,3,nx[0],nx[1],nx[2]))
Abar = np.ones((3,3))

# NOTE: numpy's gradient() function has axis definitions that are 
#       counterinuitive. Axis 0 is z, axis 1 is y, and 
#       axis 2 is x. 

for i in range(3): # iterate u_i
    for j in range(3): # iterate x_i
        A[i,j,:,:,:] = np.gradient(hit_n[i], dx[j], axis=2-j)
        Abar[i,j] = np.mean(A[i,j,:,:,:])
        Ap[i,j] = A[i,j,:,:,:] - Abar[i,j] # for 2.2

print('<A>xyz = \n')
print(Abar)
print('\n')

######################################################################
# FIND A' AND PLOT PDFs [2.2]
# Ap = A' and found in section above. 

# Reshape data:
A11p_vec = np.reshape(Ap[0,0,:,:,:], (nx[0]*nx[1]*nx[2]))
A12p_vec = np.reshape(Ap[0,1,:,:,:], (nx[0]*nx[1]*nx[2]))

# Define bins: 
nbins=100

# Take histogram of this (density True for PDF): 
A11p_hist = np.histogram(A11p_vec, bins=nbins, density=True)
A12p_hist = np.histogram(A12p_vec, bins=nbins, density=True)

# Define Gaussian function:
def pdf(x,mu,sigma):
    return 1/np.sqrt(2*pi)*1/sigma*np.exp(-(x)**2/(2*sigma**2))

# Find Standard Deviation:
A11p_sigma = np.std(A11p_vec)
A12p_sigma = np.std(A12p_vec)

# Find mean:
A11p_mean = np.mean(A11p_vec)
A12p_mean = np.mean(A12p_vec)

# Get the Gaussian profile (these are normalized by the hist already):
A11p_g = pdf(A11p_hist[1][1:], A11p_mean, A11p_sigma)
A12p_g = pdf(A12p_hist[1][1:], A12p_mean, A12p_sigma)

# Plot the data:
plt.figure(figsize=(10,6), dpi=160)
plt.suptitle('HIT A\'ij PDFs')
legsize=6

plt.subplot(121)
plt.plot(A11p_hist[1][1:], A11p_hist[0],label='A11') 
plt.plot(A11p_hist[1][1:], A11p_g, color='red',label='Gaussian')
plt.legend(prop={'size': legsize})

plt.subplot(122)
plt.plot(A12p_hist[1][1:], A12p_hist[0],label='A11') 
plt.plot(A12p_hist[1][1:], A12p_g, color='red',label='Gaussian')
plt.legend(prop={'size': legsize})

######################################################################
# FIND SKEWNESS AND KURTOSIS OF A'11 [2.3]
# Define for clarity:
A11p = Ap[0,0,:,:,:]

# Find skew and kurtosis:
A11p_s = np.mean(A11p**3)/A11p_sigma**3
A11p_k = np.mean(A11p**4)/A11p_sigma**4

# print data:
print('A\'11 S: \n')
print(A11p_s)
print('\n')
print('A\'11 K: \n')
print(A11p_k)
print('\n')

# Now do the same for the Gaussian data:
# s=0 and k=3 for gaussian. 

######################################################################
# FLUCTUATING STRAIN RATE FIELD AND PSEUDO ENERGY DISSIPATION [2.4] 
# Allocate:
Sp = np.zeros((3,3,nx[0],nx[1],nx[2]))
enu = np.zeros((nx[0],nx[1],nx[2]))

for i in range(3):
    for j in range(3):
        # Calculate S'ij=0.5(A'ij+A'ji):
        Sp[i,j,:,:,:] = 0.5*(Ap[i,j,:,:,:]+Ap[j,i,:,:,:])

        # Pseudo energy dissipation field:
        enu[:,:,:] = enu + 2*Sp[i,j,:,:,:]*Sp[i,j,:,:,:]

# Now volume-averaged:
enu_mean = np.mean(enu)

# Print results:
print('Volume-Averaged Epsilon/Nu: \n')
print(enu_mean)
print('\n')

######################################################################
# 2D X-Y PLOT OF  eps/<eps>xyz [2.5]
# Colormap max and min:
slcmin = np.min(enu)
slcmax = np.max(enu)
slcmap = 'jet'

# Plot:
plt.plot(figsize=(10,6), dpi=160)
plt.title('HIT Epsilon/<Epsilon>_xyz at k=128')
plt.imshow(enu[:,:,127]/enu_mean, vmin=slcmin, vmax=slcmax, cmap=slcmap)

######################################################################
# PRINT LINE END
print('\n')
print('-'*60)
print('\n')

######################################################################
# SHOW FIGURES
#plt.close('all')
plt.show()
