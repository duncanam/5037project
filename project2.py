######################################################################
#        ____            _           _     ____
#       |  _ \ _ __ ___ (_) ___  ___| |_  |___ \
#       | |_) | '__/ _ \| |/ _ \/ __| __|   __) |
#       |  __/| | | (_) | |  __/ (__| |_   / __/
#       |_|   |_|  \___// |\___|\___|\__| |_____|
#                     |__/
#
# Purpose: Main script for ASEN 5037 Project 2
#
# Author: Duncan McGough
#
# Date Created:     4/11/19
# Date Edited:      4/29/19
#
######################################################################
# PRINT WELCOME MESSAGE
print('\n'*100)
print('-'*60)
print('TURBULENCE PROJECT 2 SCRIPT')
print('Author: Duncan McGough')
print('-'*60)
print('\n')

######################################################################
# ASK THE USER FOR PREFERENCES
# This option enables or disables live-script preferences, otherwise
#   change the options manually and hardcoded after:
userinput = False

# Manually hardcoded values:
plotsurf = False
plotslc = False
plot15 = False
plot16 = False
plot17 = False

# Live-script preference menu [don't edit]:
if userinput == True:
    isoinput = input('Would you like to calculate and plot HIT/HST isosurfaces? [y/n] ')
    slcinput = input('Would you like to plot the velocity slices? [y/n] ')
    p15input = input('Would you like to plot the xz averages as a function of y? [y/n]')
    p16input = input('Would you like to plot the velocity slices mius the xz averages? [y/n]')
    
    print('')
    
    if isoinput == 'y':
        plotsurf = True
    else:
        plotsurf = False

    if slcinput == 'y':
        plotslc = True
    else:
        plotslc = False

    if p15input == 'y':
        plot15 = True
    else:
        plot15 = False

    if p16input == 'y':
        plot16 = True
    else:
        plot16 = False
    
######################################################################
# IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, './generic/') # path to generic functions
sys.path.insert(0, './problems/') # path to problems
import time

# Import generic functions:
from read_data import read_dat
from importdata import dataimport

# Import each problem:
from p11 import p11
from p12 import p12
from p13 import p13
from p15 import p15
from p16 import p16
from p17 import p17

######################################################################
# START TIMER
ti = time.perf_counter()

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
data = dataimport(filenames, nx)

######################################################################
# FIND MAX AND MIN VALUES FOR EACH COMPONENT AND PRINT THEM [1.1]
data_maxmin = p11(data)

######################################################################
# PLOT ISOSURFACES OF TURBULENCE DATA [1.2]
if plotsurf == True:
    p12(data, data_maxmin)

######################################################################
# SLICE DATA AND PLOT AS IMAGE [1.3]
if plotslc == True: 
    p13(data, data_maxmin)

######################################################################
# CALCULATE X-Z AVERAGES OF U V W [1.5]
if plot15 == True:
    [hst_uxz, hst_vxz, hst_wxz] = p15(data, nx, lx)

######################################################################
# CALCULATE AND PLOT XY FIELDS OF FLUCTUATING COMPONENTS [1.6]
if plot16 == True:
    p16(data, nx, hst_uxz, hst_vxz, hst_wxz)

######################################################################
# PLOT SLICES OF FLUCTUATIONS MINUS FULL VOLUME AVERAGES [1.7]
if plot17 == True:
    p17(data,nx)
    
######################################################################
# CALCULATE THE REYNOLDS STRESSES [1.8]



######################################################################
# END TIMER AND PRINT RESULT
tf = time.perf_counter()
print('\n')
print('Time elapsed: ', tf-ti)
print('-'*60)
print('\n')

######################################################################
# SHOW FIGURES
if plotslc == True:
    plt.show()
plt.show()

