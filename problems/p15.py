######################################################################
#    ____  _   ____  
#   |  _ \/ | | ___| 
#   | |_) | | |___ \ 
#   |  __/| |_ ___) |
#   |_|   |_(_)____/ 
#   
# Author: Duncan McGough
#
# Created: 4/11/19
# Edited: 4./29/19
#
######################################################################
# IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt

######################################################################
# CALCULATE X-Z AVERAGES OF U V W [1.5]
def p15(data, nx, lx):

    # Define our imported data:
    hit_u       = data[0]
    hit_v       = data[1]
    hit_w       = data[2]
    hst_u       = data[3]
    hst_v       = data[4]
    hst_w       = data[5]
                                    
    # Average HIT:
    hit_uxz = 1/(nx[0]*nx[2])*np.sum(hit_u, axis=(0,2))
    hit_vxz = 1/(nx[0]*nx[2])*np.sum(hit_v, axis=(0,2))
    hit_wxz = 1/(nx[0]*nx[2])*np.sum(hit_w, axis=(0,2))
    
    # Average HST:
    hst_uxz = 1/(nx[0]*nx[2])*np.sum(hst_u, axis=(0,2))
    hst_vxz = 1/(nx[0]*nx[2])*np.sum(hst_v, axis=(0,2))
    hst_wxz = 1/(nx[0]*nx[2])*np.sum(hst_w, axis=(0,2))
    
    # Plot as a function of y:
    plt.figure(figsize=(10,6), dpi=160)
    
    plt.subplot(121)
    plt.plot(np.linspace(0,lx[1],nx[1]), hit_uxz, label='uxz Average')
    plt.plot(np.linspace(0,lx[1],nx[1]), hit_vxz, label='vxz Average')
    plt.plot(np.linspace(0,lx[1],nx[1]), hit_wxz, label='wxz Average')
    plt.title('HIT')
    plt.xlabel('y')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(np.linspace(0,lx[1],nx[1]), hst_uxz, label='uxz Average')
    plt.plot(np.linspace(0,lx[1],nx[1]), hst_vxz, label='vxz Average')
    plt.plot(np.linspace(0,lx[1],nx[1]), hst_wxz, label='wxz Average')
    plt.title('HST')
    plt.xlabel('y')
    plt.legend()

    return [hst_uxz, hst_vxz, hst_wxz] 
