######################################################################
#    ____  _   ____  
#   |  _ \/ | |___ \ 
#   | |_) | |   __) |
#   |  __/| |_ / __/ 
#   |_|   |_(_)_____|
#   
# Author: Duncan McGough
#
# Created: 4/11/19
# Edited: 4/29/19
#
######################################################################
# IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes_lewiner as mcl

######################################################################
# PLOT ISOSURFACES OF TURBULENCE DATA [1.2]
def p12(data, data_maxmin):
    # Define our imported data:
    hit_u       = data[0]
    hit_v       = data[1]
    hit_w       = data[2]
    hst_u       = data[3]
    hst_v       = data[4]
    hst_w       = data[5]

    max_hit_u   = data_maxmin[0]  
    max_hit_v   = data_maxmin[1]  
    max_hit_w   = data_maxmin[2]  
    max_hst_u   = data_maxmin[3]  
    max_hst_v   = data_maxmin[4]  
    max_hst_w   = data_maxmin[5]  
    min_hit_u   = data_maxmin[6]  
    min_hit_v   = data_maxmin[7]  
    min_hit_w   = data_maxmin[8]  
    min_hst_u   = data_maxmin[9]  
    min_hst_v   = data_maxmin[10]  
    min_hst_w   = data_maxmin[11]  

    ## Do HIT:
    # Calculate triangular-based isosurfaces:
    verts1, faces1, _, _ = mcl(hit_u, max_hit_u/2)
    verts2, faces2, _, _ = mcl(hit_u, min_hit_u/2)
                                                         
    # Start a new figure and 3D subplot:
    fig1 = plt.figure(figsize=(12,12))
    ax1 = fig1.add_subplot(111,projection='3d')
                                                         
    # Plot the surfaces (LARGE CALCULATION):
    ax1.plot_trisurf(verts1[:, 0], verts1[:,1], faces1, 
            verts1[:, 2], color=(1,0,0,1), lw=1)
    ax1.plot_trisurf(verts2[:, 0], verts2[:,1], faces2, 
            verts2[:, 2], color=(0,0,1,1), lw=1)

    ## Now do HST:
    # Calculate triangular-based isosurfaces:
    verts3, faces3, _, _ = mcl(hst_u, max_hst_u/2)
    verts4, faces4, _, _ = mcl(hst_u, min_hst_u/2)
                                                         
    # Start a new figure and 3D subplot:
    fig2 = plt.figure(figsize=(12,12))
    ax2 = fig2.add_subplot(111,projection='3d')
                                                         
    # Plot the surfaces (LARGE CALCULATION):
    ax2.plot_trisurf(verts3[:, 0], verts3[:,1], faces3, 
            verts3[:, 2], color=(1,0,0,1), lw=1)
    ax2.plot_trisurf(verts4[:, 0], verts4[:,1], faces4, 
            verts4[:, 2], color=(0,0,1,1), lw=1)

    # Show the data:
    plt.show()

