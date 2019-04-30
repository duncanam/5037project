######################################################################
#    ____  _   _____ 
#   |  _ \/ | |___ / 
#   | |_) | |   |_ \ 
#   |  __/| |_ ___) |
#   |_|   |_(_)____/ 
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

######################################################################
# SLICE DATA AND PLOT AS IMAGE [1.3]
def p13(data, data_maxmin):
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

    # Define the colormap max and mins:
    slcmin = np.min([min_hit_u, min_hit_v, min_hit_w, min_hst_u, 
                    min_hst_v, min_hst_w])
    slcmax = np.max([max_hit_u, max_hit_v, max_hit_w, max_hst_u, 
                    max_hst_v, max_hst_w])
                                                                                    
    slccmap = 'jet' # set the colormap 
    fsize = 8 # fontsize
    tsize = 6 # ticksize
    
    ## Define the subplot figure:
    hitslc, hitaxs = plt.subplots(3,3, figsize=(10,6), dpi=160)
    hitslc.suptitle('HIT Velocity Slices: u,v,w')
                                                                                    
    
    # Plot HIT k=1:
    hitaxs[0,0].imshow(hit_u[:,:,0], vmin=slcmin, vmax=slcmax, cmap=slccmap)
    hitaxs[0,0].set_title('HIT u @ k=1',fontsize=fsize)
    hitaxs[1,0].imshow(hit_v[:,:,0], vmin=slcmin, vmax=slcmax, cmap=slccmap)
    hitaxs[1,0].set_title('HIT v @ k=1',fontsize=fsize)
    hitaxs[2,0].imshow(hit_w[:,:,0], vmin=slcmin, vmax=slcmax, cmap=slccmap)
    hitaxs[2,0].set_title('HIT w @ k=1',fontsize=fsize)
    
    # Plot HIT k=128:
    hitaxs[0,1].imshow(hit_u[:,:,127], vmin=slcmin, vmax=slcmax, cmap=slccmap)
    hitaxs[0,1].set_title('HIT u @ k=128',fontsize=fsize)
    hitaxs[1,1].imshow(hit_v[:,:,127], vmin=slcmin, vmax=slcmax, cmap=slccmap)
    hitaxs[1,1].set_title('HIT v @ k=128',fontsize=fsize)
    hitaxs[2,1].imshow(hit_w[:,:,127], vmin=slcmin, vmax=slcmax, cmap=slccmap)
    hitaxs[2,1].set_title('HIT w @ k=128',fontsize=fsize)
    
    # Plot HIT k=256:
    hitaxs[0,2].imshow(hit_u[:,:,255], vmin=slcmin, vmax=slcmax, cmap=slccmap)
    hitaxs[0,2].set_title('HIT u @ k=256',fontsize=fsize)
    hitaxs[1,2].imshow(hit_v[:,:,255], vmin=slcmin, vmax=slcmax, cmap=slccmap)
    hitaxs[1,2].set_title('HIT v @ k=256',fontsize=fsize)
    im = hitaxs[2,2].imshow(hit_w[:,:,255], vmin=slcmin, vmax=slcmax, cmap=slccmap)
    hitaxs[2,2].set_title('HIT w @ k=256',fontsize=fsize)
                                                                                    
    # Set the colorbar:
    cax = hitslc.add_axes([0.03, 0.03, 0.93, 0.02])
    hitslc.colorbar(im,cax=cax, orientation='horizontal')
    
    # Define the subplot figure:
    hstslc, hstaxs = plt.subplots(3,3, figsize=(10,6), dpi=160)
    hstslc.suptitle('HST Velocity Slices: u,v,w')
    
    # Plot HST k=1:
    hstaxs[0,0].imshow(hst_u[:,:,0], vmin=slcmin, vmax=slcmax, cmap=slccmap)
    hstaxs[0,0].set_title('HST u @ k=1',fontsize=fsize)
    hstaxs[1,0].imshow(hst_v[:,:,0], vmin=slcmin, vmax=slcmax, cmap=slccmap)
    hstaxs[1,0].set_title('HST v @ k=1',fontsize=fsize)
    hstaxs[2,0].imshow(hst_w[:,:,0], vmin=slcmin, vmax=slcmax, cmap=slccmap)
    hstaxs[2,0].set_title('HST w @ k=1',fontsize=fsize)
    
    # Plot HST k=128:
    hstaxs[0,1].imshow(hst_u[:,:,127], vmin=slcmin, vmax=slcmax, cmap=slccmap)
    hstaxs[0,1].set_title('HST u @ k=128',fontsize=fsize)
    hstaxs[1,1].imshow(hst_v[:,:,127], vmin=slcmin, vmax=slcmax, cmap=slccmap)
    hstaxs[1,1].set_title('HST v @ k=128',fontsize=fsize)
    hstaxs[2,1].imshow(hst_w[:,:,127], vmin=slcmin, vmax=slcmax, cmap=slccmap)
    hstaxs[2,1].set_title('HST w @ k=128',fontsize=fsize)
    
    # Plot HST k=256:
    hstaxs[0,2].imshow(hst_u[:,:,255], vmin=slcmin, vmax=slcmax, cmap=slccmap)
    hstaxs[0,2].set_title('HST u @ k=256',fontsize=fsize)
    hstaxs[1,2].imshow(hst_v[:,:,255], vmin=slcmin, vmax=slcmax, cmap=slccmap)
    hstaxs[1,2].set_title('HST v @ k=256',fontsize=fsize)
    hstaxs[2,2].imshow(hst_w[:,:,255], vmin=slcmin, vmax=slcmax, cmap=slccmap)
    hstaxs[2,2].set_title('HST w @ k=256',fontsize=fsize)
    
    # Set the colorbar:
    cax = hstslc.add_axes([0.03, 0.03, 0.93, 0.02])
    hstslc.colorbar(im,cax=cax, orientation='horizontal')
                                                                                    
    # Set tick font size for all:
    for i in range(3):
        for j in range(3):
            plt.sca(hitaxs[i,j])
            plt.xticks(fontsize=tsize)
            plt.yticks(fontsize=tsize)
    
            plt.sca(hstaxs[i,j])
            plt.xticks(fontsize=tsize)
            plt.yticks(fontsize=tsize)
    
    # Show the plot:
    plt.show()
