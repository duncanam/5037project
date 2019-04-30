######################################################################
#    ____  _   __   
#   |  _ \/ | / /_  
#   | |_) | || '_ \ 
#   |  __/| || (_) |
#   |_|   |_(_)___/ 
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
# CALCULATE AND PLOT XY FIELDS OF FLUCTUATING COMPONENTS [1.6]
def p16(data, nx, hst_uxz, hst_vxz, hst_wxz):

    # Define our imported data:
    hit_u       = data[0]
    hit_v       = data[1]
    hit_w       = data[2]
    hst_u       = data[3]
    hst_v       = data[4]
    hst_w       = data[5]
                                    
    # Allocate for memory:
    hst_up = np.zeros((nx[0],nx[1],nx[2]))
    hst_vp = np.zeros((nx[0],nx[1],nx[2]))
    hst_wp = np.zeros((nx[0],nx[1],nx[2]))
    
    # Calculate each quantity:
    for j in range(nx[1]):
        hst_up[:,j,:] = hst_u[:,j,:] - hst_uxz[j]
        hst_vp[:,j,:] = hst_v[:,j,:] - hst_vxz[j]
        hst_wp[:,j,:] = hst_w[:,j,:] - hst_wxz[j]
    
    # Set some imshow settings:
    slccmap = 'jet' # set the colormap 
    fsize = 8 # fontsize
    tsize = 6 # ticksize
    
    # Find the min and max of these values:
    hst_up_min = np.min(hst_up)
    hst_vp_min = np.min(hst_vp)
    hst_wp_min = np.min(hst_wp)
    hstp_min = np.min([hst_up_min, hst_vp_min, hst_wp_min])
    
    hst_up_max = np.max(hst_up)
    hst_vp_max = np.max(hst_vp)
    hst_wp_max = np.max(hst_wp)
    hstp_max = np.max([hst_up_max, hst_vp_max, hst_wp_max])
    
    # Now we plot the results:
    # Define the subplot figure:
    hstpslc, hstpaxs = plt.subplots(3,3, figsize=(10,6), dpi=160)
    hstpslc.suptitle('Velocity slice: u-u_xz')
    
    hstpaxs[0,0].imshow(hst_up[:,:,0], vmin=hstp_min, vmax=hstp_max, cmap=slccmap)
    hstpaxs[0,0].set_title('HST u\' @ k=1',fontsize=fsize)
    hstpaxs[1,0].imshow(hst_vp[:,:,0], vmin=hstp_min, vmax=hstp_max, cmap=slccmap)
    hstpaxs[1,0].set_title('HST v\' @ k=1',fontsize=fsize)
    hstpaxs[2,0].imshow(hst_wp[:,:,0], vmin=hstp_min, vmax=hstp_max, cmap=slccmap)
    hstpaxs[2,0].set_title('HST w\' @ k=1',fontsize=fsize)
    
    # Plot HST k=128:
    hstpaxs[0,1].imshow(hst_up[:,:,127], vmin=hstp_min, vmax=hstp_max, cmap=slccmap)
    hstpaxs[0,1].set_title('HST u\' @ k=128',fontsize=fsize)
    hstpaxs[1,1].imshow(hst_vp[:,:,127], vmin=hstp_min, vmax=hstp_max, cmap=slccmap)
    hstpaxs[1,1].set_title('HST v\' @ k=128',fontsize=fsize)
    hstpaxs[2,1].imshow(hst_wp[:,:,127], vmin=hstp_min, vmax=hstp_max, cmap=slccmap)
    hstpaxs[2,1].set_title('HST w\' @ k=128',fontsize=fsize)
    
    # Plot HST k=256:
    hstpaxs[0,2].imshow(hst_up[:,:,255], vmin=hstp_min, vmax=hstp_max, cmap=slccmap)
    hstpaxs[0,2].set_title('HST u\' @ k=256',fontsize=fsize)
    hstpaxs[1,2].imshow(hst_vp[:,:,255], vmin=hstp_min, vmax=hstp_max, cmap=slccmap)
    hstpaxs[1,2].set_title('HST v\' @ k=256',fontsize=fsize)
    im = hstpaxs[2,2].imshow(hst_wp[:,:,255], vmin=hstp_min, vmax=hstp_max, cmap=slccmap)
    hstpaxs[2,2].set_title('HST w\' @ k=256',fontsize=fsize)
    
    # Set the colorbar:
    #cax = hstpslc.add_axes([0.85, 0.05, 0.01, 0.9])
    cax = hstpslc.add_axes([0.03, 0.03, 0.93, 0.02])
    hstpslc.colorbar(im,cax=cax, orientation='horizontal')
    
    for i in range(3):
        for j in range(3):
            plt.sca(hstpaxs[i,j])
            plt.xticks(fontsize=tsize)
            plt.yticks(fontsize=tsize)

