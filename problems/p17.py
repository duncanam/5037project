######################################################################
#    ____  _  _____ 
#   |  _ \/ ||___  |
#   | |_) | |   / / 
#   |  __/| |_ / /  
#   |_|   |_(_)_/   
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
# PLOT SLICES OF FLUCTUATIONS MINUS FULL VOLUME AVERAGES [1.7]
def p17(data, nx):

     # Define our imported data:
     hit_u       = data[0]
     hit_v       = data[1]
     hit_w       = data[2]
     hst_u       = data[3]
     hst_v       = data[4]
     hst_w       = data[5]

     # Calculate Full Volume Averages:
     hit_u_xyz = 1/(nx[0]*nx[1]*nx[2])*np.sum(hit_u)
     hit_v_xyz = 1/(nx[0]*nx[1]*nx[2])*np.sum(hit_v)
     hit_w_xyz = 1/(nx[0]*nx[1]*nx[2])*np.sum(hit_w)
     
     # Find the fluctuations now:
     hit_upf = hit_u - hit_u_xyz
     hit_vpf = hit_v - hit_v_xyz
     hit_wpf = hit_w - hit_w_xyz
     
     # Set some imshow settings:
     slccmap = 'jet' # set the colormap 
     fsize = 8 # fontsize
     tsize = 6 # ticksize
     
     # Find the min and max of these values:
     hit_upa_min = np.min(hit_upf)
     hit_vpa_min = np.min(hit_vpf)
     hit_wpa_min = np.min(hit_wpf)
     hitpf_min = np.min([hit_upa_min, hit_vpa_min, hit_wpa_min])
     
     hit_upa_max = np.max(hit_upf)
     hit_vpa_max = np.max(hit_vpf)
     hit_wpa_max = np.max(hit_wpf)
     hitpf_max = np.max([hit_upa_max, hit_vpa_max, hit_wpa_max])
     
     # Now we plot the results:
     # Define the subplot figure:
     hitpslc, hitpaxs = plt.subplots(3,3, figsize=(10,6), dpi=160)
     hitpslc.suptitle('Velocity slice: u-u_xyz')
     
     hitpaxs[0,0].imshow(hit_upf[:,:,0], vmin=hitpf_min, vmax=hitpf_max, cmap=slccmap)
     hitpaxs[0,0].set_title('HIT u\' @ k=1',fontsize=fsize)
     hitpaxs[1,0].imshow(hit_vpf[:,:,0], vmin=hitpf_min, vmax=hitpf_max, cmap=slccmap)
     hitpaxs[1,0].set_title('HIT v\' @ k=1',fontsize=fsize)
     hitpaxs[2,0].imshow(hit_wpf[:,:,0], vmin=hitpf_min, vmax=hitpf_max, cmap=slccmap)
     hitpaxs[2,0].set_title('HIT w\' @ k=1',fontsize=fsize)
     
     # Plot HIT k=128:
     hitpaxs[0,1].imshow(hit_upf[:,:,127], vmin=hitpf_min, vmax=hitpf_max, cmap=slccmap)
     hitpaxs[0,1].set_title('HIT u\' @ k=128',fontsize=fsize)
     hitpaxs[1,1].imshow(hit_vpf[:,:,127], vmin=hitpf_min, vmax=hitpf_max, cmap=slccmap)
     hitpaxs[1,1].set_title('HIT v\' @ k=128',fontsize=fsize)
     hitpaxs[2,1].imshow(hit_wpf[:,:,127], vmin=hitpf_min, vmax=hitpf_max, cmap=slccmap)
     hitpaxs[2,1].set_title('HIT w\' @ k=128',fontsize=fsize)
     
     # Plot HIT k=256:
     hitpaxs[0,2].imshow(hit_upf[:,:,255], vmin=hitpf_min, vmax=hitpf_max, cmap=slccmap)
     hitpaxs[0,2].set_title('HIT u\' @ k=256',fontsize=fsize)
     hitpaxs[1,2].imshow(hit_vpf[:,:,255], vmin=hitpf_min, vmax=hitpf_max, cmap=slccmap)
     hitpaxs[1,2].set_title('HIT v\' @ k=256',fontsize=fsize)
     im = hitpaxs[2,2].imshow(hit_wpf[:,:,255], vmin=hitpf_min, vmax=hitpf_max, cmap=slccmap)
     hitpaxs[2,2].set_title('HIT w\' @ k=256',fontsize=fsize)
     
     # Set the colorbar:
     #cax = hitpslc.add_axes([0.85, 0.05, 0.01, 0.9])
     cax = hitpslc.add_axes([0.03, 0.03, 0.93, 0.02])
     hitpslc.colorbar(im,cax=cax, orientation='horizontal')
     
     for i in range(3):
         for j in range(3):
             plt.sca(hitpaxs[i,j])
             plt.xticks(fontsize=tsize)
             plt.yticks(fontsize=tsize)
