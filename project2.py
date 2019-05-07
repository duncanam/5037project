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
# Note: for best results, use an iPython interpreter. 
#
# Author: Duncan McGough
#
# Date Created:     4/11/19
# Date Edited:      5/2/19
#
######################################################################
# IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
#import cProfile
#from numba import jit
import sys
sys.path.insert(0, './generic/') # path to generic functions

# Import generic functions:
from read_data import read_dat
from importdata import dataimport

######################################################################
# PRINT WELCOME MESSAGE
print('\n'*100)
print('-'*60)
print('TURBULENCE PROJECT 2 SCRIPT')
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
# FIND MAX AND MIN VALUES FOR EACH COMPONENT AND PRINT THEM [1.1]
print('-'*60)
print('MIN AND MAX VALUES:\n')

max_hit_u = np.max(hit_u)
max_hit_v = np.max(hit_v)
max_hit_w = np.max(hit_w)
max_hst_u = np.max(hst_u)
max_hst_v = np.max(hst_v)
max_hst_w = np.max(hst_w)

print('Max HIT u:', max_hit_u)
print('Max HIT v:', max_hit_v)
print('Max HIT w:', max_hit_w)
print('Max HST u:', max_hst_u)
print('Max HST v:', max_hst_v)
print('Max HST w:', max_hst_w)
print('')

min_hit_u = np.min(hit_u)
min_hit_v = np.min(hit_v)
min_hit_w = np.min(hit_w)
min_hst_u = np.min(hst_u)
min_hst_v = np.min(hst_v)
min_hst_w = np.min(hst_w)

print('Min HIT u:', min_hit_u)
print('Min HIT v:', min_hit_v)
print('Min HIT w:', min_hit_w)
print('Min HST u:', min_hst_u)
print('Min HST v:', min_hst_v)
print('Min HST w:', min_hst_w)
print('')

######################################################################
# PLOT ISOSURFACES OF TURBULENCE DATA [1.2]
### Do HIT:
## Calculate triangular-based isosurfaces:
#verts1, faces1, _, _ = mcl(hit_u, max_hit_u/2)
#verts2, faces2, _, _ = mcl(hit_u, min_hit_u/2)
#                                                     
## Start a new figure and 3D subplot:
#fig1 = plt.figure(figsize=(12,12))
#ax1 = fig1.add_subplot(111,projection='3d')
#                                                     
## Plot the surfaces (LARGE CALCULATION):
#ax1.plot_trisurf(verts1[:, 0], verts1[:,1], faces1, 
#        verts1[:, 2], color=(1,0,0,1), lw=1)
#ax1.plot_trisurf(verts2[:, 0], verts2[:,1], faces2, 
#        verts2[:, 2], color=(0,0,1,1), lw=1)
#                                                      
### Now do HST:
## Calculate triangular-based isosurfaces:
#verts3, faces3, _, _ = mcl(hst_u, max_hst_u/2)
#verts4, faces4, _, _ = mcl(hst_u, min_hst_u/2)
#                                                     
## Start a new figure and 3D subplot:
#fig2 = plt.figure(figsize=(12,12))
#ax2 = fig2.add_subplot(111,projection='3d')
#                                                     
## Plot the surfaces (LARGE CALCULATION):
#ax2.plot_trisurf(verts3[:, 0], verts3[:,1], faces3, 
#        verts3[:, 2], color=(1,0,0,1), lw=1)
#ax2.plot_trisurf(verts4[:, 0], verts4[:,1], faces4, 
#        verts4[:, 2], color=(0,0,1,1), lw=1)

######################################################################
# SLICE DATA AND PLOT AS IMAGE [1.3]

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

######################################################################
# CALCULATE X-Z AVERAGES OF U V W [1.5]
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

######################################################################
# CALCULATE AND PLOT XY FIELDS OF FLUCTUATING COMPONENTS [1.6]
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

######################################################################
# PLOT SLICES OF FLUCTUATIONS MINUS FULL VOLUME AVERAGES [1.7]
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

######################################################################
# CALCULATE THE REYNOLDS STRESSES [1.8]
#hst_ui_xz = np.zeros((3,nx[1],3))
#for i in range(3):
#    for k in range(3):
#        hst_ui_xz[i,:,k] = hst_nxz[i]*hst_nxz[k]
#
#hst_uu_xz = hst_uxz*hst_uxz

# Combine the HST u'_i data into one data structure
hst_np = [hst_up, hst_vp, hst_wp]

# Allocate for this massive structure about to come
hst_uipuip = np.zeros((3,3,256,129,256))

# Multiply the u'_i together at every location
for i in range(3):
    for j in range(3):






######################################################################
# PRINT LINE END
print('\n')
print('-'*60)
print('\n')

######################################################################
# SHOW FIGURES
plt.close('all')
#plt.show()
