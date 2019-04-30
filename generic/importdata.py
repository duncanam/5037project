######################################################################
#        ___                            _     ____        _        
#       |_ _|_ __ ___  _ __   ___  _ __| |_  |  _ \  __ _| |_ __ _ 
#        | || '_ ` _ \| '_ \ / _ \| '__| __| | | | |/ _` | __/ _` |
#        | || | | | | | |_) | (_) | |  | |_  | |_| | (_| | || (_| |
#       |___|_| |_| |_| .__/ \___/|_|   \__| |____/ \__,_|\__\__,_|
#                     |_|                                          
#
# Purpose: parse and import all data from HIT and HST data
#
# Author: Duncan McGough
#
# Date Created:     4/11/19
# Date Edited:      4/11/19
#
######################################################################
# IMPORT PACKAGES
import numpy as np
from read_data import read_dat

######################################################################
# READ IN DATA AND PARSE
def dataimport(filenames, nx):
    # Allocate memory:
    u = np.zeros((nx[0],nx[1],nx[2],len(filenames)))

    # Step through each file:
    for i in range(len(filenames)):
        # Read in the file:
        tmp = read_dat(filenames[i])                   

        # Shift the file such that it can be reshaped:
        tmp0 = tmp[1:(nx[0]*nx[1]*nx[2]+1)]

        # Reshape into a block of data:
        u[:,:,:,i] = np.reshape(tmp0, (nx[0],nx[1],nx[2]))
    
    # Return each u,v,w for HIT and HST respectively:
    return [u[:,:,:,0], u[:,:,:,1], u[:,:,:,2],
            u[:,:,:,3], u[:,:,:,4], u[:,:,:,5]]




