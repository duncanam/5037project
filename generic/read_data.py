######################################################################
#        ____                _   ____        _        
#       |  _ \ ___  __ _  __| | |  _ \  __ _| |_ __ _ 
#       | |_) / _ \/ _` |/ _` | | | | |/ _` | __/ _` |
#       |  _ <  __/ (_| | (_| | | |_| | (_| | || (_| |
#       |_| \_\___|\__,_|\__,_| |____/ \__,_|\__\__,_|
#
# Purpose: Read in binary HIT and HST data for Project 2
#
# Author: Duncan McGough, adapted from Peter Hamlington's script
#        
# Creation Date:    4/11/19
# Edited Date:      4/11/19
#
######################################################################
# IMPORT PACKAGES
import numpy as np

######################################################################
def read_dat(fn):
    # Open the file:
    fid = open(fn, 'rb')

    # Read in the data:
    out = np.fromfile(fid, np.single)

    # Close the file:
    fid.close()

    # Return the data:
    return out




