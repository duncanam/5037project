######################################################################
#    ____  _   _ 
#   |  _ \/ | / |
#   | |_) | | | |
#   |  __/| |_| |
#   |_|   |_(_)_|
#   
# Author: Duncan McGough 
#
# Created 4/11/19
# Edited: 4/29/19
#
######################################################################
# IMPORT PACKAGES
import numpy as np

######################################################################
# FIND MAX AND MIN VALUES FOR EACH COMPONENT AND PRINT THEM [1.1]

def p11(data):
    # Define our imported data:
    hit_u       = data[0]
    hit_v       = data[1]
    hit_w       = data[2]
    hst_u       = data[3]
    hst_v       = data[4]
    hst_w       = data[5]

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

    return [max_hit_u, 
            max_hit_v,
            max_hit_w,
            max_hst_u,
            max_hst_v,
            max_hst_w,
            min_hit_u,
            min_hit_v,
            min_hit_w,
            min_hst_u,
            min_hst_v,
            min_hst_w]


