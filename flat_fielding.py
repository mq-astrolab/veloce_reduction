'''
Created on 26 Apr. 2018

@author: christoph
'''

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import medfilt
from scipy import ndimage

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

            



def onedim_pixtopix_variations(f_flat, filt='gaussian', filter_width=25):
    """
    This routine applies a filter ('gaussian' / 'savgol' / 'median') to an observed flat field in order to determine the pixel-to-pixel sensitivity variations
    as well as the fringing pattern in the red orders. This is done in 1D, ie for the already extracted spectrum.
    
    INPUT:
    'f_flat'        : dictionary containing the extracted flux from the flat field (master white) (keys = orders)
    'filt'          : method of filtering ('gaussian' / 'savgol' / 'median') - WARNING: ONLY GAUSSIAN FILTER HAS BEEN IMPLEMENTED SO FAR!!!
    'filter_width'  : the width of the kernel for the filtering in pixels; defined differently for the different types of filters (see description of scipy.ndimage....)
    
    OUTPUT:
    'pix_sens'      : dictionary of the pixel-to-pixel sensitivities (keys = orders)
    'filtered_flat' : dictionary of the smoothed (ie filtered) whites (keys = orders)
    
    MODHIST:
    24/05/2018 - CMB create
    """
    
    pix_sens = {}
    filtered_flat = {}
    
    while filt.lower() not in ['g','gaussian','s','savgol','m','median']:
        print("ERROR: filter choice not recognised!")
        filt = raw_input("Please try again: ['(G)aussian','(S)avgol','(M)edian']")
    
    #loop over all orders
    for ord in sorted(f_flat.keys()): 
        if filt.lower() in ['g','gaussian']:
            #Gaussian filter
            filtered_flat[ord] = ndimage.gaussian_filter(f_flat[ord], filter_width)    
            pix_sens[ord] = f_flat[ord] / filtered_flat[ord]
        elif filt.lower() in ['s','savgol']:
            print('WARNING: SavGol filter not implemented yet!!!')
            break
        elif filt.lower() in ['m','median']:
            print('WARNING: Median filter not implemented yet!!!')
            break
        else:
            #This should never happen!!!
            print("ERROR: filter choice still not recognised!")
            break
        
    return filtered_flat, pix_sens
    
    
    
    
    
def deblaze_orders(f, filtered_flat, mask=None):

    return




    
# def blaze_function(x,a):
#     return np.sinc(a*x)*np.sinc(a*x)
# 
# 
# 
# 
# 
# 
# 
# xdisp_boxsize = 1
# disp_boxsize = 15
# medfiltered_flat = medfilt(MW,[xdisp_boxsize,disp_boxsize])
# 
# pix_sens_image = MW / medfiltered_flat
# 
# smoothed_MW = MW / pix_sens_image    #ie for the flat fields that means that smoothed_MW = Running_Meanfilt...
# smoothed_img = img / pix_sens_image