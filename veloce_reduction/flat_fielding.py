'''
Created on 26 Apr. 2018

@author: christoph
'''

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import medfilt
from scipy import ndimage





# xdisp_boxsize = 1
# disp_boxsize = 15
# medfiltered_flat = medfilt(MW,[xdisp_boxsize,disp_boxsize])
# OR
# dum, filtered_flat = make_model_stripes(...) --> see spatial_profiles.py
#
# pix_sens_image = MW / medfiltered_flat   #should be roughly flat & scattering around 1
# 
# smoothed_MW = MW / pix_sens_image    #ie for the flat fields that means that smoothed_MW = filtered flat...
# smoothed_img = img / pix_sens_image
#

# maybe the not-fitted offsets in the model_stripes from "make_model_stripes_gausslike" are causing a Fubar in the division here!?!?!?





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
    'smoothed_flat' : dictionary of the smoothed (ie filtered) whites (keys = orders)
    
    MODHIST:
    24/05/2018 - CMB create
    """
    
    pix_sens = {}
    smoothed_flat = {}
    
    while filt.lower() not in ['g','gaussian','s','savgol','m','median']:
        print("ERROR: filter choice not recognised!")
        filt = raw_input("Please try again: ['(G)aussian','(S)avgol','(M)edian']")
    
    #loop over all orders
    for ord in sorted(f_flat.keys()): 
        if filt.lower() in ['g','gaussian']:
            #Gaussian filter
            smoothed_flat[ord] = ndimage.gaussian_filter(f_flat[ord], filter_width)    
            pix_sens[ord] = f_flat[ord] / smoothed_flat[ord]
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
        
    return smoothed_flat, pix_sens
    
    
    
    
def onedim_pixtopix_variations_single_order(f_flat, filt='gaussian', filter_width=25):
    """
    This routine applies a filter ('gaussian' / 'savgol' / 'median') to an observed flat field in order to determine the pixel-to-pixel sensitivity variations
    as well as the fringing pattern in the red orders. This is done in 1D, ie for the already extracted spectrum.
    
    INPUT:
    'f_flat'        : 1-dim array containing the extracted flux from the flat field (master white) for one order
    'filt'          : method of filtering ('gaussian' / 'savgol' / 'median') - WARNING: ONLY GAUSSIAN FILTER HAS BEEN IMPLEMENTED SO FAR!!!
    'filter_width'  : the width of the kernel for the filtering in pixels; defined differently for the different types of filters (see description of scipy.ndimage....)
    
    OUTPUT:
    'pix_sens'      : dictionary of the pixel-to-pixel sensitivities (keys = orders)
    'smoothed_flat' : dictionary of the smoothed (ie filtered) whites (keys = orders)
    
    MODHIST:
    05/10/2018 - CMB create   (clone of "onedim_pixtopix_variations")
    """
    
    while filt.lower() not in ['g','gaussian','s','savgol','m','median']:
        print("ERROR: filter choice not recognised!")
        filt = raw_input("Please try again: ['(G)aussian','(S)avgol','(M)edian']")
    
    if filt.lower() in ['g','gaussian']:
        #Gaussian filter
        smoothed_flat = ndimage.gaussian_filter(f_flat, filter_width)    
        pix_sens = f_flat / smoothed_flat
    elif filt.lower() in ['s','savgol']:
        print('WARNING: SavGol filter not implemented yet!!!')
        return
    elif filt.lower() in ['m','median']:
        print('WARNING: Median filter not implemented yet!!!')
        return
    else:
        #This should never happen!!!
        print("ERROR: filter choice still not recognised!")
        return
        
    return smoothed_flat, pix_sens    
    
    
    
    
    
def deblaze_orders(f, wl, smoothed_flat, mask, err=None, degpol=1, gauss_filter_sigma=3., maxfilter_size=100):
    
    f_dblz = {}
    if err is not None:
        err_dblz = {}
    
    #if using cross-correlation to get RVs, we need to de-blaze the spectra first
    for o in f.keys():
        #first, divide by the "blaze-function", ie the smoothed flat, which we got from filtering the MASTER WHITE
        f_dblz[o] = f[o] / (smoothed_flat[o]/np.max(smoothed_flat[o]))
        #get rough continuum shape by performing a series of filters
        cont_rough = ndimage.maximum_filter(ndimage.gaussian_filter(f_dblz[o],gauss_filter_sigma), size=maxfilter_size)
        #now fit polynomial to that rough continuum
        p = np.poly1d(np.polyfit(wl[o][mask[o]], cont_rough[mask[o]], degpol))
        #divide by that polynomial
        f_dblz[o] = f_dblz[o] / (p(wl[o]) / np.median(p(wl[o])[mask[o]]))
        #need to treat the error arrays in the same way, as need to keep relative error the same
        if err is not None:
            err_dblz[o] = err[o] / (smoothed_flat[o]/np.max(smoothed_flat[o]))
            err_dblz[o] = err_dblz[o] / (p(wl[o]) / np.median(p(wl[o])[mask[o]]))

    if err is not None:
        return f_dblz,err_dblz
    else:
        return f_dblz




