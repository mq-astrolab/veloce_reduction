"""
Created on 11 Aug. 2017

@author: christoph
"""

import astropy.io.fits as pyfits
import numpy as np
#import veloce_reduction.optics as optics
import itertools
import warnings
import time
import math
import datetime
from astropy.modeling import models, fitting
import collections
#from scipy.special import erf
#from scipy.special import gamma
from scipy import ndimage
from scipy import special, signal
from numpy.polynomial import polynomial


#from veloce_reduction.calibration import crop_overscan_region




def linfunc(p, x):
    """linear function"""
    c,m = p
    return m*x + c



def gauss2D(xytuple, amp, x0, y0, x_sig, y_sig, theta):
    x,y = xytuple
    a = ((np.cos(theta))**2 / (2*x_sig**2)) + ((np.sin(theta))**2 / (2*y_sig**2))
    b = -np.sin(2*theta)/(4*x_sig**2) + np.sin(2*theta)/(4*y_sig**2)
    c = ((np.sin(theta))**2 / (2*x_sig**2)) + ((np.cos(theta))**2 / (2*y_sig**2))

    return amp * np.exp( -(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2) ) 



def fibmodel_fwhm(xarr, mu, fwhm, beta=2, alpha=0, norm=0):
    # model for the fibre profiles
    """WARNING: I think the relationship between sigma and FWHM is a function of beta!!!
        Maybe better to use sigma instead of FWHM!!!
    """
    
    #define constant (FWHM vs sigma, because FWHM = sigma * 2*sqrt(2*log(2))
    cons = 2*np.sqrt(np.log(2))
    
    #calculate "Gauss-like" model (if beta = 2 that's just a standard Gaussian)
    phi = np.exp(-(np.absolute(xarr-mu) * cons / fwhm) ** beta)
    
              
    if alpha == 0:
        #return "Gauss-like" model (symmetric)
        if norm ==1:
            #normfac = 1. / (np.sqrt(2 * np.pi) * (fwhm/(cons*np.sqrt(2))))
            normfac = beta / (2. * np.sqrt(2) * (fwhm/(cons*np.sqrt(2))) * special.gamma(1./beta))   #that's correct but make sure the wings have reached zero in the size of the array!!!
            phinorm = phi * normfac
            #print('normfac =',normfac)
            return phinorm
        else:
            return phi
    else:
        #now this allows for skewness, ie f(x)=2*phi(x)*PHI(alpha*x), where
        #phi(x)=Gaussian, and PHI(x)=CDF (cumulative distrobution funtion) = INT_-inf^x phi(x')dx' = 0.5*[1+erf(x/sqrt(2))]
#         if norm == 1:
#             norm2 = beta / (2. * (fwhm/(cons*np.sqrt(2))) * gamma(1./beta))
#             return norm2 * phi * (1. + erf(alpha * xarr / np.sqrt(2.)))
#         else:
            return phi * (1. + special.erf(alpha * xarr / np.sqrt(2.)))



def fibmodel(xarr, mu, sigma, beta=2, alpha=0, norm=0):
    # model for the fibre profiles (gauss-like function)
    #define constant (FWHM vs sigma, because FWHM = sigma * 2*sqrt(2*log(2))
    #cons = 2*np.sqrt(np.log(2))
    
    #calculate "Gauss-like" model (if beta = 2 that's just a standard Gaussian)
    phi = np.exp(- (np.absolute(xarr-mu) / (np.sqrt(2.)*sigma)) ** beta)
    
              
    if alpha == 0:
        #return "Gauss-like" model (symmetric)
        if norm ==1:
            #normfac = 1. / (np.sqrt(2 * np.pi) * (fwhm/(cons*np.sqrt(2))))
            normfac = beta / (2. * np.sqrt(2) * sigma * special.gamma(1./beta))   #that's correct but make sure the wings have reached zero in the size of the array!!!
            phinorm = phi * normfac
            #print('normfac =',normfac)
            return phinorm
        else:
            return phi
    else:
        #now this allows for skewness, ie f(x)=2*phi(x)*PHI(alpha*x), where
        #phi(x)=Gaussian, and PHI(x)=CDF (cumulative distrobution funtion) = INT_-inf^x phi(x')dx' = 0.5*[1+erf(x/sqrt(2))]
#         if norm == 1:
#             norm2 = beta / (2. * (fwhm/(cons*np.sqrt(2))) * gamma(1./beta))
#             return norm2 * phi * (1. + erf(alpha * xarr / np.sqrt(2.)))
#         else:
            return phi * (1. + special.erf(alpha * xarr / np.sqrt(2.)))



def CMB_multi_gaussian(x, *p):
    f = np.zeros(len(x))
    for i in range(len(p)//3):
        f += CMB_pure_gaussian(x, *p[i*3:i*3+3])
    return f

def CMB_pure_gaussian(x, mu, sig, amp):
    return (amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))

def gaussian_with_offset(x, mu, sig, amp, off):
    return (amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))) + off

def gaussian_with_offset_and_slope(x, mu, sig, amp, off, slope):
    return (amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))) + off + slope*x
 
def gaussian_with_slope(x, mu, sig, amp, slope):
    return (amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))) + slope*x 
 
def fibmodel_with_amp(x, mu, sigma, amp, beta):
    return amp * fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0)

def norm_fibmodel_with_amp(x, mu, sigma, amp, beta):
    return amp * fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=1)

def multi_fibmodel_with_amp(x, *p):
    f = np.zeros(len(x))
    for i in range(len(p)//4):
        f += fibmodel_with_amp(x, *p[i*4:i*4+4])
    return f

def fibmodel_with_offset(x, mu, sigma, beta, offset):
    return fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0) + offset

def fibmodel_with_amp_and_offset(x, mu, sigma, amp, beta, offset):
    return amp * fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0) + offset

def norm_fibmodel_with_amp_and_offset(x, mu, sigma, amp, beta, offset):
    return amp * fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=1) + offset

def fibmodel_with_amp_and_offset_and_slope(x, mu, sigma, amp, beta, offset, slope):
    return amp * fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0) + offset + slope*x

def gausslike_with_amp_and_offset_and_slope(x, mu, sigma, amp, beta, offset, slope):
    return amp * fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0) + offset + slope*x




def make_norm_profiles(x, o, col, fibparms, fibs='stellar', slope=False, offset=False):  
    
    #same number of fibres for every order, of course
    if fibs == 'all':
        #nfib = len(fibparms[o])  
        nfib = 28
        userange = np.arange(nfib) 
    elif fibs == 'stellar':
        nfib=19
        userange = np.arange(5,24,1)
    elif fibs == 'laser':
        nfib=1
        userange = np.arange(0,1,2)
    elif fibs == 'thxe':
        nfib=1
        userange = np.arange(27,28,2)
    elif fibs == 'sky3':
        nfib=3
        userange = np.arange(1,4,1)
    elif fibs == 'sky2':
        nfib=2
        userange = np.arange(25,27,1)
    else:
        print('ERROR: fibre selection not recognised')
        return
    
    #do we want to include extra "fibres" to take care of slope and/or offset? Default is NO for both (as this should be already taken care of globally)
    if offset:
        nfib += 1
    if slope:
        nfib += 1
    
    phi = np.zeros((len(x),28))
    
    for k,fib in enumerate(sorted(fibparms[o].keys())):
        mu = fibparms[o][fib]['mu_fit'](col)
        sigma = fibparms[o][fib]['sigma_fit'](col)
        beta = fibparms[o][fib]['beta_fit'](col)
        phi[:,k] = fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0)
    
    #deprecate phi-array to only use wanted fibres
    phi = phi[:,userange]
    
    if offset:
        phi[:,-2] = 1.
    if slope:
        phi[:,-1] = x - x[0]
    
    #return normalized profiles
    phinorm =  phi/np.sum(phi,axis=0)

    return phinorm

def make_norm_profiles_2(x, col, fppo, fibs='stellar', slope=False, offset=False):  
    """
    clone of "make_norm_profiles", but takes as "fppo" (= fibparms per order) as input, rather
    than "ord" and the entire "fibparms" dictionary
    """
    #same number of fibres for every order, of course
    if fibs == 'all':
        #nfib = len(fibparms[ord])  
        nfib = 28
        userange = np.arange(nfib) 
    elif fibs == 'stellar':
        nfib=19
        userange = np.arange(5,24,1)
    elif fibs == 'laser':
        nfib=1
        userange = np.arange(0,1,2)
    elif fibs == 'thxe':
        nfib=1
        userange = np.arange(27,28,2)
    elif fibs == 'sky3':
        nfib=3
        userange = np.arange(1,4,1)
    elif fibs == 'sky2':
        nfib=2
        userange = np.arange(25,27,1)
    else:
        print('ERROR: fibre selection not recognised!!!')
        return
    
    #do we want to include extra "fibres" to take care of slope and/or offset? Default is NO for both (as this should be already taken care of globally)
    if offset:
        nfib += 1
    if slope:
        nfib += 1
    
    phi = np.zeros((len(x),28))
    
    for k,fib in enumerate(sorted(fppo.keys())):
        mu = fppo[fib]['mu_fit'](col)
        sigma = fppo[fib]['sigma_fit'](col)
        beta = fppo[fib]['beta_fit'](col)
        phi[:,k] = fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0)
    
    #deprecate phi-array to only use wanted fibres
    phi = phi[:,userange]
    
    if offset:
        phi[:,-2] = 1.
    if slope:
        phi[:,-1] = x - x[0]
    
    #return normalized profiles
    phinorm =  phi/np.sum(phi,axis=0)

    return phinorm

def make_norm_profiles_temp(x, o, col, fibparms, slope=False, offset=False):  
    
    #xx = np.arange(4096)
    
    #same number of fibres for every order, of course
    nfib = 19
    if offset:
        nfib += 1
    if slope:
        nfib += 1
    
    phi = np.zeros((len(x),nfib))
    
    
    mu = fibparms[o]['fibre_03']['mu_fit'](col)
    sigma = fibparms[o]['fibre_03']['sigma_fit'](col)
    beta = fibparms[o]['fibre_03']['beta_fit'](col)
    for k in range(nfib):
        phi[:,k] = fibmodel(x, mu-k*1.98, sigma, beta=beta, alpha=0, norm=0)
    
    if offset:
        phi[:,-2] = 1.
    if slope:
        phi[:,-1] = x - x[0]
    
    #return normalized_profiles
    return phi/np.sum(phi,axis=0)

def make_norm_single_profile_temp(x, o, col, fibparms, slope=False, offset=False):  

    
    #same number of fibres for every order, of course
    nfib = 1
    if offset:
        nfib += 1
    if slope:
        nfib += 1
    
    phi = np.zeros((len(x),nfib))
    
    
    mu = fibparms[o]['mu'][col]
    sigma = fibparms[o]['sigma'][col]
    beta = fibparms[o]['beta'][col]
    for k in range(nfib):
        phi[:,k] = fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0)
    
    if offset:
        phi[:,-2] = 1.
    if slope:
        phi[:,-1] = x - x[0]
    
    #return normalized_profiles
    return phi/np.sum(phi,axis=0)





def blaze(x, alpha, amp, shift):
    return amp * (np.sin(np.pi * alpha * (x-shift))/(np.pi * alpha * (x-shift)))**2



def compose_matrix(sx,sy,shear,rot,tx,ty):
    m = np.zeros((3,3))
    m[0,0] = sx * math.cos(rot)
    m[1,0] = sx * math.sin(rot)
    m[0,1] = -sy * math.sin(rot+shear)
    m[1,1] = sy * math.cos(rot+shear)
    m[0,2] = tx
    m[1,2] = ty
    m[2,2] = 1
    return m



def center(df, width, height):
    m = compose_matrix(df['scale_x'], df['scale_y'], df['shear'], df['rotation'], df['translation_x'], df['translation_y'])
    xy = m.dot([width, height, 1])
    return xy[0], xy[1]



def polyfit2d(x, y, z, order=3, return_res=False):
    """The result (m) is an array of the polynomial coefficients in the model f  = sum_i sum_j a_ij x^i y^j, 
       has the form m = [a00,a01,a02,a03,a10,a11,a12,a13,a20,.....,a33] for order=3
    """
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, res, rank, s = np.linalg.lstsq(G, z)
    if return_res:
        return m, res
    else:
        return m



def test_polyfit2d(x, y, f, deg=3):
    x = np.asarray(x)
    y = np.asarray(y)
    f = np.asarray(f)
    deg = np.asarray(deg)
    vander = polynomial.polyvander2d(x, y, deg)
    vander = vander.reshape((-1,vander.shape[-1]))
    f = f.reshape((vander.shape[0],))
    c = np.linalg.lstsq(vander, f)[0]
    return c.reshape(deg+1)



def polyval2d(x, y, m):
    """
    Returns a 2-dim array of values with the parameters m from 'polyfit2d'.
    e.g.: m = [a00,a01,a02,a03,a10,a11,a12,a13,a20,.....,a33] for order=3
    """
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z



def fit_poly_surface_2D(x_norm, y_norm, z, weights=None, polytype = 'chebyshev', poly_deg=5, timit=False, debug_level=0):
    """
    Calculate 2D polynomial fit to normalized x and y values.
    Wrapper function for using the astropy fitting library.
    
    INPUT:
    'x_norm'      : x-values (pixels) of all the lines, re-normalized to [-1,+1]
    'm_norm'      : order numbers of all the lines, re-normalized to [-1,+1]
    'z'           : the 2-dim array of 'observed' values
    'weights'     : weights to use in the fitting
    'polytype'    : types of polynomials to use (either '(p)olynomial' (default), '(l)egendre', or '(c)hebyshev' are accepted)
    'poly_deg'    : degree of the polynomials
    'timit'       : boolean - do you want to measure execution run time?
    'debug_level' : for debugging... 
        
    OUTPUT:
    'p'  : coefficients of the best-fit polynomials
    """
    
    if timit:
        start_time = time.time()
    
    if polytype.lower() in ['p','polynomial']:
        p_init = models.Polynomial2D(poly_deg)
        if debug_level > 0:
            print('OK, using standard polynomials...')
    elif polytype.lower() in ['c','chebyshev']:
        p_init = models.Chebyshev2D(poly_deg,poly_deg)
        if debug_level > 0:
            print('OK, using Chebyshev polynomials...')
    elif polytype.lower() in ['l','legendre']:
        p_init = models.Legendre2D(poly_deg,poly_deg)  
        if debug_level > 0:
            print('OK, using Legendre polynomials...')   
    else:
        print("ERROR: polytype not recognised ['(P)olynomial' / '(C)hebyshev' / '(L)egendre']")    
        return
    
    fit_p = fitting.LevMarLSQFitter()  

    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(p_init, x_norm, y_norm, z, weights=weights)
        
    if timit:
        print('Time elapsed: '+np.round(time.time() - start_time,2).astype(str)+' seconds...')
        
    return p



def find_blaze_peaks(flux,P_id):
    """ find the peaks of the blaze function for each order """
    
    xx = np.arange(len(flux['order_01']))
    
    #first, smooth the spectra a little to remove cosmics or pixel-to-pixel sensitivity variations
    medfiltered = {}
    rough_peaks = {}
    peaks = {}
    for o in sorted(P_id.iterkeys()):
        medfiltered[o] = signal.medfilt(flux[o],9)
        rough_peaks[o] = np.mean(xx[medfiltered[o] == np.max(medfiltered[o])])   # need mean if there is more than one index where flux is max
        ix = rough_peaks[o].astype(int)
        fitrange = xx[ix-500:ix+501]
        #fit 2nd order polynomial to fitrange
        parms = np.poly1d(np.polyfit(fitrange, medfiltered[o][fitrange], 2))
        peakix = signal.argrelextrema(parms(fitrange), np.greater)[0]
        peaks[o] = fitrange[peakix]
        
    return peaks    



def find_nearest(arr,value,return_index=False):
    """
    This routine finds either the index or the value of the element of a 1-dim array that is closest to a given value.
    
    INPUT:
    'arr'   : the array in which to look 
    'value' : the value the closest thing to which you want to find
    'return_index' : boolean - do you want to return the index or the value?
    
    OUTPUT:
    'idx'      : the index of the closest thing to 'value' within 'arr' (if 'return_index' is set to TRUE)
    'arr[idx]' : the value within 'arr' that is closest to 'value' (if 'return_index' is set to FALSE)
    """
    idx = (np.abs(arr-value)).argmin()
    if return_index:
        return idx
    else:
        return arr[idx]        



def binary_indices(x):
    """
    Calculates and returns the non-zero indices in a binary representation of x, in order of increasing powers of 2.
    
    EXAMPLES:

    x = 11 = '1011'        --->   returns [0, 1, 3]      --->   because 11 = 2^1 + 2^1 + 2^3
    x = 153 = '10011001'   --->   returns [0, 3, 4, 7]   --->   because 153 = 2^0 + 2^3 + 2^4 + 2^7
    x = 201 = '11001001'   --->   returns [0, 3, 6, 7]   --->   because 153 = 2^0 + 2^3 + 2^6 + 2^7
    """
    binum = np.binary_repr(x)
    liste = list(reversed(binum))
    intlist = [int(i) for i in liste]
    return np.flatnonzero(np.array(intlist))
    
    
    
def sigma_clip(x,tl,th=None,centre='median'):    
    """
    Perform sigma-clipping of 1D array.
    
    INPUT:
    'x'       : the 1D array to be sigma-clipped
    'tl'      : lower threshold (in terms of sigma)
    'th'      : higher threshold (in terms of sigma) (if only one threshold is given then th=tl=t)
    'centre'  : method to determine the centre ('median' or 'mean')
    
    OUTPUT:
    'x'  : the now sigma-clipped array
    """
    
    #make sure both boundaries are defined
    if th is None:
        th = tl
    
    clipped = x.copy()
    
    while True:    
        rms = np.std(clipped)
        if centre.lower() == 'median':
            bad_high = clipped - np.median(clipped) > th*rms 
            bad_low = np.median(clipped) - clipped > tl*rms 
        elif centre.lower() == 'mean':
            bad_high = clipped - np.mean(clipped) > th*rms 
            bad_low = np.mean(clipped) - clipped > tl*rms 
        else:
            print('ERROR: Method for computing centre must be "median" or "mean"')
            return
        goodix = ~bad_high & ~bad_low      
        if np.sum(~goodix) == 0:
            break
        else:
            clipped = clipped[goodix]
    
    return clipped
    
    
    
def offset_pseudo_gausslike(x, G_amplitude, L_amplitude, G_center, L_center, G_sigma, L_sigma, beta):
    """ similar to Pseudo-Voigt-Model (e.g. see here: https://lmfit.github.io/lmfit-py/builtin_models.html), 
        but allows for offset between two functions and allows for beta to vary """
    G = G_amplitude * fibmodel(x, G_center, G_sigma, beta=beta, alpha=0, norm=1)
    L = L_amplitude * L_sigma/((L_center - L_sigma)**2 + L_sigma**2)
    return G+L
    
     
    
def get_iterable(x):
    """ make any python object an 'iterable' so that you can loop over it and your code does not crash, even if it's just a number """
    if isinstance(x, collections.Iterable):
        return x
    else:
        return (x,)    
    
    
        
def get_datestring():
    """ get the current date in a string of the format: 'YYYYMMDD' """
    now = datetime.datetime.now()
    dum = str(now)[:10]
    datestring = ''.join(dum.split('-'))
    return datestring



def get_mean_snr(flux, err=None, per_order=False):
    """ 
    Calculate the mean SNR of the extracted 1-dim spectrum.
    
    INPUT:
    'flux'      : dictionary of the 1-dim extracted spectrum (keys = orders)
    'err'       : dictionary of the corresponding uncertainties (if not provided, the SQRT of the flux is used by default)
    'per_order' : boolean - do you want to return the mean SNR of each order? 
    
    OUTPUT:
    'snr_ord' : he mean snr (per 'super-pixel' / collapsed pixel) per order
    'snr'     : the mean snr (per 'super-pixel' / collapsed pixel) of the input spectrum
    
    MODHIST:
    18/05/2018 - CMB create
    30/05/2018 - CMB added 'per_order' keyword
    """
    
    #calculate errors if not provided
    if err is None:
        print('WARNING: error-array not provided! Using SQRT(flux) as an estimate instead...')
        print
        err = {}
        for o in sorted(flux.keys()):
            mask1 = flux[o] < 0.
            flux[o][mask1] = 0.
            err[o] = np.sqrt(flux[o])
            
    snr_ord = np.array([])
    
    for o in sorted(flux.keys()):
        #can't have negative flux or zero error
        mask2 = np.logical_and(flux[o] >= 0., err[o] > 0.)
        snr_ord = np.append(snr_ord, np.mean(flux[o][mask2] / err[o][mask2]))
    
    snr =  np.mean(snr_ord)
    
    if per_order:
        return snr_ord
    else:
        return snr



def central_parts_of_mask(mask):
    """
    This routine reduces the True parts of an order mask to only the large central posrtion if there are multiple True parts.
    These are the parts we want to use for the cross-correlation to get RVs.
    
    INPUT:
    'mask'    : mask dictionary from "make_mask_dict" (keys = orders)
    
    OUTPUT:
    'cenmask' : mask containing the central true parts only
    
    MODHIST:
    01/06/2018 - CMB create    
    03/08/2018 - fixed bug if mask is asymmetric - now requires closest upstep location to be to the left of the order centre and nearest downstep location to 
                 be to the right of the order centre
    """
    
    cenmask = {}
    #loop over all masks
    for o in sorted(mask.keys()):
        ordmask = mask[o]
        if ordmask[len(ordmask)//2]:
            upstep_locations = np.argwhere(np.diff(ordmask.astype(int)) == 1)
            downstep_locations = np.argwhere(np.diff(ordmask.astype(int)) == -1)
            cenmask[o] = ordmask.copy()
            if len(upstep_locations) >= 1:
                up = np.squeeze(find_nearest(upstep_locations[upstep_locations < len(ordmask)//2],len(ordmask)//2,return_index=False))
                cenmask[o][:up+1] = False
            if len(downstep_locations) >= 1:
                down = np.squeeze(find_nearest(downstep_locations[downstep_locations > len(ordmask)//2],len(ordmask)//2,return_index=False))
                cenmask[o][down+1:] = False           
        else:
            print('ERROR: order centre is masked out!!!')
            return

    return cenmask



def short_filenames(file_list):
    dum = [longname.split('/')[-1] for longname in file_list]
    fnarr = ['.'.join(fn.split('.')[0:-1]) for fn in dum]
    return fnarr



def make_quadrant_masks(nx,ny):
    
    #define four quadrants via masks
    q1 = np.zeros((ny,nx),dtype='bool')
    q1[:(ny/2), :(nx/2)] = True
    q2 = np.zeros((ny,nx),dtype='bool')
    q2[:(ny/2), (nx/2):] = True
    q3 = np.zeros((ny,nx),dtype='bool')
    q3[(ny/2):, (nx/2):] = True
    q4 = np.zeros((ny,nx),dtype='bool')
    q4[(ny/2):, :(nx/2)] = True
    
    return q1,q2,q3,q4



def correct_orientation(img, orient=1):
    """
    (1) = same orientation as the simulated spectra, ie wavelength decreases from left to right and bottom to top
    """
    if orient == 1:
        img = np.fliplr(img.T)
    else:
        print('ERROR: selected orientation not defined!!!')
    
    return img



def make_median_image(imglist, MB=None, raw=False):
    """
    Make a median image from a given list of images.
    
    INPUT:
    'imglist'  : list of files (incl. directories)
    'MB'       : master bias frame - if provided, it will be subtracted from every image before median image is computed
    'raw'      : boolean - set to TRUE if you want to retain the original size and orientation; 
                 otherwise the image will be brought to the 'correct' orientation and the overscan regions will be cropped
                 
    OUTPUT:
    'medimg'   : median image             
    """
    
    from veloce_reduction.calibration import crop_overscan_region
    
    #prepare array
    allimg = []
    
    #loop over all files in "dark_list"
    for file in imglist:
        #read in dark image
        img = pyfits.getdata(file)
        if not raw:
            #bring to "correct" orientation
            img = correct_orientation(img)
            #remove the overscan region
            img = crop_overscan_region(img)
        if MB is not None: 
            #subtract master bias (if provided)
            img = img - MB
        
        #add image to list
        allimg.append(img)

    #get median image
    medimg = np.median(np.array(allimg), axis=0)
    
    return medimg



def find_maxima(data, gauss_filter_sigma=0., min_peak=0.1, return_values=0):
    """
    not currently used!!!
    """
    # smooth image slightly for noise reduction
    smooth_data = ndimage.gaussian_filter(data, gauss_filter_sigma)
    # find all local maxima
    peaks = np.r_[True, smooth_data[1:] > smooth_data[:-1]] & np.r_[smooth_data[:-1] > smooth_data[1:], True]
    # only use peaks higher than a certain threshold
    idx = np.logical_and(peaks, smooth_data > min_peak * np.max(smooth_data))
    maxix = np.arange(len(data))[idx]
    maxima = data[maxix]
    
    if return_values != 0:
        return maxix,maxima
    else:
        return maxix



def affine_matrix(scale_x=1, scale_y=1, theta=0, dx=0, dy=0, shear_x=0, shear_y=0):
    """
    theta in degs clockwise
    
    Order:
    (1) rotation
    (2) translation
    (3) scaling
    (4) shear in x
    (5) shear in y
    """
    #m = np.zeros((3,3))
    #m[zeile,spalte] = m[row,column]
    
    m_scale = np.eye(3)
    m_scale[0,0] = scale_x
    m_scale[1,1] = scale_y
    
    m_rot = np.eye(3)
    m_rot[0,0] = np.cos(np.deg2rad(theta))
    m_rot[0,1] = np.sin(np.deg2rad(theta))
    m_rot[1,0] = -np.sin(np.deg2rad(theta))
    m_rot[1,1] = np.cos(np.deg2rad(theta))
    
    m_trans = np.eye(3)
    m_trans[0,2] = dx
    m_trans[1,2] = dy
    
    m_shear_x = np.eye(3)
    m_shear_x[0,1] = shear_x
    
    m_shear_y = np.eye(3)
    m_shear_y[1,0] = shear_y
    
    m = np.matmul(m_shear_y, np.matmul(m_shear_x, np.matmul(m_scale, np.matmul(m_trans, m_rot))))
    
    return m



def affine_transformation(xytuple, scale_x=1, scale_y=1, theta=0, dx=0, dy=0, shear_x=0, shear_y=0):
    """
    dummy dunction, not currently used; xy-tuple may not be correctly implemented yet
    """
#     x,y = xytuple
#     #create affine transformation matrix
#     m = affine_matrix(scale_x=scale_x, scale_y=scale_y, theta=theta, dx=dx, dy=dy, shear_x=shear_x, shear_y=shear_y)
#     #calculate set of transformed points
#     new_points = np.dot(m,points)
#     
#     return np.dot(m,x)
    return


