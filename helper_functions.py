"""
Created on 11 Aug. 2017

@author: christoph
"""

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
from scipy import special, signal
from numpy.polynomial import polynomial





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
        print('ERROR: fibre selection not recognised')
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
    
    INPUT:
    x_norm: x-values (pixels) of all the lines, re-normalized to [-1,+1]
    m_norm: order numbers of all the lines, re-normalized to [-1,+1]
    orders: order numbers of all the lines
    
    OUTPUT:
    polytype: either 'polynomial' (default), 'legendre', or 'chebyshev' are accepted
    """
    
    if timit:
        start_time = time.time()
    
    if polytype in ['Polynomial','polynomial','p','P']:
        p_init = models.Polynomial2D(poly_deg)
        if debug_level > 0:
            print('OK, using standard polynomials...')
    elif polytype in ['Chebyshev','chebyshev','c','C']:
        p_init = models.Chebyshev2D(poly_deg,poly_deg)
        if debug_level > 0:
            print('OK, using Chebyshev polynomials...')
    elif polytype in ['Legendre','legendre','l','L']:
        p_init = models.Legendre2D(poly_deg,poly_deg)  
        if debug_level > 0:
            print('OK, using Legendre polynomials...')   
    else:
        print("ERROR: polytype not recognised ['(P)olynomial' / '(C)hebyshev' / '(L)egendre']")    
        
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
    Calculate the mean SNR of the extracted 1dim spectrum.
    
    INPUT:
    'flux'      : dictionary of the 1dim extracted spectrum (keys = orders)
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
    """
    
    cenmask = {}
    #loop over all masks
    for o in sorted(mask.keys()):
        ordmask = mask[o]
        if ordmask[len(ordmask)//2]:
            upstep_locations = np.argwhere(np.diff(ordmask.astype(int))==1)
            downstep_locations = np.argwhere(np.diff(ordmask.astype(int))==-1)
            cenmask[o] = ordmask.copy()
            if len(upstep_locations) >= 1:
                up = np.squeeze(find_nearest(upstep_locations,len(ordmask)//2,return_index=False))
                cenmask[o][:up+1] = False
            if len(downstep_locations) >= 1:
                down = np.squeeze(find_nearest(downstep_locations,len(ordmask)//2,return_index=False))
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
    
    
    
    
# def make_lenslets(input_arm, fluxes=[], mode='', seeing=0.8, llet_offset=0):
#     """Make an image of the lenslets with sub-pixel sampling.
#
#     Parameters
#     ----------
#     fluxes: float array (optional)
#         Flux in each lenslet
#
#     mode: string (optional)
#         'high' or 'std', i.e. the resolving power mode of the spectrograph. Either
#         mode or fluxes must be set.
#
#     seeing: float (optional)
#         If fluxes is not given, then the flux in each lenslet is defined by the seeing.
#
#     llet_offset: int
#         Offset in lenslets to apply to the input spectrum"""
#     print("Computing a simulated slit image...")
#     szx = input_arm.im_slit_sz
#     szy = 256
#     fillfact = 0.98
#     s32 = np.sqrt(3) / 2
#     hex_scale = 1.15
#     conv_fwhm = 30.0  # equivalent to a 1 degree FWHM for an f/3 input ??? !!! Double-check !!!
#     if len(fluxes) == 28:
#         mode = 'high'
#     elif len(fluxes) == 17:
#         mode = 'std'
#     elif len(mode) == 0:
#         print("Error: 17 or 28 lenslets needed... or mode should be set")
#         raise UserWarning
#     if mode == 'std':
#         nl = 17
#         lenslet_width = input_arm.lenslet_std_size
#         yoffset = (lenslet_width / input_arm.microns_pix / hex_scale * np.array(
#             [0, -s32, s32, 0, -s32, s32, 0])).astype(int)
#         xoffset = (lenslet_width / input_arm.microns_pix / hex_scale * np.array(
#             [-1, -0.5, -0.5, 0, 0.5, 0.5, 1.0])).astype(int)
#     elif mode == 'high':
#         nl = 28
#         lenslet_width = input_arm.lenslet_high_size
#         yoffset = (lenslet_width / input_arm.microns_pix / hex_scale * s32 * np.array(
#             [-2, 2, -2, -1, -1, 0, -1, -1, 0, 0, 0, 1, 1, 0, 1, 1, 2, -2, 2])).astype(int)
#         xoffset = (lenslet_width / input_arm.microns_pix / hex_scale * 0.5 * np.array(
#             [-2, 0, 2, -3, 3, -4, -1, 1, -2, 0, 2, -1, 1, 4, -3, 3, -2, 0, 2])).astype(int)
#     else:
#         print("Error: mode must be standard or high")
#
#     # Some preliminaries...
#     cutout_hw = int(lenslet_width / input_arm.microns_pix * 1.5)
#     im_slit = np.zeros((szy, szx))
#     x = np.arange(szx) - szx / 2.0
#     y = np.arange(szy) - szy / 2.0
#     xy = np.meshgrid(x, y)
#     # r and wr enable the radius from the lenslet center to be indexed
#     r = np.sqrt(xy[0] ** 2 + xy[1] ** 2)
#     wr = np.where(r < 2 * lenslet_width / input_arm.microns_pix)
#     # g is a Gaussian used for FRD
#     g = np.exp(-r ** 2 / 2.0 / (conv_fwhm / input_arm.microns_pix / 2.35) ** 2)
#     g = np.fft.fftshift(g)
#     g /= np.sum(g)
#     gft = np.conj(np.fft.rfft2(g))
#     pix_size_slit = input_arm.px_sz * (
#                 input_arm.f_col / input_arm.assym) / input_arm.f_cam * 1000.0 / input_arm.microns_pix
#     pix = np.zeros((szy, szx))
#     pix[np.where((np.abs(xy[0]) < pix_size_slit / 2) * (np.abs(xy[1]) < pix_size_slit / 2))] = 1
#     pix = np.fft.fftshift(pix)
#     pix /= np.sum(pix)
#     pix_ft = np.conj(np.fft.rfft2(pix))
#     # Create some hexagons. We go via a "cutout" for efficiency.
#     h_cutout = optics.hexagon(szy, lenslet_width / input_arm.microns_pix * fillfact / hex_scale)
#     hbig_cutout = optics.hexagon(szy, lenslet_width / input_arm.microns_pix * fillfact)
#     h = np.zeros((szy, szx))
#     hbig = np.zeros((szy, szx))
#     h[:, szx / 2 - szy / 2:szx / 2 + szy / 2] = h_cutout
#     hbig[:, szx / 2 - szy / 2:szx / 2 + szy / 2] = hbig_cutout
#     if len(fluxes) != 0:
#         # If we're not simulating seeing, the image-plane is uniform, and we only use
#         # the values of "fluxes" to scale the lenslet fluxes.
#         im = np.ones((szy, szx))
#         # Set the offsets to zero because we may be simulating e.g. a single Th/Ar lenslet
#         # and not starlight (from the default xoffset etc)
#         xoffset = np.zeros(len(fluxes), dtype=int)
#         yoffset = np.zeros(len(fluxes), dtype=int)
#     else:
#         # If we're simulating seeing, create a Moffat function as our input profile,
#         # but just make the lenslet fluxes uniform.
#         im = np.zeros((szy, szx))
#         im_cutout = optics.moffat2d(szy, seeing * input_arm.microns_arcsec / input_arm.microns_pix / 2, beta=4.0)
#         im[:, szx / 2 - szy / 2:szx / 2 + szy / 2] = im_cutout
#         fluxes = np.ones(len(xoffset))
#
#     # Go through the flux vector and fill in each lenslet.
#     for i in range(len(fluxes)):
#         im_one = np.zeros((szy, szx))
#         im_cutout = np.roll(np.roll(im, yoffset[i], axis=0), xoffset[i], axis=1) * h
#         im_cutout = im_cutout[szy / 2 - cutout_hw:szy / 2 + cutout_hw, szx / 2 - cutout_hw:szx / 2 + cutout_hw]
#         prof = optics.azimuthalAverage(im_cutout, returnradii=True, binsize=1)
#         prof = (prof[0], prof[1] * fluxes[i])
#         xprof = np.append(np.append(0, prof[0]), np.max(prof[0]) * 2)
#         yprof = np.append(np.append(prof[1][0], prof[1]), 0)
#         im_one[wr] = np.interp(r[wr], xprof, yprof)
#         im_one = np.fft.irfft2(np.fft.rfft2(im_one) * gft) * hbig
#         im_one = np.fft.irfft2(np.fft.rfft2(im_one) * pix_ft)
#         # !!! The line below could add tilt offsets... important for PRV simulation !!!
#         # im_one = np.roll(np.roll(im_one, tilt_offsets[0,i], axis=1),tilt_offsets[1,i], axis=0)*hbig
#         the_shift = int((llet_offset + i - nl / 2.0) * lenslet_width / input_arm.microns_pix)
#         im_slit += np.roll(im_one, the_shift, axis=1)
#         # print('the shift for fibre ',i, ' is : ',the_shift)
#     return im_slit
#
#
# def spectral_format(input_arm, xoff=0.0, yoff=0.0, ccd_centre={}):
#     """Create a spectrum, with wavelengths sampled in 2 orders.
#
#     Parameters
#     ----------
#     xoff: float
#         An input offset from the field center in the slit plane in
#         mm in the x (spatial) direction.
#     yoff: float
#         An input offset from the field center in the slit plane in
#         mm in the y (spectral) direction.
#     ccd_centre: dict
#         An input describing internal parameters for the angle of the center of the
#         CCD. To run this program multiple times with the same co-ordinate system,
#         take the returned ccd_centre and use it as an input.
#
#     Returns
#     -------
#     x:  (nm, ny) float array
#         The x-direction pixel co-ordinate corresponding to each y-pixel and each
#         order (m).
#     wave: (nm, ny) float array
#         The wavelength co-ordinate corresponding to each y-pixel and each
#         order (m).
#     blaze: (nm, ny) float array
#         The blaze function (pixel flux divided by order center flux) corresponding
#         to each y-pixel and each order (m).
#     ccd_centre: dict
#         Parameters of the internal co-ordinate system describing the center of the
#         CCD.
#     """
#     # Parameters for the Echelle. Note that we put the
#     # co-ordinate system along the principle Echelle axis, and
#     # make the beam come in at the gamma angle.
#     u1 = -np.sin(np.radians(input_arm.gamma) + xoff / input_arm.f_col)
#     u2 = np.sin(yoff / input_arm.f_col)
#     u3 = np.sqrt(1 - u1 ** 2 - u2 ** 2)
#     u = np.array([u1, u2, u3])
#     l = np.array([1.0, 0, 0])
#     s = np.array([0, np.cos(np.radians(input_arm.theta)), -np.sin(np.radians(input_arm.theta))])
#     # Orders for each wavelength. We choose +/- 1 free spectral range.
#     ms = np.arange(input_arm.m_min, input_arm.m_max + 1)
#     wave_mins = 2 * input_arm.d * np.sin(np.radians(input_arm.theta)) / (ms + 1.0)
#     wave_maxs = 2 * input_arm.d * np.sin(np.radians(input_arm.theta)) / (ms - 1.0)
#     wave = np.empty((len(ms), int(input_arm.nwave)))  # used to be: wave = np.empty( (len(ms),self.nwave))
#     for i in range(len(ms)):
#         wave[i, :] = np.linspace(wave_mins[i], wave_maxs[i], int(
#             input_arm.nwave))  # used to be: wave[i,:] = np.linspace(wave_mins[i],wave_maxs[i],self.nwave)
#     wave = wave.flatten()
#     ms = np.repeat(ms, input_arm.nwave)
#     order_frac = np.abs(ms - 2 * input_arm.d * np.sin(np.radians(input_arm.theta)) / wave)
#     ml_d = ms * wave / input_arm.d
#     # Propagate the beam through the Echelle.
#     nl = len(wave)
#     v = np.zeros((3, nl))
#     for i in range(nl):
#         v[:, i] = optics.grating_sim(u, l, s, ml_d[i])
#     ## Find the current mean direction in the x-z plane, and magnify
#     ## the angles to represent passage through the beam reducer.
#     if len(ccd_centre) == 0:
#         mean_v = np.mean(v, axis=1)
#         ## As the range of angles is so large in the y direction, the mean
#         ## will depend on the wavelength sampling within an order. So just consider
#         ## a horizontal beam.
#         mean_v[1] = 0
#         ## Re-normalise this mean direction vector
#         mean_v /= np.sqrt(np.sum(mean_v ** 2))
#     else:
#         mean_v = ccd_centre['mean_v']
#     for i in range(nl):
#         ## Expand the range of angles around the mean direction.
#         temp = mean_v + (v[:, i] - mean_v) * input_arm.assym
#         ## Re-normalise.
#         v[:, i] = temp / np.sum(temp ** 2)
#
#     ## Here we diverge from Veloce. We will ignore the glass, and
#     ## just consider the cross-disperser.
#     l = np.array([0, -1, 0])
#     theta_xdp = -input_arm.theta_i + input_arm.gamma
#     # Angle on next line may be negative...
#     s = optics.rotate_xz(np.array([1, 0, 0]), theta_xdp)
#     n = np.cross(s, l)  # The normal
#     print('Incidence angle in air: {0:5.3f}'.format(np.degrees(np.arccos(np.dot(mean_v, n)))))
#     # W is the exit vector after the grating.
#     w = np.zeros((3, nl))
#     for i in range(nl):
#         w[:, i] = optics.grating_sim(v[:, i], l, s, wave[i] / input_arm.d_x)
#     mean_w = np.mean(w, axis=1)
#     mean_w[1] = 0
#     mean_w /= np.sqrt(np.sum(mean_w ** 2))
#     print('Grating exit angle in glass: {0:5.3f}'.format(np.degrees(np.arccos(np.dot(mean_w, n)))))
#     # Define the CCD x and y axes by the spread of angles.
#     if len(ccd_centre) == 0:
#         ccdy = np.array([0, 1, 0])
#         ccdx = np.array([1, 0, 0]) - np.dot([1, 0, 0], mean_w) * mean_w
#         ccdx[1] = 0
#         ccdx /= np.sqrt(np.sum(ccdx ** 2))
#     else:
#         ccdx = ccd_centre['ccdx']
#         ccdy = ccd_centre['ccdy']
#     # Make the spectrum on the detector.
#     xpx = np.zeros(nl)
#     ypx = np.zeros(nl)
#     xy = np.zeros(2)
#     ## There is definitely a more vectorised way to do this.
#     for i in range(nl):
#         xy[0] = np.dot(ccdx, w[:, i]) * input_arm.f_cam / input_arm.px_sz
#         xy[1] = np.dot(ccdy, w[:, i]) * input_arm.f_cam / input_arm.px_sz
#         # Rotate the chip to get the orders along the columns.
#         rot_rad = np.radians(input_arm.drot)
#         rot_matrix = np.array([[np.cos(rot_rad), np.sin(rot_rad)], [-np.sin(rot_rad), np.cos(rot_rad)]])
#         xy = np.dot(rot_matrix, xy)
#         xpx[i] = xy[0]
#         ypx[i] = xy[1]
#     ## Center the spectra on the CCD in the x-direction.
#     if len(ccd_centre) == 0:
#         w = np.where((ypx < input_arm.szy / 2) * (ypx > -input_arm.szy / 2))[0]
#         xpix_offset = 0.5 * (np.min(xpx[w]) + np.max(xpx[w]))
#     else:
#         xpix_offset = ccd_centre['xpix_offset']
#     xpx -= xpix_offset
#     ## Now lets interpolate onto a pixel grid rather than the arbitrary wavelength
#     ## grid we began with.
#     nm = input_arm.m_max - input_arm.m_min + 1
#     x_int = np.zeros((nm, input_arm.szy))
#     wave_int = np.zeros((nm, input_arm.szy))
#     blaze_int = np.zeros((nm, input_arm.szy))
#     plt.clf()
#     for m in range(input_arm.m_min, input_arm.m_max + 1):
#         ww = np.where(ms == m)[0]
#         y_int_m = np.arange(np.max([np.min(ypx[ww]).astype(int), -input_arm.szy / 2]), \
#                             np.min([np.max(ypx[ww]).astype(int), input_arm.szy / 2]), dtype=int)
#         ix = y_int_m + input_arm.szy / 2
#         x_int[m - input_arm.m_min, ix] = np.interp(y_int_m, ypx[ww], xpx[ww])
#         wave_int[m - input_arm.m_min, ix] = np.interp(y_int_m, ypx[ww], wave[ww])
#         blaze_int[m - input_arm.m_min, ix] = np.interp(y_int_m, ypx[ww], np.sinc(order_frac[ww]) ** 2)
#         plt.plot(x_int[m - input_arm.m_min, ix], y_int_m)
#     plt.axis((-input_arm.szx / 2, input_arm.szx / 2, -input_arm.szx / 2, input_arm.szx / 2))
#     plt.draw()
#     return x_int, wave_int, blaze_int, {'ccdx': ccdx, 'ccdy': ccdy, 'xpix_offset': xpix_offset, 'mean_v': mean_v}
#
#
# def spectral_format_with_matrix(input_arm):
#     """Create a spectral format, including a detector to slit matrix at every point.
#
#     Returns
#     -------
#     x: (nm, ny) float array
#         The x-direction pixel co-ordinate corresponding to each y-pixel and each
#         order (m).
#     w: (nm, ny) float array
#         The wavelength co-ordinate corresponding to each y-pixel and each
#         order (m).
#     blaze: (nm, ny) float array
#         The blaze function (pixel flux divided by order center flux) corresponding
#         to each y-pixel and each order (m).
#     matrices: (nm, ny, 2, 2) float array
#         2x2 slit rotation matrices.
#     """
#     x, w, b, ccd_centre = spectral_format(input_arm)
#     x_xp, w_xp, b_xp, dummy = spectral_format(input_arm, xoff=-1e-3, ccd_centre=ccd_centre)
#     x_yp, w_yp, b_yp, dummy = spectral_format(input_arm, yoff=-1e-3, ccd_centre=ccd_centre)
#     dy_dyoff = np.zeros(x.shape)
#     dy_dxoff = np.zeros(x.shape)
#     # For the y coordinate, spectral_format output the wavelength at fixed pixel, not
#     # the pixel at fixed wavelength. This means we need to interpolate to find the
#     # slit to detector transform.
#     isbad = w * w_xp * w_yp == 0
#     for i in range(x.shape[0]):
#         ww = np.where(isbad[i, :] == False)[0]
#         dy_dyoff[i, ww] = np.interp(w_yp[i, ww], w[i, ww], np.arange(len(ww))) - np.arange(len(ww))
#         dy_dxoff[i, ww] = np.interp(w_xp[i, ww], w[i, ww], np.arange(len(ww))) - np.arange(len(ww))
#         # Interpolation won't work beyond the end, so extrapolate manually (why isn't this a numpy
#         # option???)
#         dy_dyoff[i, ww[-1]] = dy_dyoff[i, ww[-2]]
#         dy_dxoff[i, ww[-1]] = dy_dxoff[i, ww[-2]]
#
#     # For dx, no interpolation is needed so the numerical derivative is trivial...
#     dx_dxoff = x_xp - x
#     dx_dyoff = x_yp - x
#
#     # flag bad data...
#     x[isbad] = np.nan
#     w[isbad] = np.nan
#     b[isbad] = np.nan
#     dy_dyoff[isbad] = np.nan
#     dy_dxoff[isbad] = np.nan
#     dx_dyoff[isbad] = np.nan
#     dx_dxoff[isbad] = np.nan
#     matrices = np.zeros((x.shape[0], x.shape[1], 2, 2))
#     amat = np.zeros((2, 2))
#
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             ## Create a matrix where we map input angles to output coordinates.
#             amat[0, 0] = dx_dxoff[i, j]
#             amat[0, 1] = dx_dyoff[i, j]
#             amat[1, 0] = dy_dxoff[i, j]
#             amat[1, 1] = dy_dyoff[i, j]
#             ## Apply an additional rotation matrix. If the simulation was complete,
#             ## this wouldn't be required.
#             r_rad = np.radians(input_arm.extra_rot)
#             dy_frac = (j - x.shape[1] / 2.0) / (x.shape[1] / 2.0)
#             extra_rot_mat = np.array([[np.cos(r_rad * dy_frac), np.sin(r_rad * dy_frac)],
#                                       [-np.sin(r_rad * dy_frac), np.cos(r_rad * dy_frac)]])
#             amat = np.dot(extra_rot_mat, amat)
#             ## We actually want the inverse of this (mapping output coordinates back
#             ## onto the slit.
#             matrices[i, j, :, :] = np.linalg.inv(amat)
#     return x, w, b, matrices



def correct_orientation(img,orient=1):
    """
    (1) = same orientation as the simulated spectra, ie wavelength decreases from left to right and bottom to top
    """
    if orient == 1:
        img = np.fliplr(img.T)
    else:
        print('ERROR: selected orientation not defined!!!')
    
    return img











