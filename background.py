'''
Created on 5 Feb. 2018

@author: christoph
'''

import numpy as np
from astropy.modeling import models, fitting
import warnings
import scipy.sparse as sparse

from veloce_reduction.helper_functions import *


# #make simulated background
# ny, nx = img.shape
# x = np.repeat(np.arange(nx) - nx/2,nx)
# y = np.tile(np.arange(ny) - ny/2,ny)
# xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx), np.linspace(y.min(), y.max(), ny))
# #m =    [a00,a01,a02,  a03  ,a10,a11,a12,a13,  a20 ,a21,a22,a23, a30 ,a31,a32,a33] for order=3
# parms = [90., 0., 0., 1.5e-9, 0., 0., 0., 0., -4e-9, 0., 0., 0., 1e-9, 0., 0., 0.]
# #parms = np.array([1000., 0., -5.e-5, 0., 0., 0., 0., 0., -5.e-5, 0., 0., 0., 0., 0., 0., 0.])
# zz_nf = polyval2d(xx, yy, parms)
# #add white noise
# noise = np.resize(np.random.normal(0, 1, nx*ny),(ny,nx))
# scaled_noise = noise * np.sqrt(zz_nf)
# zz = zz_nf + scaled_noise
# imgtest = img + zz




def extract_background(img, P_id, slit_height=25, output_file=None, return_mask=False, timit=False, debug_level=0):
    """
    This function marks all relevant pixels for extraction. Extracts the background (ie the inter-order regions = everything outside the order stripes)
    from the original 2D spectrum to a sparse matrix containing only relevant pixels.
    
    INPUT:
    'img'          : 2D echelle spectrum [np.array]
    'P_id'         : dictionary of the form of {order: np.poly1d} (as returned by make_P_id / identify_stripes)
    'slit_height'  : half the total slit height in pixels
    'output_file'  : path to file where result is saved
    'return_mask'  : boolean - do you want to return the mask of the background locations as well?
    'timit'        : for timing tests...
    'debug_level'  : debug level...
    
    OUTPUT:
    'mat.tocsc()'  : scipy.sparse_matrix containing the locations and values of the inter-order regions    
    """
    
    if timit:
        start_time = time.time()
    
    #logging.info('Extracting background...')
    print('Extracting background...')

    ny, nx = img.shape
    xx = np.arange(nx, dtype='f8')
    yy = np.arange(ny, dtype='f8')
    x_grid, y_grid = np.meshgrid(xx, yy, copy=False)
    bg_mask = np.ones((ny, nx), dtype=bool)

    for o, p in P_id.items():
        #xx = np.arange(nx, dtype=img.dtype)
        xx = np.arange(nx, dtype='f8')
        #yy = np.arange(ny, dtype=img.dtype)
        yy = np.arange(ny, dtype='f8')
    
        y = np.poly1d(p)(xx)
        x_grid, y_grid = np.meshgrid(xx, yy, copy=False)
    
        distance = y_grid - y.repeat(ny).reshape((nx, ny)).T
        indices = abs(distance) > slit_height
        
        bg_mask *= indices
        

    mat = sparse.coo_matrix((img[bg_mask], (y_grid[bg_mask], x_grid[bg_mask])), shape=(ny, nx))
    # return mat.tocsr()
    
    if timit:
        print('Elapsed time: ',time.time() - start_time,' seconds')
    
    if not return_mask:
        return mat.tocsc()
    else:
        return mat.tocsc(), bg_mask





def fit_background(bg, deg=3, timit=False, return_full=True):
    
    if timit:
        start_time = time.time()
        
    #find the non-zero parts of the sparse matrix
    #format is:
    #contents[0] = row indices
    #contents[1] = column indices
    #contents[2] = values
    contents = sparse.find(bg)
    
    ny, nx = bg.todense().shape
    
    #re-normalize to [-1,+1] - otherwise small errors in parms have huge effects
    x_norm = (contents[0] / ((nx-1)/2.)) - 1.
    y_norm = (contents[1] / ((ny-1)/2.)) - 1.
    z = contents[2]
    
    
    #m = polyfit2d(contents[0]-int(ny/2), contents[1]-int(nx/2), contents[2], order=deg)
    coeffs = polyfit2d(x_norm, y_norm, z, order=deg)
    #The result (m) is an array of the polynomial coefficients in the model f  = sum_i sum_j a_ij x^i y^j, 
    #eg:    m = [a00,a01,a02,a03,a10,a11,a12,a13,a20,.....,a33] for order=3
    
    
    if timit:
        print('Time taken for fitting background model: '+np.round(time.time() - start_time,2).astype(str)+' seconds...')
        start_time_2 = time.time()
    
    if return_full:
        xx = np.arange(nx)          
        xxn = (xx / ((nx-1)/2.)) - 1.
        yy = np.arange(ny)
        yyn = (yy / ((ny-1)/2.)) - 1.
        X,Y = np.meshgrid(xxn,yyn)
        #bkgd_img = polyval2d(X,Y,coeffs)
        bkgd_img = polyval2d(Y,X,coeffs)    #IDKY, but the indices are the wrong way around if I do it like in the line above!!!!!
    
    if timit:
        print('Time taken for constructing full background image: '+np.round(time.time() - start_time_2,2).astype(str)+' seconds...')
        print('Total time elapsed: '+np.round(time.time() - start_time,2).astype(str)+' seconds...')
    
    if return_full:
        return coeffs, bkgd_img
    else:
        return coeffs
 
 
 
 
 
def fit_background_astropy(bg, poly_deg=5, polytype='chebyshev', return_full=True, timit=False, debug_level=0):
    """ 
    WARNING: While this works just fine, it is MUCH MUCH slower than 'fit_background' above. The astropy fitting/modelling must be to blame...
    
    INPUT:
    'bg'               : sparse matrix containing the inter-order regions of the 2D image
    'poly_deg'         : the order of the polynomials to use in the fit (for both dimensions)
    'polype'           : either 'polynomial' (default), 'legendre', or 'chebyshev' are accepted
    'return_full'      : boolean - if TRUE, then the background model for each pixel for each order is returned; otherwise just the set of coefficients that describe it
    'timit'            : time it...
    'debug_level'      : for debugging only
    
    OUTPUT:
    EITHER
    'bkgd_coeffs'      : functional form of the coefficients that describe the background model
    OR 
    'bkgd_img'         : full background image constructed from best-fit model
    (selection between outputs is controlled by the 'return_full' keyword)
    
    TODO:
    figure out how to properly use weights here
    """
    
    if timit:
        start_time = time.time()
    
    #find the non-zero parts of the sparse matrix
    #format is:
    #contents[0] = row indices
    #contents[1] = column indices
    #contents[2] = values
    contents = sparse.find(bg)
    
    ny, nx = bg.todense().shape
    
    #re-normalize arrays to [-1,+1]    
    x_norm = (contents[0] / ((nx-1)/2.)) - 1.
    y_norm = (contents[1] / ((ny-1)/2.)) - 1.
    z = contents[2]
           
    #call the surface fitting routine
    bkgd_coeffs = fit_poly_surface_2D(x_norm, y_norm, z, weights=None, polytype = polytype, poly_deg=poly_deg)    
#     cheb_coeffs = fit_poly_surface_2D(x_norm, y_norm, z, weights=None, polytype = 'c', poly_deg=3, timit=True)  
#     lege_coeffs = fit_poly_surface_2D(x_norm, y_norm, z, weights=None, polytype = 'l', poly_deg=3, timit=True)  
#     poly_coeffs = fit_poly_surface_2D(x_norm, y_norm, z, weights=None, polytype = 'p', poly_deg=3, timit=True)

    if return_full:
        xx = np.arange(nx)          
        xxn = (xx / ((nx-1)/2.)) - 1.
        yy = np.arange(ny)
        yyn = (yy / ((ny-1)/2.)) - 1.
        X,Y = np.meshgrid(xxn,yyn)
        #bkgd_img = bkgd_coeffs(X,Y)
        bkgd_img = bkgd_coeffs(Y,X)     #IDKY, but the indices are the wrong way around if I do it like in the line above!!!!!

    if timit:
        print('Time elapsed: '+np.round(time.time() - start_time,2).astype(str)+' seconds...')    

    if return_full:
        return bkgd_coeffs,bkgd_img
    else:
        return bkgd_coeffs


