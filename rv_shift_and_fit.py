'''
Created on 12 Jul. 2019

@author: christoph
'''
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as op




def calculate_rv_shift(wave_ref, ref_spect, flux, var, wave, return_fitted_spects=False, bad_threshold=10):
    """Calculates the Radial Velocity of each spectrum
    
    The radial velocity shift of the reference spectrum required
    to match the flux in each order in each input spectrum is calculated.
    
    The input fluxes to this method are flat-fielded data, which are then fitted with 
    a barycentrically corrected reference spectrum :math:`R(\lambda)`, according to 
    the following equation:

    .. math::
        f(x) = R( \lambda(x)  (1 - p_0/c) ) \\times \exp(p_1 x^2 + p_2 x + p_3)

    The first term in this equation is simply the velocity corrected spectrum, based on a 
    the arc-lamp derived reference wavelength scale :math:`\lambda(x)` for pixels coordinates x.
    The second term in the equation is a continuum normalisation - a shifted Gaussian was 
    chosen as a function that is non-zero everywhere. The scipy.optimize.leastsq function is used
    to find the best fitting set fof parameters :math:`p_0` through to :math`p_3`. 

    The reference spectrum function :math:`R(\lambda)` is created using a wavelength grid 
    which is over-sampled with respect to the data by a factor of 2. Individual fitted 
    wavelengths are then found by cubic spline interpolation on this :math:`R_j(\lambda_j)` 
    discrete grid.
    
    Parameters
    ----------
    wave_ref: 3D np.array(float)
        Wavelength coordinate map of form (Order, Wavelength/pixel*2+2), 
        where the wavelength scale has been interpolated.
    ref_spect: 3D np.array(float)
        Reference spectrum of form (Order, Flux/pixel*2+2), 
        where the flux scale has been interpolated.
    fluxes: 3D np.array(float)
        Fluxes of form (Observation, Order, Flux/pixel)
    vars: 3D np.array(float)
        Variance of form (Observation, Order, Variance/pixel)    
    bcors: 1D np.array(float)
        Barycentric correction for each observation.
    wave: 3D np.array(float)
        Wavelength coordinate map of form (Order, Wavelength/pixel)

    Returns
    -------
    rvs: 2D np.array(float)
        Radial velocities of format (Observation, Order)
    rv_sigs: 2D np.array(float)
        Radial velocity sigmas of format (Observation, Order)
    """
    
    nm = flux.shape[0]
    nf = flux.shape[1]
    npix = flux.shape[2]
    
    # initialise output arrays
    rvs = np.zeros( (nm,nf) )
    rv_sigs = np.zeros( (nm,nf) )
    redchi2_arr = np.zeros( (nm,nf) )
#     thefit_100 = np.zeros( (nm,nf) )

    initp = np.zeros(4)
    initp[3] = 0.5
#     initp[0] = 0.0
    spect_sdev = np.sqrt(var)
    if return_fitted_spects:
        fitted_spects = np.empty(flux.shape)
    
    print "fibre "
    
    # loop over all fibres
    for fib in range(nf):

        print str(fib+1),
        # Start with initial guess of no intrinsic RV for the target.
        nbad = 0
        # loop over all orders (skipping first order, as wl-solution is crap!)
        for o in range(1,nm):
            
            spl_ref = interp.InterpolatedUnivariateSpline(wave_ref[o,fib,::-1], ref_spect[o,fib,::-1], k=3)
            args = (wave[o,fib,:], flux[o,fib,:], spect_sdev[o,fib,:], spl_ref)
            
            # Remove edge effects in a slightly dodgy way (20 pixels is about 30km/s) 
            args[2][:20] = np.inf
            args[2][-20:] = np.inf

            the_fit = op.leastsq(rv_shift_resid, initp, args=args, diag=[1e3,1,1,1], Dfun=rv_shift_jac, full_output=True, xtol=1e-6)
            # the_fit[0] are the best-fit parms
            # the_fit[1] is the covariance matrix
            # the_fit[2] is auxiliary information on the fit (in form of a dictionary)
            # the_fit[3] is a string message giving information about the cause of failure.
            # the_fit[4] is an integer flag 
            # (if it is equal to 1, 2, 3 or 4, the solution was found. Otherwise, the solution was not found -- see op.leastsq documentation for more info)
            residBefore = rv_shift_resid(the_fit[0], *args)
            wbad = np.where(np.abs(residBefore) > bad_threshold)[0]
            nbad += len(wbad)
            
            chi2 = rv_shift_chi2(the_fit[0], *args)
            redchi2 = chi2 / (npix - len(initp))
            
            try:
#                 errorSigma = np.sqrt(chi2 * the_fit[1][0,0])
##                 normalisedValsBefore = normalised(residBefore, errorSigma)
                
                fittedSpec = rv_shift_resid(the_fit[0], *args, return_spect=True)
                
                # make the errors for the "bad" pixels infinity (so as to make the weights zero)
                args[2][np.where(np.abs(residBefore) > bad_threshold)] = np.inf

                the_fit = op.leastsq(rv_shift_resid, initp, args=args, diag=[1e3,1,1,1], Dfun=rv_shift_jac, full_output=True, xtol=1e-6)
                residAfter = rv_shift_resid( the_fit[0], *args)
                chi2 = rv_shift_chi2(the_fit[0], *args)
                redchi2 = chi2 / (npix - len(initp))
                
#                 errorSigma = np.sqrt(chi2 * the_fit[1][0,0])
##                 normalisedValsAfter = normalised(residAfter, errorSigma)

                redchi2_arr[o,fib] = redchi2

            except:
                pass

            
            # Some outputs for testing
            if return_fitted_spects:
                fitted_spects[o,fib,:] = rv_shift_resid(the_fit[0], *args, return_spect=True)
            
            #Save the fit and the uncertainty (the_fit[0][0] is the RV shift)
            rvs[o,fib] = the_fit[0][0]

            try:
                rv_sigs[o,fib] = np.sqrt(redchi2 * the_fit[1][0,0])
            except:
                rv_sigs[o,fib] = np.NaN
                
#             # the_fit[1] is the covariance matrix)
#             try:
#                 thefit_100[o,fib] = the_fit[1][0,0]
#             except:
#                 thefit_100[o,fib] = np.NaN

    if return_fitted_spects:
        return rvs, rv_sigs, redchi2_arr, fitted_spects
    else:
        return rvs, rv_sigs, redchi2_arr
    
    
    
def normalised(residVals, errorSigma):
    return residVals/errorSigma