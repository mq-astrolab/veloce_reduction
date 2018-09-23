'''
Created on 29 Nov. 2017

@author: christoph
'''

import time
import numpy as np
from scipy import signal
from scipy import ndimage
import matplotlib.pyplot as plt
import scipy.optimize as op
import lmfit
from lmfit import Model
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, PseudoVoigtModel, MoffatModel, Pearson7Model, StudentsTModel, BreitWignerModel, LognormalModel
from lmfit.models import DampedOscillatorModel, DampedHarmonicOscillatorModel, ExponentialGaussianModel, SkewedGaussianModel, DonaichModel
from readcol import readcol 
import datetime

from .helper_functions import fibmodel_with_amp, CMB_pure_gaussian, multi_fibmodel_with_amp, CMB_multi_gaussian, offset_pseudo_gausslike, fit_poly_surface_2D






def find_suitable_peaks(rawdata, thresh = 5000., bgthresh = 2000., maxthresh = None, gauss_filter_sigma=1., slope=1e-4,
                        clip_edges=True, remove_bg=False, return_masks=False, debug_level=0, timit=False):
    """
    This routine finds (rough, to within 1 pixel) peak locations in 1D data.
    Detection threshold, background threshold, maximum threshold, and smoothing kernel width can be provided as keyword parameters.

    INPUT:
    'rawdata'   : one-dimensional (raw) data array

    KEYWORD PARAMETERS:
    'thresh'              : minimum height of peak in the FILTERED data to be considered a good peak
    'bgthresh'            : minimum height of peak in the filtered data to be considered a real (but not good) peak
    'maxthresh'           : maximum height of peak in the filtered data to be included (ie you can exclude saturated peaks etc)
    'gauss_filter_sigma'  : width of the Gaussian smoothing kernel (set to 0 if you want no smoothing)
    'slope'               : value of the (tiny) slope that's added in order to break possible degeneracies between adjacent pixels
    'clip_edges'          : boolean - do you want to clip the first and last peak, just to be on the safe side and avoid edge effects?
    'remove_bg'           : boolean - do you want to run a crude background removal before identifying peaks?
    'return_masks'        : boolean - do you want to return the masks as well as the data arrays?
    'debug_level'         : for debugging...
    'timit'               : boolean - do you want to clock execution run-time?

    OUTPUT:
    'goodpeaks'   : 1D array containing the "good" peak locations (above thresh and below maxthresh)
    'mostpeaks'   : 1D array containing all "real" peak locations (above bg_thresh and below maxthresh)
    'allpeaks'    : 1D array containing ALL peak locations (ie all local maxima, except for edges)
    'first_mask'  : mask that masks out shallow noise peaks from allpeaks (ONLY IF 'return_masks' is set to TRUE)
    'second_mask' : mask that masks out saturated lines above maxthresh from mostpeaks (ONLY IF 'return_masks' is set to TRUE)
    'third_mask'  : mask that masks out lines below thresh from mostpeaks (ONLY IF 'return_masks' is set to TRUE)
                    ie: goodpeaks = allpeaks[first_mask][second_mask][third_mask]
    
    TODO:
    (1) make a relative threshold criterion
    """

    #this routine is extremely fast, no need to optimise for speed
    if timit:
        start_time = time.time()
    
    xx = np.arange(len(rawdata))
    
    #run a crude background removal first:
    if remove_bg:
        bgfit = ndimage.gaussian_filter(ndimage.minimum_filter(rawdata,size=50.), 10.)
    else:
        bgfit = np.zeros(len(rawdata))
    data = rawdata - bgfit
    
    #smooth data to make sure we are not finding noise peaks, and add tiny slope to make sure peaks are found even when pixel-values are like [...,3,6,18,41,41,21,11,4,...]
    filtered_data = ndimage.gaussian_filter(data.astype(np.float), gauss_filter_sigma) + xx*slope
    
    #find all local maxima in smoothed data (and exclude the leftmost and rightmost maxima to avoid nasty edge effects...)
    allpeaks = signal.argrelextrema(filtered_data, np.greater)[0]
    
    ### this alternative version of finding extrema is completely equivalent (except that the below version accepts first and last as extrema, which I do not want)
    #testix = np.r_[True, filtered_data[1:] > filtered_data[:-1]] & np.r_[filtered_data[:-1] > filtered_data[1:], True]
    #testpeaks = np.arange(len(xx))[testix]
    
    #exclude first and last peaks to avoid nasty edge effects
    if clip_edges:
        allpeaks = allpeaks[1:-1]
    
    #make first mask to determine which peaks from linelist to use in wavelength solution
    first_mask = np.ones(len(allpeaks), dtype='bool')
    
    #remove shallow noise peaks
    first_mask[filtered_data[allpeaks] < bgthresh] = False
    mostpeaks = allpeaks[first_mask]
    
    #make mask which we need later to determine which peaks from linelist to use in wavelength solution
    second_mask = np.ones(len(mostpeaks), dtype='bool')
    
    #remove saturated lines
    if maxthresh is not None:
        second_mask[filtered_data[mostpeaks] > maxthresh] = False
        mostpeaks = mostpeaks[second_mask]
     
    #make mask which we need later to determine which peaks from linelist to use in wavelength solution    
    third_mask = np.ones(len(mostpeaks), dtype='bool')
    
    #only select good peaks higher than a certain threshold
    third_mask[filtered_data[mostpeaks] < thresh] = False
    goodpeaks = mostpeaks[third_mask]           #ie goodpeaks = allpeaks[first_mask][second_mask][third_mask]
    
    #for testing and debugging...
    if debug_level >= 1:
        print('Total number of peaks found: '+str(len(allpeaks)))
        print('Number of peaks found that are higher than '+str(int(thresh))+' counts: '+str(len(goodpeaks)))
        plt.figure()
        plt.plot(data)
        plt.plot(filtered_data)
        plt.scatter(goodpeaks, data[goodpeaks], marker='x', color='r', s=40)
        plt.plot((0,len(data)),(bgthresh,bgthresh),'r--')
        plt.plot((0,len(data)),(thresh,thresh),'g--')
        plt.xlabel('pixel')
        plt.ylabel('counts')
        #plt.vlines(thar_pos_guess, 0, np.max(data))
        plt.show()
    
    if timit:
        delta_t = time.time() - start_time
        print('Time elapsed: '+str(round(delta_t,5))+' seconds')
    
    if return_masks:
        return goodpeaks, mostpeaks, allpeaks, first_mask, second_mask, third_mask
    else:
        return goodpeaks, mostpeaks, allpeaks





def fit_emission_lines(data, fitwidth=4, thresh = 5000., bgthresh = 2000., maxthresh = None, slope=1e-4, laser=False,
                       varbeta=True, offset=False, minsigma=0., maxsigma=np.inf, sigma_0=1., minamp=0., maxamp=np.inf,
                       minbeta=1., maxbeta=4., beta_0=2., return_all_pars=False, return_qualflag=False, verbose=False, timit=False):
    """
    This routine identifies and fits emission lines in a 1-dim spectrum (ie generally speaking it finds and fits peaks in a 1dim array), using "scipy.optimize.curve_fit".
    Detection threshold, background threshold, and maximum threshold can be provided as keyword parameters. Different models for the peak-like function can be selected.

    INPUT:
    'data'   : one-dimensional data array

    KEYWORD PARAMETERS:
    'fitwidth'         : range around the identified peaks to be used for the peak-fitting 
    'thresh'           : minimum height of peak in the FILTERED data to be considered a good peak
    'bgthresh'         : minimum height of peak in the FILTERED data to be considered a real (but not good) peak
    'maxthresh'        : maximum height of peak in the FILTERED data to be included (ie you can exclude saturated peaks etc)
    'slope'            : value of the (tiny) slope that's added in order to break possible degeneracies between adjacent pixels
    'laser'            : boolean - is this a LFC spectrum? (if set to true, there is no check for blended lines)
    'varbeta'          : boolean - if set to TRUE, use a Gauss-like function for fitting, if set to FALSE use a plain Gaussian
    'offset'           : boolean - do you want to fit an offset to each peak as well?
    'minsigma'         : lower threshold for allowed sigma values
    'maxsigma'         : upper threshold for allowed sigma values
    'sigma_0'          : initial guess for sigma
    'minamp'           : lower threshold for allowed amplitude values
    'maxamp'           : upper threshold for allowed amplitude values
    'minbeta'          : lower threshold for allowed beta values
    'maxbeta'          : upper threshold for allowed beta values
    'beta_0'           : initial guess for beta
    'return_all_pars'  : boolean - do you want to return all fit parameters?
    'return_qualflag'  : boolean - do you want to return a quality flag for each line fit?
    'verbose'          : boolean - for debugging...
    'timit'            : boolean - do you want to clock execution run-time?    
    
    OUTPUT:
    'line_pos_fitted'    : fitted line positions
    'line_sigma_fitted'  : fitted line sigmas
    'line_amp_fitted'    : fitted line amplitudes
    'line_beta_fitted'   : fitted line betas (exponent in Gauss-like functions, which is equal to 2 for pure Gaussians)
    'qualflag'           : quality flag (1=good, 0=bad) for each line (only if 'return_qualflag' is set to TRUE)
    
    TODO:
    (1) make a relative threshold criterion
    """
    
    print('ATTENTION: You should consider using the more up-to-date and more robust function "fit_emission_lines_lmfit" instead!')
#     choice = None
#     while choice is None:
#         choice = raw_input("Do you want to continue? [y/n]: ")
#         if choice.lower() not in ('y','n'):
#             print('Invalid input! Please try again...')
#             choice = None
#    
#     #stop here if you'd rather use "fit_emission_lines_lmfit"
#     if choice.lower() == 'n':
#         return
    
    #OK, continue, you have been warned...
    if timit:
        start_time = time.time()
    
    xx = np.arange(len(data))
    
    #find rough peak locations
    goodpeaks,mostpeaks,allpeaks = find_suitable_peaks(data, thresh=thresh, bgthresh=bgthresh, maxthresh=maxthresh, slope=slope)
    
    if verbose:
        print('Fitting '+str(len(goodpeaks))+' emission lines...')
    
    line_pos_fitted = []
    if return_all_pars:
        line_amp_fitted = []
        line_sigma_fitted = []
        if varbeta:
            line_beta_fitted = []
    if return_qualflag:
        qualflag = []
        
    for xguess in goodpeaks:
        if verbose:
            print('xguess = ',xguess)
        ################################################################################################################################################################################
        #METHOD 1 (using curve_fit; slightly faster than method 2, but IDK how to make sure the fit converged (as with .ier below))

        if not laser:
            #check if there are any other peaks in the vicinity of the peak in question (exclude the peak itself)
            checkrange = np.r_[xx[np.max([0,xguess - 2*fitwidth]) : xguess], xx[xguess+1 : np.min([xguess + 2*fitwidth+1, len(data)-1])]]
            peaks = np.r_[xguess]
            #while len((set(checkrange) & set(allpeaks))) > 0:    THE RESULTS ARE COMPARABLE, BUT USING MOSTPEAKS IS MUCH FASTER
            while len((set(checkrange) & set(mostpeaks))) > 0:
                #where are the other peaks?
                #other_peaks = np.intersect1d(checkrange, allpeaks)    THE RESULTS ARE COMPARABLE, BUT USING MOSTPEAKS IS MUCH FASTER
                other_peaks = np.intersect1d(checkrange, mostpeaks)
                peaks = np.sort(np.r_[peaks, other_peaks])
                #define new checkrange
                checkrange = xx[np.max([0,peaks[0] - 2*fitwidth]) : np.min([peaks[-1] + 2*fitwidth + 1,len(data)-1])]
                dum = np.in1d(checkrange, peaks)
                checkrange = checkrange[~dum]
        else:
            peaks = np.r_[xguess]

        npeaks = len(peaks)
        xrange = xx[np.max([0,peaks[0] - fitwidth]) : np.min([peaks[-1] + fitwidth + 1,len(data)-1])]      # this should satisfy: len(xrange) == len(checkrange) - 2*fitwidth + len(peaks)

        if npeaks == 1:
            if offset:
                if varbeta:
                    guess = np.array([xguess, sigma_0, data[xguess], beta_0, 0])
                    popt, pcov = op.curve_fit(fibmodel_with_amp_and_offset, xrange, data[xrange], p0=guess, bounds=([xguess-2,minsigma,minamp,minbeta,0],[xguess+2,maxsigma,maxamp,maxbeta,thresh]))
                else:
                    guess = np.array([xguess, sigma_0, data[xguess], 0])
                    popt, pcov = op.curve_fit(gaussian_with_offset, xrange, data[xrange], p0=guess, bounds=([xguess-2,minsigma,minamp,0],[xguess+2,maxsigma,maxamp,]))
            else:
                if varbeta:
                    guess = np.array([xguess, sigma_0, data[xguess], beta_0])
                    popt, pcov = op.curve_fit(fibmodel_with_amp, xrange, data[xrange], p0=guess, bounds=([xguess-2,minsigma,minamp,minbeta],[xguess+2,maxsigma,maxamp,maxbeta]))
                else:
                    guess = np.array([xguess, sigma_0, data[xguess]])
                    popt, pcov = op.curve_fit(CMB_pure_gaussian, xrange, data[xrange], p0=guess, bounds=([xguess-2,minsigma,minamp],[xguess+2,maxsigma,maxamp]))
            fitted_pos = popt[0]
            if return_all_pars:
                fitted_sigma = popt[1]
                fitted_amp = popt[2]
                if varbeta:
                    fitted_beta = popt[3]
        else:
            guess = []
            lower_bounds = []
            upper_bounds = []
            for i in range(npeaks):
                if varbeta:
                    guess.append(np.array([peaks[i], sigma_0, data[peaks[i]], beta_0]))
                    lower_bounds.append([peaks[i]-2,minsigma,minamp,minbeta])
                    upper_bounds.append([peaks[i]+2,maxsigma,maxamp,maxbeta])
                else:
                    guess.append(np.array([peaks[i], sigma_0, data[peaks[i]]]))
                    lower_bounds.append([peaks[i]-2,minsigma,minamp])
                    upper_bounds.append([peaks[i]+2,maxsigma,maxamp])
            guess = np.array(guess).flatten()
            lower_bounds = np.array(lower_bounds).flatten()
            upper_bounds = np.array(upper_bounds).flatten()
            if varbeta:
                popt, pcov = op.curve_fit(multi_fibmodel_with_amp, xrange, data[xrange], p0=guess, bounds=(lower_bounds,upper_bounds))
            else:
                popt, pcov = op.curve_fit(CMB_multi_gaussian, xrange, data[xrange], p0=guess, bounds=(lower_bounds,upper_bounds))

            #now figure out which peak is the one we wanted originally
            q = np.argwhere(peaks==xguess)[0]
            if varbeta:
                fitted_pos = popt[q*4]
                if return_all_pars:
                    fitted_sigma = popt[q*4+1]
                    fitted_amp = popt[q*4+2]
                    fitted_beta = popt[q*4+3]
            else:
                fitted_pos = popt[q*3]
                if return_all_pars:
                    fitted_sigma = popt[q*3+1]
                    fitted_amp = popt[q*3+2]


        print('haehaehae')
        #make sure we actually found a good peak
        if abs(fitted_pos - xguess) >= 2.:
            line_pos_fitted.append(xguess)
            if return_qualflag:
                qualflag.append(0)
            if return_all_pars:
                line_sigma_fitted.append(fitted_sigma)
                line_amp_fitted.append(fitted_amp)
                if varbeta:
                    line_beta_fitted.append(fitted_beta)
        else:
            line_pos_fitted.append(fitted_pos)
            if return_qualflag:
                qualflag.append(1)
            if return_all_pars:
                line_sigma_fitted.append(fitted_sigma)
                line_amp_fitted.append(fitted_amp)
                if varbeta:
                    line_beta_fitted.append(fitted_beta)
        ################################################################################################################################################################################
        
#         ################################################################################################################################################################################
#         #METHOD 2 (using lmfit) (NOTE THAT THE TWO METHODS HAVE different amplitudes for the Gaussian b/c of different normalization, but we are only interested in the position)
#         #xguess = int(xguess)
#         gm = GaussianModel()
#         gm_pars = gm.guess(data[xguess - fitwidth:xguess + fitwidth], xx[xguess - fitwidth:xguess + fitwidth])
#         gm_fit_result = gm.fit(data[xguess - fitwidth:xguess + fitwidth], gm_pars, x=xx[xguess - fitwidth:xguess + fitwidth])
#         
#         #make sure we actually found the correct peak
#         if gm_fit_result.ier not in (1,2,3,4):     #if this is any other value it means the fit did not converge
#         #if gm_fit_result.ier > 4:   
#             # gm_fit_result.plot()
#             # plt.show()
#             thar_pos_fitted.append(xguess)
#         elif abs(gm_fit_result.best_values['center'] - xguess) > 2.:
#             thar_pos_fitted.append(xguess)
#         else:
#             thar_pos_fitted.append(gm_fit_result.best_values['center'])
#         ################################################################################################################################################################################
    
    if verbose:    
        plt.figure()
        plt.plot(xx,data)
        #plt.vlines(thar_pos_guess, 0, np.max(data))
        plt.vlines(line_pos_fitted, 0, np.max(data) * 1.2, color='g', linestyles='dotted')
        plt.show()
    
    if timit:
        print('Time taken for fitting emission lines: '+str(time.time() - start_time)+' seconds...')
    
    if return_all_pars:
        if varbeta:
            if return_qualflag:
                return np.array(line_pos_fitted), np.array(line_sigma_fitted), np.array(line_amp_fitted), np.array(line_beta_fitted), np.array(qualflag)
            else:
                return np.array(line_pos_fitted), np.array(line_sigma_fitted), np.array(line_amp_fitted), np.array(line_beta_fitted)
        else:
            if return_qualflag:
                return np.array(line_pos_fitted), np.array(line_sigma_fitted), np.array(line_amp_fitted), np.array(qualflag)
            else:
                return np.array(line_pos_fitted), np.array(line_sigma_fitted), np.array(line_amp_fitted)
    else:
        if return_qualflag:
            return np.array(line_pos_fitted), np.array(qualflag)
        else:
            return np.array(line_pos_fitted)





def fit_emission_lines_lmfit(data, fitwidth=None, thresh=5000., bgthresh=2000., maxthresh=None, laser=False, model='gauss', no_weights=False,
                             offset=False, return_all_pars=False, return_qualflag=False, return_stats=False, return_full=False, timit=False, debug_level=0):
    """
    This routine identifies and fits emission lines in a 1-dim spectrum (ie generally speaking it finds and fits peaks in a 1dim array), using the LMFIT package.
    Detection threshold, background threshold, and maximum threshold can be provided as keyword parameters. Different models for the peak-like function can be selected.

    INPUT:
    'data'   : one-dimensional data array

    KEYWORD PARAMETERS:
    'fitwidth'         : range around the identified peaks to be used for the peak-fitting 
                         (if None, it will automatically be determined by the minimum spacing between peaks - this option should only be used for LFC spectra, won't work properly for ThAr or ThXe)
    'thresh'           : minimum height of peak in the FILTERED data to be considered a good peak
    'bgthresh'         : minimum height of peak in the FILTERED data to be considered a real (but not good) peak
    'maxthresh'        : maximum height of peak in the FILTERED data to be included (ie you can exclude saturated peaks etc)
    'laser'            : boolean - is this a LFC spectrum? (if set to true, there is no check for blended lines)
    'model'            : peak-like function model to be used for the fitting; valid options are (check code below for exact names):
                         [gaussian, gauss-like, double gaussian, lorentzian, voigt, pseudo-voigt, offset pseudo-voigt, offset "pseudo-gausslike", 
                         moffat, pearson7, students, breit-wigner, lognormal, damped oscillator, damped harmonic oscillator, exp gauss, skew gauss, donaich] 
    'no_weights'       : boolean - set to TRUE if you DO NOT WANT to use weights (not recommended)                        
    'offset'           : boolean - do you want to estimate and subtract an offset before fitting?                        
    'return_all_pars'  : boolean - do you want to return all fit parameters?
    'return_qualflag'  : boolean - do you want to return a quality flag for each line fit?
    'return_stats'     : boolean - do you want to return goodness of fit statistics (ie AIC, BIC, CHISQ and REDCHISQ)?
    'return_full'      : boolean - do you want to return the full "fit_result" class for each line? (WARNING: this ONLY returns the line positions and the full 'fit_result' classes,
                         ie it overwrites the 'return_all_pars' / 'return_qualflag' / 'return_stats' keywords)
    'debug_level'      : boolean - for debugging...
    'timit'            : boolean - do you want to clock execution run-time?    
    
    OUTPUT:
    'full_results'     : the full results, ie the instance of the LMFIT 'ModelResult' class (if 'return_full' is set to TRUE)
    -------------------
    'line_pos_fitted'  : fitted line positions
    'allpars'          : (only if 'return_all_pars' is set to TRUE)
    'qualflag'         : quality flag (1=good, 0=bad) for each line (only if 'return_qualflag' is set to TRUE)
    'stats'            : (only if 'return_stats' is set to TRUE)
    
    TODO:
    (1) make a relative threshold criterion
    """

    if timit:
        start_time = time.time()

    xx = np.arange(len(data))

    if model.lower() == 'gausslike':
        mod = Model(fibmodel_with_amp)
    if model.lower() in ('double', 'dbl', 'dg', 'dbl_gauss', 'double_gaussian'):
        mod1 = GaussianModel(prefix='first_')
        mod2 = GaussianModel(prefix='second_')
        mod = mod1 + mod2
    if model.lower() in ('gauss', 'gaussian'):
        mod = GaussianModel()
    if model.lower() in ('lorentz', 'lorentzian'):
        mod = LorentzianModel()
    if model.lower() == 'voigt':
        mod = VoigtModel()
    if model.lower() in ('pseudo', 'pseudovoigt'):
        mod = PseudoVoigtModel()
    if model.lower() == 'offset_pseudo':
        gmod = GaussianModel(prefix='G_')
        lmod = LorentzianModel(prefix='L_')
        mod = gmod + lmod
    if model.lower() == 'offset_pseudo_gausslike':
        mod = Model(offset_pseudo_gausslike)
    if model.lower() == 'moffat':
        mod = MoffatModel()
    if model.lower() in ('pearson', 'pearson7'):
        mod = Pearson7Model(nan_policy='omit')
    if model.lower() in ('student', 'students', 'studentst'):
        mod = StudentsTModel()
    if model.lower() == 'breitwigner':
        mod = BreitWignerModel()
    if model.lower() == 'lognormal':
        mod = LognormalModel()
    if model.lower() == 'dampedosc':
        mod = DampedOscillatorModel()
    if model.lower() == 'dampedharmosc':
        mod = DampedHarmonicOscillatorModel()
    if model.lower() == 'expgauss':
        mod = ExponentialGaussianModel(nan_policy='omit')
    if model.lower() == 'skewgauss':
        mod = SkewedGaussianModel()
    if model.lower() == 'donaich':
        mod = DonaichModel()


    # find rough peak locations
    goodpeaks, mostpeaks, allpeaks = find_suitable_peaks(data, thresh=thresh, bgthresh=bgthresh, maxthresh=maxthresh, remove_bg=True)

    #determine fit-window if not explicitly given
    if fitwidth is None:
        fitwidth = np.min(np.diff(goodpeaks))//2
    
    print('Fitting ' + str(len(goodpeaks)) + ' emission lines...')

    #prepare output variables
    line_pos_fitted = []
    if return_full:
        full_results = []
    else:
        if return_qualflag:
            qualflag = []
        if return_all_pars:
            allpars = []
        if return_stats:
            stats = []
    
    
    #loop over all lines
    for xguess in goodpeaks:
        
        if debug_level >= 2:
            print('xguess = ',xguess)

        if not laser:
            # check if there are any other peaks in the vicinity of the peak in question (exclude the peak itself)
            checkrange = np.r_[xx[np.max([0, xguess - 2 * fitwidth]): xguess], xx[xguess + 1: np.min([xguess + 2 * fitwidth + 1, len(data) - 1])]]
            peaks = np.r_[xguess]
            # while len((set(checkrange) & set(allpeaks))) > 0:    THE RESULTS ARE COMPARABLE, BUT USING MOSTPEAKS IS MUCH FASTER
            while len((set(checkrange) & set(mostpeaks))) > 0:
                # where are the other peaks?
                # other_peaks = np.intersect1d(checkrange, allpeaks)    THE RESULTS ARE COMPARABLE, BUT USING MOSTPEAKS IS MUCH FASTER
                other_peaks = np.intersect1d(checkrange, mostpeaks)
                peaks = np.sort(np.r_[peaks, other_peaks])
                # define new checkrange
                checkrange = xx[np.max([0, peaks[0] - 2 * fitwidth]): np.min([peaks[-1] + 2 * fitwidth + 1, len(data) - 1])]
                dum = np.in1d(checkrange, peaks)
                checkrange = checkrange[~dum]
        else:
            peaks = np.r_[xguess]

        npeaks = len(peaks)
        xrange = xx[np.max([0, peaks[0] - fitwidth]): np.min([peaks[-1] + fitwidth + 1, len(data) - 1])]   # this should satisfy: len(xrange) == len(checkrange) - 2*fitwidth + len(peaks)


        ################################################################################################################################################################################
        #using LMFIT package:

        if npeaks == 1:

            #estimate and subtract an offset if desired
            if offset:
                off = np.mean([data[xrange][0],data[xrange][1],data[xrange][-1],data[xrange][-2]])
            else:
                off = 0.
            fitdata = data[xrange] - off

            # create instance of Parameters-class needed for fitting with LMFIT
            if model.lower() in ('gausslike', 'offset_pseudo', 'offset_pseudo_gausslike', 'double', 'dbl', 'dg', 'dbl_gauss', 'double_gaussian'):
                parms = lmfit.Parameters()
            else:
                parms = mod.guess(fitdata, x=xrange)

            # fill/tweak initial Parameters() instance
            if model.lower() in ('gauss', 'gaussian'):
                parms['amplitude'].set(min=0.)
                parms['sigma'].set(min=0.)
            if model.lower() in ('double', 'dbl', 'dg', 'dbl_gauss', 'double_gaussian'):
                parms = mod1.guess(fitdata, x=xrange)
                parms.update(mod2.guess(fitdata, x=xrange))
                parms['second_amplitude'].set(0., vary=True)
                parms['first_center'].set(min=xguess - 3, max=xguess + 3)
                parms['second_center'].set(min=xguess - 3, max=xguess + 3)
            if model.lower() == 'gausslike':
                gmod = GaussianModel()
                gparms = gmod.guess(fitdata,xrange)
                parms.add('mu', xguess, min=xguess - 3, max=xguess + 3)
                parms.add('sigma', gparms['sigma'].value, min=0.2)
                parms.add('amp', data[xguess], min=0.)
                parms.add('beta', 2., min=1., max=4.)
            if model.lower() == 'moffat':
                parms['amplitude'].set(min=0.)
                parms['sigma'].set(min=0.)
                parms['beta'].set(min=0.)
            if model.lower() in ('pseudo', 'pseudovoigt'):
                # parms['fraction'].set(0.5,min=0.,max=1.)
                parms['amplitude'].set(min=0.)
                parms['sigma'].set(min=0.)
            if model.lower() == 'lognormal':
                #parms['sigma'].set(value=1e-4, vary=True, expr='')
                parms['center'].set(value=np.log(xguess), vary=True, expr='', min=0., max=8.5)
                parms['amplitude'].set(data[xguess], vary=True, min=0., expr='')
            # if model.lower() == 'dampedosc':
            #     parms['sigma'].set(1e-4, vary=True, expr='')
            #     parms['amplitude'].set(1e-4, vary=True, min=0., expr='')
            if model.lower() == 'offset_pseudo':
                parms = gmod.guess(fitdata, x=xrange)
                parms.update(lmod.guess(fitdata, x=xrange))
                parms['G_amplitude'].set(parms['G_amplitude'] / 2., min=0., vary=True)
                parms['L_amplitude'].set(parms['L_amplitude'] / 2., min=0., vary=True)
                parms['G_center'].set(min=xguess - 3, max=xguess + 3)
                parms['L_center'].set(min=xguess - 3, max=xguess + 3)
            if model.lower() == 'skewgauss':
                parms['amplitude'].set(min=0.)
                parms['sigma'].set(min=0.)
            if model.lower() == 'offset_pseudo_gausslike':
                gmod = GaussianModel(prefix='G_')
                lmod = LorentzianModel(prefix='L_')
                parms = gmod.guess(fitdata, x=xrange)
                parms.update(lmod.guess(fitdata, x=xrange))
                parms['G_amplitude'].set(parms['G_amplitude'] / 2., min=0., vary=True)
                parms['L_amplitude'].set(parms['L_amplitude'] / 2., min=0., vary=True)
                parms['G_center'].set(min=xguess - 3, max=xguess + 3)
                parms['L_center'].set(min=xguess - 3, max=xguess + 3)
                parms.add('beta', 2., min=1., max=4.)

            
            #perform the actual fit (get weights from array before offset subtraction)
            if not no_weights:
                fit_result = mod.fit(fitdata, parms, x=xrange, weights=np.sqrt(data[xrange]))
            else:
                fit_result = mod.fit(fitdata, parms, x=xrange)

            #goodness of fit statistics
            fitstats = {}
            fitstats['aic'] = fit_result.aic
            fitstats['bic'] = fit_result.bic
            fitstats['chi2'] = fit_result.chisqr
            fitstats['chi2red'] = fit_result.redchi 

            #fill bestpos variable depending on model used
            if model.lower() not in ('offset_pseudo', 'offset_pseudo_gausslike', 'double', 'dbl', 'dg', 'dbl_gauss', 'double_gaussian'):
                try:
                    bestpos = fit_result.best_values['center']
                except:
                    bestpos = fit_result.best_values['mu']
            elif model.lower() in ('offset_pseudo', 'offset_pseudo_gausslike'):
                bestpos = np.array((fit_result.best_values['G_center'], fit_result.best_values['L_center']))
            elif model.lower() in ('double', 'dbl', 'dg', 'dbl_gauss', 'double_gaussian'):
                bestpos = np.array((fit_result.best_values['first_center'], fit_result.best_values['second_center']))
            else:
                print('ERROR!!! No valid model defined!!!')
                return

            #make sure we actually found a good / the correct peak
            if fit_result.ier not in (1,2,3,4):     #if this is any other value it means the fit did not converge
            #if fit_result.ier > 4:
                fit_result.plot()
                plt.show()
                print('WARNING: Bad fit encountered!')
                choice = raw_input('Do you wish to continue? [y/n]')
                #choice = 'y'
                if choice.lower() == 'y':
                    line_pos_fitted.append(xguess)
                    if return_full:
                        full_results.append(fit_result)
                    else:
                        if return_qualflag:
                            qualflag.append(0)
                        if return_all_pars:
                            allpars.append(fit_result.params)
                        if return_stats:
                            stats.append(fitstats)    
                else:
                    return
            elif np.all(abs(bestpos - xguess) >= 2.):
                fit_result.plot()
                plt.show()
                print('WARNING: Bad fit encountered!')
                choice = raw_input('Do you wish to continue? [y/n]')
                #choice = y
                if choice.lower() == 'y':
                    line_pos_fitted.append(xguess)
                    if return_full:
                        full_results.append(fit_result)
                    else:
                        if return_qualflag:
                            qualflag.append(0)
                        if return_all_pars:
                            allpars.append(fit_result.params)
                        if return_stats:
                            stats.append(fitstats)
                else:
                    return
            else:
                line_pos_fitted.append(bestpos)
                if return_full:
                    full_results.append(fit_result)
                else:
                    if return_qualflag:
                        qualflag.append(0)
                    if return_all_pars:
                        allpars.append(fit_result.params)
                    if return_stats:
                        stats.append(fitstats)

# #             # # #this way you can quickly plot the indidvidual fits for debugging
#             plot_osf = 10
#             plot_os_grid = np.linspace(xrange[0], xrange[-1], plot_osf * (len(xrange) - 1) + 1)
#             guessmodel = mod.eval(fit_result.init_params, x=plot_os_grid)
#             bestmodel = mod.eval(fit_result.params, x=plot_os_grid)
#             plt.figure()
#             plt.title('model = ' + model.title())
#             plt.xlabel('pixel number (dispersion direction)')
#             plt.plot(xrange, fitdata, 'bo')
#             plt.plot(plot_os_grid, guessmodel, 'k--', label='initial guess')
#             plt.plot(plot_os_grid, bestmodel, 'r-', label='best-fit model')
#             plt.legend()


        else:
            print('ERROR: multi-peak fitting with LMFIT has not been implemented yet')
            return
        ################################################################################################################################################################################

    if debug_level >= 1:
        plt.figure()
        plt.plot(xx, data)
        # plt.vlines(thar_pos_guess, 0, np.max(data))
        plt.vlines(line_pos_fitted, 0, np.max(data) * 1.2, color='g', linestyles='dotted')
        plt.show()

    if timit:
        print('Time taken for fitting emission lines: ' + str(time.time() - start_time) + ' seconds...')

    #return results
    if return_full:
        return full_results 
    else:
        if return_stats:
            if return_all_pars:
                if return_qualflag:
                    return np.array(line_pos_fitted), allpars, np.array(qualflag), stats
                else:
                    return np.array(line_pos_fitted), allpars, stats
            else:
                if return_qualflag:
                    return np.array(line_pos_fitted), np.array(qualflag), stats
                else:
                    return np.array(line_pos_fitted), stats
        else:
            if return_all_pars:
                if return_qualflag:
                    return np.array(line_pos_fitted), allpars, np.array(qualflag)
                else:
                    return np.array(line_pos_fitted), allpars
            else:
                if return_qualflag:
                    return np.array(line_pos_fitted), np.array(qualflag)
                else:
                    return np.array(line_pos_fitted)





def get_wavelength_solution_from_thorium(thflux, poly_deg=5, polytype='chebyshev', savetable=False, return_full=True, timit=False):
    """ 
    INPUT:
    'thflux'           : extracted 1-dim thorium / laser-only image (initial tests used this one : '/Users/christoph/OneDrive - UNSW/veloce_spectra/reduced/tests/...')
    'poly_deg'         : the order of the polynomials to use in the fit (for both dimensions)
    'polytype'         : either 'polynomial', 'legendre', or 'chebyshev' (default) are accepted 
    'return_full'      : boolean - if TRUE, then the wavelength solution for each pixel for each order is returned; otherwise just the set of coefficients that describe it
    'saveplots'        : boolean - do you want to create plots for each order? 
    'savetable'        : boolean - if TRUE, then an output file is created, containing a summary of all lines used in the fit, and details about the fit
    'timit'            : time it...
    
    OUTPUT:
    'p'      : functional form of the coefficients that describe the wavelength solution
    'p_wl'   : wavelength solution for each pixel for each order (n_ord x n_pix numpy-array) 
    
    TODO:
    include order 40 (m=65) as well (just 2 lines!?!?!?)
    figure out how to properly use weights here (ie what should we use as weights?)
    """
    
    if timit:
        start_time = time.time()
    
#     #read in pre-defined thresholds (needed for line identification)
#     thresholds = np.load('/Users/christoph/OneDrive - UNSW/linelists/AAT_folder/thresholds.npy').item()
    
    #prepare arrays for fitting
    x = np.array([])
    order = np.array([])
    m_order = np.array([])
    wl = np.array([])

    for ord in sorted(thflux.keys()):    #make sure there are enough lines in every order
    #for ord in sorted(thflux.keys())[:-1]:    #don't have enough emission lines in order 40
        ordnum = ord[-2:]
        #m = 105 - int(ordnum)   #original orientation of the spectra
        m = 64 + int(ordnum)    #'correct' orientation of the spectra
        print('OK, fitting '+ord+'   (m = '+str(m)+')')
        
        data = thflux[ord]        
        xx = np.arange(len(data))
        
        #adjust thresholds here
        fitted_line_pos = fit_emission_lines(data,return_all_pars=False,varbeta=False,timit=False,verbose=False,thresh=thresholds['thresh'][ord],bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord])
        #fitted_line_pos = fit_emission_lines_lmfit(data, fitwidth=4, model='gausslike', return_all_pars=False,thresh=thresholds['thresh'][ord],
        #                                          bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord]) 
        goodpeaks,mostpeaks,allpeaks = find_suitable_peaks(data,thresh=thresholds['thresh'][ord],bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord])    
        
        #########################################################################################################################################
        #THIS IS THE STEP THAT NEEDS TO BE AUTOMATED
        line_number, refwlord = readcol('/Users/christoph/OneDrive - UNSW/linelists/AAT_folder/ThAr_linelist_order_'+ordnum+'.dat',fsep=';',twod=False)        
        #this tells us which peaks we have known wavelengths for
        mask_order = np.load('/Users/christoph/OneDrive - UNSW/linelists/posmasks/mask_order'+ordnum+'.npy')
        xord = fitted_line_pos[mask_order]
        
#         #ie we want to do sth like this:
#         guess_wls = get_wavelengths_for_peaks(fitted_line_pos)   # ie get a rough guess for thew wavelengths of the lines we found
#         refwlord = identify_lines(guess_wls, linelist)              # ie find the corresponding "true" wavelengths
        #########################################################################################################################################
        
        #fill arrays for fitting
        x = np.append(x, xord)
        order = np.append(order, np.repeat(int(ordnum), len(xord)))
        #m_order = np.append(m_order, np.repeat(105 - int(ordnum), len(xord)))
        m_order = np.append(m_order, np.repeat(64 + int(ordnum), len(xord)))
        wl = np.append(wl, refwlord)
        
    
    
    #now re-normalize arrays to [-1,+1]
    x_norm = (x / ((len(data)-1)/2.)) - 1.
    order_norm = ((order-1) / ((len(thflux)-1)/2.)) - 1.       
           
    #call the fitting routine
    p = fit_poly_surface_2D(x_norm, order_norm, wl, weights=None, polytype=polytype, poly_deg=poly_deg, debug_level=0)    
    
#     if return_all_pars:
#         return dispsol,fitshapes
#     else:
#         return dispsol

    if savetable:
        now = datetime.datetime.now()
        model_wl = p(x_norm, order_norm)
        resid = wl - model_wl
        outfn = '/Users/christoph/OneDrive - UNSW/linelists/AAT_folder/lines_used_in_fit_as_of_'+str(now)[:10]+'.dat'
        outfn = open(outfn, 'w')
        outfn.write('line number   order_number   physical_order_number     pixel      reference_wl[A]   model_wl[A]    residuals[A]\n')
        outfn.write('=====================================================================================================================\n')
        for i in range(len(x)):
                outfn.write("   %3d             %2d                 %3d           %11.6f     %11.6f     %11.6f     %9.6f\n" %(i+1, order[i], m_order[i], x[i], wl[i], model_wl[i], resid[i]))
        outfn.close()
              

    if return_full:
        #xx = np.arange(len(data))            # already done above
        xxn = (xx / ((len(xx)-1)/2.)) - 1.
        oo = np.arange(1,len(thflux))
        oon = ((oo-1) / ((len(thflux)-1)/2.)) - 1.   
        X,O = np.meshgrid(xxn,oon)
        p_wl = p(X,O)

    
    if timit:
        print('Time elapsed: ',time.time() - start_time,' seconds')
        

    if return_full:
        return p,p_wl
    else:
        return p





def get_wl(p,xx,yy):
    """
    Get the wavelength for "full" 2-dim grid of x-pixel number and order number, or x-pixel number and y-pixel number, 
    depending on how 'p' was created, ie either lambda(x,y) or lambda (x,ord).
    
    Example: for the full Veloce Rosso chip (nx,ny) = (4112,4096), call it like so:
             wl = get_wl(p, np.arange(4112), np.arange(4096))
             OR for (nx,n_ord) = (4112,40)
             wl = get_wl(p, np.arange(4112), np.arange(1,41))   #note that orders range from 1...40 (not from 0...39)
    
    INPUT:
    'p'     : the coefficients describing the 2-dim polynomial surface fit from "fit_poly_surface_2D"   
    'xx'    : x-coordinate(s) at which you want to evaluate the wavelength solution
    'yy'    : y-coordinate(s) at which you want to evaluate the wavelength solution
    
    OUTPUT:
    p_wl'  : wavelength solution for each pair of coordinates
    """
    
    #re-normalize arrays to [-1,+1]
    print('WRONG!!!!!')
    xxn = np.linspace(-1, 1, len(xx))
    yyn = np.linspace(-1, 1, len(yy))
    #make 2D grids of coordinates
    X,Y = np.meshgrid(xxn,yyn)
    #actually calculate the wavelengths from the polynomial coefficients
    p_wl = p(X,Y)
    
    return p_wl





def get_wavelength_solution_labtests(thflux, thflux2, poly_deg=5, polytype='chebyshev', savetable=False, return_full=True, saveplots=False, timit=False):
    """ 
    INPUT:
    'thflux'           : extracted 1-dim thorium / laser-only image (initial tests used this one : '/Users/christoph/OneDrive - UNSW/veloce_spectra/reduced/tests/...')
    'poly_deg'         : the order of the polynomials to use in the fit (for both dimensions)
    'polytype'         : either 'polynomial', 'legendre', or 'chebyshev' (default) are accepted 
    'return_full'      : boolean - if TRUE, then the wavelength solution for each pixel for each order is returned; otherwise just the set of coefficients that describe it
    'saveplots'        : boolean - do you want to create plots for each order? 
    'savetable'        : boolean - if TRUE, then an output file is created, containing a summary of all lines used in the fit, and details about the fit
    'timit'            : time it...
    
    OUTPUT:
    EITHER
    'p'      : functional form of the coefficients that describe the wavelength solution
    OR 
    'p_wl'   : wavelength solution for each pixel for each order (n_ord x n_pix numpy-array) 
    (selection between outputs is controlled by the 'return_full' keyword)
    
    TODO:
    clean-up the 2-version thing about which extracted spectrum to use (thflux / thflux2)
    include order 40 (m=65) as well (just 2 lines!?!?!?)
    figure out how to properly use weights here (ie what should we use as weights?)
    """
    
    if timit:
        start_time = time.time()
    
    #read in pre-defined thresholds (needed for line identification)
    thresholds = np.load('/Users/christoph/OneDrive - UNSW/linelists/AAT_folder/thresholds.npy').item()
    
    #wavelength solution from Zemax as a reference
    if saveplots:
        zemax_dispsol = np.load('/Users/christoph/OneDrive - UNSW/dispsol/mean_dispsol_by_orders_from_zemax.npy').item()
    
    #prepare arrays for fitting
    x = np.array([])
    order = np.array([])
    m_order = np.array([])
    wl = np.array([])

    for ord in sorted(thflux.keys())[:-1]:    #don't have enough emission lines in order 40
        ordnum = ord[-2:]
        m = 105 - int(ordnum)   #original orientation of the spectra
        #m = 64 + int(ordnum)    #'correct' orientation of the spectra
        print('OK, fitting '+ord+'   (m = '+str(m)+')')
        coll = thresholds['collapsed'][ord]
        if coll:
            data = thflux2[ord]
        else:
            data = thflux[ord]
        
        xx = np.arange(len(data))
        
#         if return_all_pars:
#             fitted_line_pos,fitted_line_sigma,fitted_line_amp = fit_emission_lines(data,return_all_pars=return_all_pars,varbeta=False,timit=False,verbose=False,thresh=thresholds['thresh'][ord],bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord])
#         else:
#             fitted_line_pos = fit_emission_lines(data,return_all_pars=return_all_pars,varbeta=False,timit=False,verbose=False,thresh=thresholds['thresh'][ord],bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord])
        fitted_line_pos = fit_emission_lines(data,return_all_pars=False,varbeta=False,timit=False,verbose=False,thresh=thresholds['thresh'][ord],bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord])
        #fitted_line_pos = fit_emission_lines_lmfit(data, fitwidth=4, model='gausslike', return_all_pars=False,thresh=thresholds['thresh'][ord],
        #                                          bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord]) 
        goodpeaks,mostpeaks,allpeaks = find_suitable_peaks(data,thresh=thresholds['thresh'][ord],bgthresh=thresholds['bgthresh'][ord],maxthresh=thresholds['maxthresh'][ord])    
        
        #########################################################################################################################################
        #THIS IS THE STEP THAT NEEDS TO BE AUTOMATED
        line_number, refwlord = readcol('/Users/christoph/OneDrive - UNSW/linelists/AAT_folder/ThAr_linelist_order_'+ordnum+'.dat',fsep=';',twod=False)
        #lam = refwlord.copy()  
        #wl_ref[ord] = lam
        
        #this tells us which peaks we have known wavelengths for
        mask_order = np.load('/Users/christoph/OneDrive - UNSW/linelists/posmasks/mask_order'+ordnum+'.npy')
        xord = fitted_line_pos[mask_order]
        #########################################################################################################################################
        
        #stupid python!?!?!?
        if ordnum == '30':
            xord = np.array([xord[0][0],xord[1][0],xord[2],xord[3]])
        
        if saveplots:
            zemax_wl = 10. * zemax_dispsol['order'+str(m)]['model'](xx[::-1])
        
        #fill arrays for fitting
        x = np.append(x, xord)
        order = np.append(order, np.repeat(int(ordnum), len(xord)))
        m_order = np.append(m_order, np.repeat(105 - int(ordnum), len(xord)))
        wl = np.append(wl, refwlord)
        
        
#         #perform the fit
#         fitdegpol = degpol
#         while fitdegpol > len(x)/2:
#             fitdegpol -= 1
#         if fitdegpol < 2:
#             fitdegpol = 2
#         thar_fit = np.poly1d(np.polyfit(x, lam, fitdegpol))
#         dispsol[ord] = thar_fit
#         if return_all_pars:
#             fitshapes[ord] = {}
#             fitshapes[ord]['x'] = x
#             fitshapes[ord]['y'] = P_id[ord](x)
#             fitshapes[ord]['FWHM'] = 2.*np.sqrt(2.*np.log(2.)) * fitted_line_sigma[mask_order]
#         
#         #calculate RMS of residuals in terms of RV
#         resid = thar_fit(x) - lam
#         rv_resid = 3e8 * resid / lam
#         rms = np.std(rv_resid)
#         
#         if saveplots:
#             #first figure: lambda vs x with fit and zemax dispsol
#             fig1 = plt.figure()
#             plt.plot(x,lam,'bo')
#             plt.plot(xx,thar_fit(xx),'g',label='fitted')
#             plt.plot(xx,zemax_wl,'r--',label='Zemax')
#             plt.title('Order '+str(m))
#             plt.xlabel('pixel number')
#             plt.ylabel(ur'wavelength [\u00c5]')
#             plt.text(3000,thar_fit(500),'n_lines = '+str(len(x)))
#             plt.text(3000,thar_fit(350),'deg_pol = '+str(fitdegpol))
#             plt.text(3000,thar_fit(100),'RMS = '+str(round(rms, 1))+' m/s')
#             plt.legend()
#             plt.savefig('/Users/christoph/OneDrive - UNSW/dispsol/lab_tests/fit_to_order_'+ordnum+'.pdf')
#             plt.close(fig1)
#             
#             #second figure: spectrum vs fitted dispsol
#             fig2 = plt.figure()
#             plt.plot(thar_fit(xx),data)
#             #plt.scatter(thar_fit(x), data[x.astype(int)], marker='o', color='r', s=40)
#             plt.scatter(thar_fit(goodpeaks), data[goodpeaks], marker='o', color='r', s=30)
#             plt.title('Order '+str(m))
#             plt.xlabel(ur'wavelength [\u00c5]')
#             plt.ylabel('counts')
#             plt.savefig('/Users/christoph/OneDrive - UNSW/dispsol/lab_tests/ThAr_order_'+ordnum+'.pdf')
#             plt.close(fig2)
    
    
    #re-normalize arrays to [-1,+1]
    x_norm = (x / ((len(data)-1)/2.)) - 1.
    order_norm = ((order-1) / (38./2.)) - 1.       #TEMP, TODO, FUGANDA, PLEASE FIX ME!!!!!
    #order_norm = ((m-1) / ((len(P_id)-1)/2.)) - 1.
           
    #call the fitting routine
    p = fit_poly_surface_2D(x_norm, order_norm, wl, weights=None, polytype=polytype, poly_deg=poly_deg, debug_level=0)    
    
#     if return_all_pars:
#         return dispsol,fitshapes
#     else:
#         return dispsol

    if savetable:
        now = datetime.datetime.now()
        model_wl = p(x_norm, order_norm)
        resid = wl - model_wl
        outfn = '/Users/christoph/OneDrive - UNSW/linelists/AAT_folder/lines_used_in_fit_as_of_'+str(now)[:10]+'.dat'
        outfn = open(outfn, 'w')
        outfn.write('line number   order_number   physical_order_number     pixel      reference_wl[A]   model_wl[A]    residuals[A]\n')
        outfn.write('=====================================================================================================================\n')
        for i in range(len(x)):
                outfn.write("   %3d             %2d                 %3d           %11.6f     %11.6f     %11.6f     %9.6f\n" %(i+1, order[i], m_order[i], x[i], wl[i], model_wl[i], resid[i]))
        outfn.close()
              

    if return_full:
        #xx = np.arange(4112)            # already done above
        xxn = (xx / ((len(xx)-1)/2.)) - 1.
        oo = np.arange(1,len(thflux))
        oon = ((oo-1) / (38./2.)) - 1.        #TEMP, TODO, FUGANDA, PLEASE FIX ME!!!!!
        #oon = ((oo-1) / ((len(thflux)-1)/2.)) - 1.   
        X,O = np.meshgrid(xxn,oon)
        p_wl = p(X,O)

    
    if timit:
        print('Time elapsed: ',time.time() - start_time,' seconds')
        

    if return_full:
        return p,p_wl
    else:
        return p





def get_simu_dispsol(fibre=None, path='/Users/christoph/OneDrive - UNSW/dispsol/', npix=4096):
    """
    Get wavelength solution from Veloce Zemax file for a given fibre (1...28)
    WARNING: If no fibre is given, a mean wavelength solution across all orders is calculated!!!
    """
    #read dispersion solution from file
    if fibre is None:
        dispsol = np.load(path + 'mean_dispsol_by_orders_from_zemax.npy').item()
        orders = dispsol.keys()
    else:
        dbf = np.load(path + 'dispsol_by_fibres_from_zemax.npy').item()
        orders = dbf['fiber_1'].keys()
     
    #read extracted spectrum from files (obviously this needs to be improved)
    xx = np.arange(npix)
     
    #this is so as to match the order number with the physical order number (66 <= m <= 108)
    # order01 corresponds to m=66
    # order43 corresponds to m=108
    wl = {}
    for o in orders:
        m = o[5:]
        ordnum = str(int(m)-65).zfill(2)
        if fibre is None:
            wl['order_'+ordnum] = dispsol['order'+m]['model'](xx)
        else:
            wl['order_'+ordnum] = dbf['fiber_'+str(fibre)][o]['fitparms'](xx)
 
    return wl






# OLDER CODE SNIPPETS
#
# THE following routine is now called fit_poly_surface_2D and lives in "helper_functions"
# def fit_dispsol_2D(x_norm, ord_norm, WL, weights=None, polytype = 'chebyshev', poly_deg=5, debug_level=0):
#     """
#     Calculate 2D polynomial wavelength fit to normalized x and order values.
#     Wrapper function for using the astropy fitting library.
# 
#     'x_norm'   : x-values (pixels) of all the lines, re-normalized to [-1,+1]
#     'ord_norm' : order numbers of all the lines, re-normalized to [-1,+1]
#     'WL'       : 
#     'polytype' : either '(p)olynomial' (default), '(l)egendre', or '(c)hebyshev' are accepted
#     """
#     
#     if polytype in ['Polynomial','polynomial','p','P']:
#         p_init = models.Polynomial2D(poly_deg)
#         if debug_level > 0:
#             print('OK, using standard polynomials...')
#     elif polytype in ['Chebyshev','chebyshev','c','C']:
#         p_init = models.Chebyshev2D(poly_deg,poly_deg)
#         if debug_level > 0:
#             print('OK, using Chebyshev polynomials...')
#     elif polytype in ['Legendre','legendre','l','L']:
#         p_init = models.Legendre2D(poly_deg,poly_deg)  
#         if debug_level > 0:
#             print('OK, using Legendre polynomials...')   
#     else:
#         print("ERROR: polytype not recognised ['(P)olynomial' / '(C)hebyshev' / '(L)egendre']")    
#         
#     fit_p = fitting.LevMarLSQFitter()  
# 
#     with warnings.catch_warnings():
#         # Ignore model linearity warning from the fitter
#         warnings.simplefilter('ignore')
#         p = fit_p(p_init, x_norm, ord_norm, WL, weights=weights)
# 
# 
# #     if debug_level > 0:
# #         plt.figure()
# #         index_include = np.array(weights, dtype=bool)
# #         plt.scatter(x_norm[index_include], WL[index_include], c=order_norm[index_include])
# #         plt.scatter(x_norm[np.logical_not(index_include)], WL[np.logical_not(index_include)], facecolors='none',
# #                     edgecolors='r')
# # 
# #         for x, o, oo, wl in zip(x_norm[index_include], order_norm[index_include], orders[index_include],
# #                                 WL[index_include]):
# #             plt.arrow(x, wl, 0, (p(x, o) / oo - wl) * 1000., head_width=0.00005, head_length=0.0001, width=0.00005)
# # 
# #         xi = np.linspace(min(x_norm[index_include]), max(x_norm[index_include]), 101)
# #         yi = np.linspace(min(order_norm[index_include]), max(order_norm[index_include]), 101)
# #         zi = griddata((x_norm[index_include], order_norm[index_include]),
# #                       ((WL[index_include] - p(x_norm[index_include], order_norm[index_include]) / orders[
# #                           index_include]) / np.mean(WL[index_include])) * 3e8,
# #                       (xi[None, :], yi[:, None]), method='linear')
# #         fig, ax = plt.subplots()
# #         ax.set_xlim((np.min(xi), np.max(xi)))
# #         ax.set_ylim((np.min(yi), np.max(yi)))
# #         ax.set_xlabel('Detector x normalized')
# #         ax.set_ylabel('order normalized')
# #         plt.title('Legendre Polynomial Degree: ' + str(poly_deg) + "\n" + "#pars: " + str(len(p.parameters)))
# # 
# #         im = ax.imshow(zi, interpolation='nearest', extent=[np.min(xi), np.max(xi), np.min(yi), np.max(yi)])
# # 
# #         divider = make_axes_locatable(ax)
# #         cax = divider.append_axes("right", size="5%", pad=0.05)
# # 
# #         cb = plt.colorbar(im, cax=cax)
# #         cb.set_label('RV deviation [m/s]')
# # 
# #         plt.tight_layout()
# 
#     return p


#
#
#
# thardata = np.load('/Users/christoph/OneDrive - UNSW/rvtest/thardata.npy').item()
# laserdata = np.load('/Users/christoph/OneDrive - UNSW/rvtest/laserdata.npy').item()
# thdata = thardata['flux']['order_01']
# ldata = laserdata['flux']['order_01']
#
#
# dispsol = np.load('/Users/christoph/OneDrive - UNSW/dispsol/lab_tests/thar_dispsol.npy').item()
# OR
# dispsol = np.load('/Users/christoph/OneDrive - UNSW/dispsol/lab_tests/thar_dispsol_2D.npy').item()
# fitshapes = np.load('/Users/christoph/OneDrive - UNSW/dispsol/lab_tests/fitshapes.npy').item()
# wavelengths = np.load('/Users/christoph/OneDrive - UNSW/dispsol/lab_tests/wavelengths.npy').item()
# wl_ref = np.load('/Users/christoph/OneDrive - UNSW/linelists/AAT_folder/wl_ref.npy').item()
# x = np.array([])
# m = np.array([])
# wl = np.array([])
# wl_ref_arr = np.array([])
# for ord in sorted(dispsol.keys()):
#     ordnum = int(ord[-2:])
#     x = np.append(x,fitshapes[ord]['x'])
#     order = np.append(order, np.repeat(ordnum,len(fitshapes[ord]['x'])))
#     wl = np.append(wl, dispsol[ord](fitshapes[ord]['x']))
#     wl_ref_arr = np.append(wl_ref_arr, wl_ref[ord])
# 
# 
# 
#
#
# 
# 
# 
# 
# 
# 
# 
# 
# # ###########################################
# # thar_refwlord01, thar_relintord01, flag = readcol('/Users/christoph/OneDrive - UNSW/linelists/test_thar_list_order_01.dat',fsep=';',twod=False)
# # thar_refwlord01 *= 1e3
# # refdata = {}
# # refdata['order_01'] = {}
# # refdata['order_01']['wl'] = thar_refwlord01[np.argwhere(flag == ' resolved')][::-1]          #note the array is turned around to match other arrays
# # refdata['order_01']['relint'] = thar_relintord01[np.argwhere(flag == ' resolved')][::-1]     #note the array is turned around to match other arrays
# # ###########################################
# 
# 
# 
# 
# 
# def get_dispsol_from_thar(thardata, refdata, deg_polynomial=5, timit=False, verbose=False):
#     """
#     NOT CURRENTLY USED
#     """
#     if timit:
#         start_time = time.time()
# 
#     thar_dispsol = {}
#     
#     #loop over all orders
#     #for ord in sorted(thardata['flux'].iterkeys()):
#     for ord in ['order_01']:
#     
#         if verbose:
#             print('Finding wavelength solution for '+str(ord))
#     
#         #find fitted x-positions of ThAr peaks
#         fitted_thar_pos, thar_qualflag = fit_emission_lines(thardata['flux'][ord], return_all_pars=False, return_qualflag=True, varbeta=False)
#         x = fitted_thar_pos.copy()
#         
#         #these are the theoretical wavelengths from the NIST linelists
#         lam = (refdata[ord]['wl']).flatten()
#         
#         #exclude some peaks as they are a blend of multiple lines: TODO: clean up
#         filt = np.ones(len(fitted_thar_pos),dtype='bool')
#         filt[[33,40,42,58,60]] = False
#         x = x[filt]
#         
#         #fit polynomial to lambda as a function of x
#         thar_fit = np.poly1d(np.polyfit(x, lam, deg_polynomial))
#         #save to output dictionary
#         thar_dispsol[ord] = thar_fit
# 
#     if timit:
#         print('Time taken for finding ThAr wavelength solution: '+str(time.time() - start_time)+' seconds...')
# 
#     return thar_dispsol
# 
# 
# # '''xxx'''
# # #################################################################
# # # the following is needed as input for "get_dispsol_from_laser" #
# # #################################################################
# # laser_ref_wl,laser_relint = readcol('/Users/christoph/OneDrive - UNSW/linelists/laser_linelist_25GHz.dat',fsep=';',twod=False)
# # laser_ref_wl *= 1e3
# # 
# # #wavelength solution from HDF file
# # #read dispersion solution from file
# # dispsol = np.load('/Users/christoph/OneDrive - UNSW/dispsol/mean_dispsol_by_orders_from_zemax.npy').item()
# # #read extracted spectrum from files (obviously this needs to be improved)
# # xx = np.arange(4096)
# # #this is so as to match the order number with the physical order number (66 <= m <= 108)
# # # order01 corresponds to m=66
# # # order43 corresponds to m=108
# # wl = {}
# # for ord in dispsol.keys():
# #     m = ord[5:]
# #     ordnum = str(int(m)-65).zfill(2)
# #     wl['order_'+ordnum] = dispsol['order'+m]['model'](xx)
#     
# 
# 
# 
# 
# def get_dispsol_from_laser(laserdata, laser_ref_wl, deg_polynomial=5, timit=False, verbose=False, return_stats=False, varbeta=False):
#     """
#     NOT CURRENTLY USED
#     """
#     
#     if timit:
#         start_time = time.time()
# 
#     if return_stats:
#         stats = {}
# 
#     #read in mask for fibre_01 (ie the Laser-comb fibre) from order_tracing as a first step in excluding low-flux regions
#     mask_01 = np.load('/Users/christoph/OneDrive - UNSW/fibre_profiles/masks/mask_01.npy').item()
# 
#     laser_dispsol = {}
#     
#     #loop over all orders
#     #order 43 does not work properly, as some laser peaks are missing!!!
#     for ord in sorted(laserdata['flux'].iterkeys())[:-1]:
# 
#         if verbose:
#             print('Finding wavelength solution for '+str(ord))
#         
#         #find fitted x-positions of ThAr peaks
#         data = laserdata['flux'][ord] * mask_01[ord]
#         goodpeaks,mostpeaks,allpeaks,first_mask,second_mask,third_mask = find_suitable_peaks(data,return_masks=True)    #from this we just want the masks this time (should be very fast)
#         #fitted_laser_pos, laser_qualflag = fit_emission_lines(data, laser=True, return_all_pars=False, return_qualflag=True, varbeta=varbeta)
#         if varbeta:
#             fitted_laser_pos, fitted_laser_sigma, fitted_laser_amp, fitted_laser_beta = fit_emission_lines(data, laser=True, return_all_pars=True, return_qualflag=False, varbeta=varbeta, timit=timit, verbose=verbose)
#         else:
#             fitted_laser_pos, fitted_laser_sigma, fitted_laser_amp = fit_emission_lines(data, laser=True, return_all_pars=True, return_qualflag=False, varbeta=varbeta, timit=timit, verbose=verbose)
#         x = fitted_laser_pos.copy()
#         #exclude the leftmost and rightmost peaks (nasty edge effects...)
# #         blue_cutoff = int(np.round((x[-1]+x[-2])/2.,0))
# #         red_cutoff = int(np.round((x[0]+x[1])/2.,0))
#         blue_cutoff = int(np.round(allpeaks[-1]+((allpeaks[-1] - allpeaks[-2])/2),0))
#         red_cutoff = int(np.round(allpeaks[0]-((allpeaks[1] - allpeaks[0])/2),0))
#         cond1 = (laser_ref_wl >= wl[ord][blue_cutoff])
#         cond2 = (laser_ref_wl <= wl[ord][red_cutoff])
#         #these are the theoretical wavelengths from the NIST linelists
#         lam = laser_ref_wl[np.logical_and(cond1,cond2)][::-1]
#         lam = lam[first_mask][second_mask][third_mask]
#         
#         #check if the number of lines found equals the number of lines from the line list
# #         if verbose:
# #             print(len(x),len(lam))
#         if len(x) != len(lam):
#             print('fuganda')
#             return 'fuganda'
#         
#         #fit polynomial to lambda as a function of x
#         laser_fit = np.poly1d(np.polyfit(x, lam, deg_polynomial))
#         
#         if return_stats:
#             stats[ord] = {}
#             resid = laser_fit(x) - lam
#             stats[ord]['resids'] = resid
#             #mean error in RV for a single line = c * (stddev(resid) / mean(lambda))
#             stats[ord]['single_rverr'] = 3e8 * (np.std(resid) / np.mean(lam))
#             stats[ord]['rverr'] = 3e8 * (np.std(resid) / np.mean(lam)) / np.sqrt(len(lam))
#             stats[ord]['n_lines'] = len(lam)
#             
#         #save to output dictionary
#         laser_dispsol[ord] = laser_fit
# 
#     
#     #let's do order 43 differently because it has the stupid gap in the middle
#     #find fitted x-positions of ThAr peaks
#     ord = 'order_43'
#     if verbose:
#             print('Finding wavelength solution for '+str(ord))
#     data = laserdata['flux'][ord] * mask_01[ord]
#     data1 = data[:2500]
#     data2 = data[2500:]
#     goodpeaks1,mostpeaks1,allpeaks1,first_mask1,second_mask1,third_mask1 = find_suitable_peaks(data1,return_masks=True)    #from this we just want use_mask this time (should be very fast)
#     goodpeaks2,mostpeaks2,allpeaks2,first_mask2,second_mask2,third_mask2 = find_suitable_peaks(data2,return_masks=True)    #from this we just want use_mask this time (should be very fast)
#     #fitted_laser_pos1, laser_qualflag1 = fit_emission_lines(data1, laser=True, return_all_pars=False, return_qualflag=True, varbeta=varbeta)
#     #fitted_laser_pos2, laser_qualflag2 = fit_emission_lines(data2, laser=True, return_all_pars=False, return_qualflag=True, varbeta=varbeta)
#     if varbeta:
#         fitted_laser_pos1, fitted_laser_sigma1, fitted_laser_amp1, fitted_laser_beta1 = fit_emission_lines(data1, laser=True, return_all_pars=True, return_qualflag=False, varbeta=varbeta)
#         fitted_laser_pos2, fitted_laser_sigma2, fitted_laser_amp2, fitted_laser_beta2 = fit_emission_lines(data2, laser=True, return_all_pars=True, return_qualflag=False, varbeta=varbeta)
#     else:
#         fitted_laser_pos1, fitted_laser_sigma1, fitted_laser_amp1 = fit_emission_lines(data1, laser=True, return_all_pars=True, return_qualflag=False, varbeta=varbeta)
#         fitted_laser_pos2, fitted_laser_sigma2, fitted_laser_amp2 = fit_emission_lines(data2, laser=True, return_all_pars=True, return_qualflag=False, varbeta=varbeta)
#     x1 = fitted_laser_pos1.copy()
#     x2 = fitted_laser_pos2.copy() + 2500
#     #exclude the leftmost and rightmost peaks (nasty edge effects...)
# #         blue_cutoff = int(np.round((x[-1]+x[-2])/2.,0))
# #         red_cutoff = int(np.round((x[0]+x[1])/2.,0))
#     blue_cutoff1 = int(np.round(allpeaks1[-1]+((allpeaks1[-1] - allpeaks1[-2])/2),0))
#     blue_cutoff2 = int(np.round(allpeaks2[-1]+((allpeaks2[-1] - allpeaks2[-2])/2)+2500,0))
#     red_cutoff1 = int(np.round(allpeaks1[0]-((allpeaks1[1] - allpeaks1[0])/2),0))
#     red_cutoff2 = int(np.round(allpeaks2[0]-((allpeaks2[1] - allpeaks2[0])/2)+2500,0))
#     cond1_1 = (laser_ref_wl >= wl[ord][blue_cutoff1])
#     cond1_2 = (laser_ref_wl >= wl[ord][blue_cutoff2])
#     cond2_1 = (laser_ref_wl <= wl[ord][red_cutoff1])
#     cond2_2 = (laser_ref_wl <= wl[ord][red_cutoff2])
#     #these are the theoretical wavelengths from the NIST linelists
#     lam1 = laser_ref_wl[np.logical_and(cond1_1,cond2_1)][::-1]
#     lam2 = laser_ref_wl[np.logical_and(cond1_2,cond2_2)][::-1]
#     lam1 = lam1[first_mask1][second_mask1][third_mask1]
#     lam2 = lam2[first_mask2][second_mask2][third_mask2]
#     
#     x = np.r_[x1,x2]
#     lam = np.r_[lam1,lam2]
#     
#     #check if the number of lines found equals the number of lines from the line list
#     if verbose:
#         print(len(x),len(lam))
#     if len(x) != len(lam):
#         print('fuganda')
#         return 'fuganda'
#     
#     #fit polynomial to lambda as a function of x
#     laser_fit = np.poly1d(np.polyfit(x, lam, deg_polynomial))
#     
#     if return_stats:
#         stats[ord] = {}
#         resid = laser_fit(x) - lam
#         stats[ord]['resids'] = resid
#         #mean error in RV for a single line = c * (stddev(resid) / mean(lambda))
#         stats[ord]['single_rverr'] = 3e8 * (np.std(resid) / np.mean(lam))
#         stats[ord]['rverr'] = 3e8 * (np.std(resid) / np.mean(lam)) / np.sqrt(len(lam))
#         stats[ord]['n_lines'] = len(lam)
#     
#     #save to output dictionary
#     laser_dispsol[ord] = laser_fit
# 
#     if timit:
#         print('Time taken for finding Laser-comb wavelength solution: '+str(time.time() - start_time)+' seconds...')
# 
#     if return_stats:
#         return laser_dispsol, stats 
#     else:
#         return laser_dispsol
#         
# 
# 
# 
# 
# ###########################################################################################################
# # laser_dispsol2,stats2 = get_dispsol_from_laser(laserdata, laser_ref_wl, verbose=True, timit=True, return_stats=True, deg_polynomial=2)
# # laser_dispsol3,stats3 = get_dispsol_from_laser(laserdata, laser_ref_wl, verbose=True, timit=True, return_stats=True, deg_polynomial=3)
# # laser_dispsol5,stats5 = get_dispsol_from_laser(laserdata, laser_ref_wl, verbose=True, timit=True, return_stats=True, deg_polynomial=5)
# # laser_dispsol11,stats11 = get_dispsol_from_laser(laserdata, laser_ref_wl, verbose=True, timit=True, return_stats=True, deg_polynomial=11)
# ###########################################################################################################









