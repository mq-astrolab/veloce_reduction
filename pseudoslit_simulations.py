'''
Created on 11 May 2018

@author: christoph
'''


import numpy as np
import astropy.io.fits as pyfits
import time
import os

import matplotlib.pyplot as plt

#from matplotlib.colors import LogNorm

from veloce_reduction.helper_functions import get_iterable, find_nearest, get_datestring
from extract_spectrum import pos

from matplotlib.cm import cmap_d
from plotting_helpers import circles

# C = []
# I = []
# O = []
# tot = []
# seeing = np.arange(.5,5,.1)
# # for fwhm in seeing:
# #     fluxes = flux_ratios_from_seeing(fwhm, return_values=True)
# #     C.append(fluxes[1])
# #     I.append(fluxes[2])
# #     O.append(fluxes[3])
# #     tot.append(fluxes[0])



#path = '/Volumes/BERGRAID/data/simu/'



def flux_ratios_from_seeing(seeing, verbose=False):
    '''
    This routine calculates, as a function of seeing (which is approximated by a 2D Gaussian with FWHM = seeing in both directions),
    the flux ratios that the Veloce IFU captures in total, as well as the ratios of the flux captured by the central fibre / inner-ring fibres / outer-ring fibres
    to the total flux captured by the IFU (NOT the TOTAL total flux!!!)
    
    INPUT:
    'seeing'        :  can be scalar or array/list; the FWHM of the seeing disk (approximated as a 2-dim Gaussian)
    
    KEYWORDS:
    'verbose'       : boolean - do you want verbose output printed to screen?
    
    OUTPUT:
    'flux_ratios'   : a python dictionary containing the results
    
    MODHIST:
    Dec 2017    - CMB create
    11/05/2018  - CMB modified output format and included possibility to iterate over 1-element seeing-array
    '''
    
    #initialise output dictionary
    flux_ratios = {}    
    
    #inner-radius r of hexagonal fibre is 0.26", therefore outer-radius R is (2/sqrt(3))*0.26" = 0.30"
    #what we really want is the "effective" radius though, for a regular hexagon that comes from A = 3*r*R = 2*sqrt(3)*r = pi * r_eff**2
    #ie r_eff = sqrt( (2*sqrt(3)) / pi )
#    fac = np.sqrt((2.*np.sqrt(3.)) / np.pi)
    rc = 0.26
    Rc = rc * 2. / np.sqrt(3.) 
#     ri = 0.78
#     ro = 1.30
#     reff = rc * fac
    di = 0.52
    do1 = 1.04
    xo2 = di + rc
    yo2 = 1.5 * Rc
#    do2 = 3. * (2./np.sqrt(3.)) * 0.26
    
    x = np.arange(-10,10.01,.01)
    y = np.arange(-10,10.01,.01)
    xx, yy = np.meshgrid(x, y)
    #define constant (FWHM vs sigma, because FWHM = sigma * 2*sqrt(2*log(2))
    cons = 2*np.sqrt(np.log(2)) 
    
    
    m1 = 1./np.sqrt(3.)  #slope1
    m2 = -m1              #slope2
    central = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.abs(xx) <= rc, yy <= m1*xx + Rc), yy >= m1*xx - Rc), yy <= m2*xx + Rc), yy >= m2*xx - Rc)
    inner = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.abs(xx-di) <= rc, yy <= m1*(xx-di) + Rc), yy >= m1*(xx-di) - Rc), yy <= m2*(xx-di) + Rc), yy >= m2*(xx-di) - Rc), ~central)
    outer1 = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.abs(xx-do1) <= rc, yy <= m1*(xx-do1) + Rc), yy >= m1*(xx-do1) - Rc), yy <= m2*(xx-do1) + Rc), yy >= m2*(xx-do1) - Rc), ~inner)
    outer2 = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.abs(xx-xo2) <= rc, yy <= m1*(xx-xo2) + Rc - yo2), yy >= m1*(xx-xo2) - Rc - yo2), yy <= m2*(xx-xo2) + Rc - yo2), yy >= m2*(xx-xo2) - Rc - yo2), ~inner), ~outer1)

#########################################################
#     this was the circular approximation:
#     central = (np.sqrt(xx*xx + yy*yy) <= rc)
#     inner = np.logical_and(np.sqrt(xx*xx + yy*yy) <= ri, ~central)
#     outer = np.logical_and(np.sqrt(xx*xx + yy*yy) <= ro, np.logical_and(~central, ~inner))
#     ifu = (np.sqrt(xx*xx + yy*yy) <= ro)
#     central = (np.sqrt(xx*xx + yy*yy) <= reff)
#     inner = np.sqrt((xx-di)*(xx-di) + yy*yy) <= reff
#     outer1 = np.sqrt((xx-do1)*(xx-do1) + yy*yy) <= reff
#     outer2 = np.sqrt((xx-do2)*(xx-do2) + yy*yy) <= reff
#     ifu = (np.sqrt(xx*xx + yy*yy) <= ro*fac)
#########################################################
    
#     if len(np.atleast_1d(seeing)) > 1:
#         
#         frac_ifu = []
#         renorm_frac_c = []
#         renorm_frac_i = []
#         renorm_frac_o = []
#         
#         for fwhm in seeing:
#         
#             if verbose:
#                 print('Simulating '+str(fwhm)+'" seeing...')
#             
#             #calculate 2-dim Gaussian flux distribution as function of input FWHM
#             fx = np.exp(-(np.absolute(xx) * cons / fwhm) ** 2.0)
#             fy = np.exp(-(np.absolute(yy) * cons / fwhm) ** 2.0)
#             f = fx * fy * 1e6
#             
#             frac_c = np.sum(f[central]) / np.sum(f)
#             frac_i = 6. * np.sum(f[inner]) / np.sum(f)
#             frac_o = 6. * ( np.sum(f[outer1]) + np.sum(f[outer2]) ) / np.sum(f)
#             ifu_frac = frac_c + frac_i + frac_o     #this is slightly overestimated (<1%) because they are not actually circular fibres
#             frac_ifu.append(ifu_frac)
#             
#             rfc = frac_c / ifu_frac
#             rfi = frac_i / ifu_frac
#             rfo = frac_o / ifu_frac
#             
#             renorm_frac_c.append(rfc)
#             renorm_frac_i.append(rfi)
#             renorm_frac_o.append(rfo)
#             
#             if verbose:
#                 print('Total fraction of flux captured by IFU: '+str(np.round(ifu_frac * 100,1))+'%')
#                 print('----------------------------------------------')
#                 print('Contribution from central fibre: '+str(np.round(rfc * 100,1))+'%')
#                 print('Contribution from inner-ring fibres: '+str(np.round(rfi * 100,1))+'%')
#                 print('Contribution from outer-ring fibres: '+str(np.round(rfo * 100,1))+'%')
#                 print
#     
#     else:
#         
#         fwhm = seeing
#         if verbose:
#                 print('Simulating '+str(fwhm)+'" seeing...')
#         #calculate 2-dim Gaussian flux distribution as function of input FWHM
#         fx = np.exp(-(np.absolute(xx) * cons / fwhm) ** 2.0)
#         fy = np.exp(-(np.absolute(yy) * cons / fwhm) ** 2.0)
#         f = fx * fy * 1e6
#         
#         frac_c = np.sum(f[central]) / np.sum(f)
#         frac_i = 6. * np.sum(f[inner]) / np.sum(f)
#         frac_o = 6. * ( np.sum(f[outer1]) + np.sum(f[outer2]) ) / np.sum(f)
#         frac_ifu = frac_c + frac_i + frac_o     #this is slightly overestimated (<1%) because they are not actually circular fibres
#         
#         renorm_frac_c = frac_c / frac_ifu
#         renorm_frac_i = frac_i / frac_ifu
#         renorm_frac_o = frac_o / frac_ifu
#         
#         if verbose:
#                 print('Total fraction of flux captured by IFU: '+str(np.round(frac_ifu * 100,1))+'%')
#                 print('----------------------------------------------')
#                 print('Contribution from central fibre: '+str(np.round(renorm_frac_c * 100,1))+'%')
#                 print('Contribution from inner-ring fibres: '+str(np.round(renorm_frac_i * 100,1))+'%')
#                 print('Contribution from outer-ring fibres: '+str(np.round(renorm_frac_o * 100,1))+'%')
#                 print
    
    
    flux_ratios['seeing'] = np.array([])
    flux_ratios['frac_ifu'] = np.array([])
    flux_ratios['central'] = np.array([])
    flux_ratios['inner'] = np.array([])
    flux_ratios['outer'] = np.array([])
    flux_ratios['outer1'] = np.array([])
    flux_ratios['outer2'] = np.array([])
    
    #stupid python!!!
    for fwhm in get_iterable(seeing):
    
        if verbose:
            print('Simulating '+str(fwhm)+'" seeing...')
        
        #calculate 2-dim Gaussian flux distribution as function of input FWHM
        fx = np.exp(-(np.absolute(xx) * cons / fwhm) ** 2.0)
        fy = np.exp(-(np.absolute(yy) * cons / fwhm) ** 2.0)
        #f = fx * fy
        f = fx * fy * 1e6
        
        frac_c = np.sum(f[central]) / np.sum(f)
        frac_i = 6. * np.sum(f[inner]) / np.sum(f)
        frac_o1 = 6. * np.sum(f[outer1]) / np.sum(f)
        frac_o2 = 6. * np.sum(f[outer2]) / np.sum(f)
        frac_o = 6. * ( np.sum(f[outer1]) + np.sum(f[outer2]) ) / np.sum(f)
        ifu_frac = frac_c + frac_i + frac_o     #this is slightly overestimated (<1%) because they are not actually circular fibres
        
        #renormalized fractions (renormalized to the total flux captured by the IFU, not the TOTAL total flux)
        rfc = frac_c / ifu_frac
        rfi = frac_i / ifu_frac
        rfo = frac_o / ifu_frac
        rfo1 = frac_o1 / ifu_frac
        rfo2 = frac_o2 / ifu_frac
        
        #fill the output dictionary  
        flux_ratios['seeing'] = np.append(flux_ratios['seeing'], fwhm)
        flux_ratios['frac_ifu'] = np.append(flux_ratios['frac_ifu'], ifu_frac)
        flux_ratios['central'] = np.append(flux_ratios['central'], rfc)
        flux_ratios['inner'] = np.append(flux_ratios['inner'], rfi)
        flux_ratios['outer'] = np.append(flux_ratios['outer'], rfo)
        flux_ratios['outer1'] = np.append(flux_ratios['outer1'], rfo1)
        flux_ratios['outer2'] = np.append(flux_ratios['outer2'], rfo2)
        
        if verbose:
            print('Total fraction of flux captured by IFU: '+str(np.round(ifu_frac * 100,1))+'%')
            print('----------------------------------------------')
            print('Contribution from central fibre: '+str(np.round(rfc * 100,1))+'%')
            print('Contribution from inner-ring fibres: '+str(np.round(rfi * 100,1))+'%')
            print('Contribution from outer-ring fibres: '+str(np.round(rfo * 100,1))+'%')
            print
    
    return flux_ratios





def get_pseudo_slitamps(central, inner, outer1, outer2):
    """
    INPUT:
    flux_ratios bzw. rel. intensities of the central / inner / outer(1/2) fibres (can be scalars or arrays)
    
    OUTPUT:
    'slit'   : 19-element array containing the rel. intensities arranged like shown in 
               "/Users/christoph/UNSW/IFU_layout/IFU_to_pseudoslit_fibre_mapping_20180222.png", ie [L S1 S3 S4 X 2 19 8 3 9 10 4 11 12 1 13 14 5 15 16 6 17 18 7 X S2 S5 ThXe]
               fibres 8,10,12,14,16,18 are "outer1"
               fibres 9,11,13,15,17,19 are "outer2"
               
    MODHIST:
    15/05/2018 - CMB create
    """
    
    slitamps = np.array([inner,outer2,outer1,inner,outer2,outer1,inner,outer2,outer1,central*6.,outer2,outer1,inner,outer2,outer1,inner,outer2,outer1,inner]) / 6.
     
    return slitamps





def make_pseudoslits(sigma=67., nfib=19):
    """
    This routine creates random-normal-sampled offsets for the stellar fibres that make up the Veloce pseudoslit. The user has the option to supply the desired level of scatter 
    in the fibre offsets, which is used to draw from a random-normal distribution the offsets (in steps of 10 m/s). Default value is 67 m/s, based on the considerations 
    in "WORD DOCUMENT HERE". Maximum offset is +/- 200 m/s, which corresponds to ~3 default sigmas, or a probability of ~0.27% that the offset more than that (in which case 
    the maximum offset is still used).
    NOTE: Alignment inaccuracies in the spatial directions are NOT treated here, as they should not influence the RV measurements.
    
    INPUT:
    'sigma'   : RMS of the fibre offsets (in m/s)
    'nfib'    : number of fibres (default for Veloce is 19 stellar fibres)
    
    OUTPUT:
    'offsets' : an nfib-element array of the individual fibre offsets (in m/s)
    
    MODHIST:
    15/05/2018 - CMB create
    """

    #For random samples from N(mu, sigma^2), use:   "sigma * np.random.randn(dimensions) + mu"
    pos = sigma * np.random.randn(nfib)
    
    #divide into discreet bins
    bins = np.arange(-200,201,10)
    
    #which bin is the right one?
    offsets = np.array([])
    for x in pos:
        offsets = np.append(offsets, find_nearest(bins,x))
        
    return offsets





def plot_pseudoslit(offsets, saveplot=False):
    """
    This routine plots the pseudo-slit corresponding to the input array 'offsets'. Fibres are color-coded blue to red.
    """
    
    ycen = np.arange(19) * 2.15
    xcen = 0.1 * offsets / 200.
    
    fig, ax = plt.subplots()
    
    circles(xcen, ycen, s=1, c=xcen, ec='black',cmap=plt.cm.get_cmap('bwr'),autoscale=True)
    ax.scatter(xcen,ycen,color='black',marker='o',s=.5)
    ax.axvline(x=0, color='grey', ls='--')
    ax.axvline(x=1, color='grey', ls='--')
    ax.axvline(x=-1, color='grey', ls='--')
    
#     plt.xlim(-1.15,1.15)
#     plt.xlim(-25,25)
#     plt.ylim(-5,45)
    
    #ax.set_aspect(aspect=.2)
    
    #plt.axis('scaled')
    plt.xlabel('dispersion direction pixels')
    plt.ylabel('spatial direction pixels')
    
    if saveplot:
        print("NIGHTMARE: I haven't done that yet...")
    return





def make_pseudo_obs(seeing, simpath='/Volumes/BERGRAID/data/simu/', nfib=19, outpath='/Volumes/BERGRAID/data/simu/composite/', savefile=True, verbose=False, debug_level=0, timit=False):    
    """
    This routine creates simulated observations by 
    (1) calculating the flux-ratios / relative intensities in the individual fibres,
    (2) simulating a pseudo-slit with a random normal distribution of relative offsets of the fibres relative to the centre (ie simulating alignment imperfections),
    (3) adding together the individual-fibre spectra from my EchelleSimulator library using the appropriate relative intensities.
    
    INPUT:
    'seeing'      : FWHM values of desired simulated seeing condition(s) - can be scalar or array
    'simpath'     : path to my EchelleSimulator spectrum library
    'nfib'        : the number of (stellar) fibres to use
    'outpath'     : root-directory for the output files (sub-directories will be created every time this routine is run because of the random-normal pseudo-slit alignments)
    'savefile'    : boolean - do you want to save the simulated spectra as FITS files?
    'verbose'     : boolean - for debugging...
    'debug_level' : boolean - for debugging...
    'timit'       : boolean - do you want to time the execution time?
    
    OUTPUT:
    'master_dict' : dictionary containing the simulated spectra for all seeing values (only if the 'savefile' keyword is not set)
    
    TODO: add SNR   -   for now this is for a given (fixed) observing time, i.e. the total flux captured by the IFU drops as the seeing increases
    
    MODHIST:
    16/05/2018 - CMB create
    """
    
    ##### (1) #####
    #first, calculate flux_ratios (ie relative intensities) for the different fibres from given seeing
    fr = flux_ratios_from_seeing(seeing, verbose=verbose)
    
    #then get an array of the relative intensities in the pseudo-slit
    slitamps = get_pseudo_slitamps(fr['central'], fr['inner'], fr['outer1'], fr['outer2'])
    
    ##### (2) #####
    #simulate fibre-offsets by using RV offsets
    offsets = make_pseudoslits()
    
    if debug_level >= 1:
        plot_pseudoslit(offsets)
    
    ##### (3) #####
    #some string manipulations for filenames in for loop below
    strshifts = np.abs(offsets).astype(int).astype(str)
    redblue = np.empty(nfib).astype(str)
    redblue[offsets > 0] = 'red'
    redblue[offsets < 0] = 'blue'
    redblue[offsets == 0] = ''
    
    if savefile:
        #create new sub-folder with README file containing info on the fibre offsets
        datestring = get_datestring()
        dum = 1
        newpath = outpath + 'test_' + datestring
        dumpath = outpath + 'test_' + datestring
        while os.path.exists(dumpath):
            dum += 1
            dumpath = newpath + '_' + str(dum)
        if dum > 1:
            newpath = newpath + '_' + str(dum)
        #create new folder
        os.makedirs(newpath)
        #write README file
        outfn = newpath + '/' + 'offsets.txt' 
        outfile = open(outfn, 'w')
        outfile.writelines(["%s\n" % item for item in offsets.astype(str)])
        outfile.close()
        
    else:
        master_dict = {}
    
    for i,fwhm in enumerate(seeing):
        if verbose:
            print('SEEING-LOOP: ',fwhm)
        #use fibre-slots 6 to 24
        for n in range(nfib):
            fibslot = str(n+6).zfill(2)
            img = pyfits.getdata(simpath+'fib'+fibslot+'_'+redblue[n]+strshifts[n]+'ms.fit') + 1.
            if n==0:
                master = img.copy().astype(float) * slitamps[n,i]
                h = pyfits.getheader(simpath+'fib'+fibslot+'_'+redblue[n]+strshifts[n]+'ms.fit')
            else:
                master += img * slitamps[n,i]
        if savefile:
            #save to file
            pyfits.writeto(newpath+'seeing'+fwhm.astype(str)+'.fits', master, h, clobber=True)
        else:
            master_dict['seeing'+fwhm.astype(str)] = master
    
    if savefile:
        #print('Offsets: ',offsets)
        return
    else:
        return master_dict









