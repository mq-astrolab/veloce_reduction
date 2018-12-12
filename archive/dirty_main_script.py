"""
Created on 10 Nov. 2017

@author: christoph
"""

import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as pyfits
from scipy import ndimage
from scipy import signal
import h5py
import scipy.sparse as sparse
import glob
import copy
#import logging
import time
import datetime
from scipy.optimize import curve_fit
import collections
from scipy import special
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit, Model
from astropy.io import ascii
from matplotlib.colors import LogNorm
from readcol import readcol
#from mpl_toolkits.mplot3d import Axes3D

#from barycorrpy import get_BC_vel

from veloce_reduction.helper_functions import *
from veloce_reduction.get_info_from_headers import identify_obstypes, get_obs_coords_from_header
from veloce_reduction.calibration import get_offmask_and_readnoise_from_bias_frames, make_master_bias_and_ronmask, make_master_dark, process_whites
from veloce_reduction.create_master_frames import create_master_img
from veloce_reduction.order_tracing import find_stripes, make_P_id, make_mask_dict, extract_stripes, find_tramlines
from veloce_reduction.cosmic_ray_removal import remove_cosmics
from veloce_reduction.background import extract_background, fit_background
from veloce_reduction.spatial_profiles import fit_profiles_from_indices, make_model_stripes_gausslike
#from veloce_reduction.find_laser_peaks_2D import find_laser_peaks_2D
from veloce_reduction.extraction import quick_extract, quick_extract_from_indices, collapse_extract, collapse_extract_from_indices, optimal_extraction, optimal_extraction_from_indices 
from veloce_reduction.wavelength_solution import get_wavelength_solution, get_simu_dispsol, fit_emission_lines_lmfit, find_suitable_peaks
from veloce_reduction.flat_fielding import onedim_pixtopix_variations, deblaze_orders
from veloce_reduction.relative_intensities import get_relints
#from veloce_reduction.pseudoslit_simulations import *
from veloce_reduction.get_radial_velocity import get_RV_from_xcorr, get_rvs_from_xcorr





#path = '/Users/christoph/OneDrive - UNSW/veloce_spectra/test1/'
#path = '/Users/christoph/OneDrive - UNSW/veloce_spectra/test2/'
#path = '/Users/christoph/OneDrive - UNSW/veloce_spectra/bias_test/'
path = '/Users/christoph/OneDrive - UNSW/veloce_spectra/test_20180517/'

#####################################################################################################################################################
# (0) identify bias / darks / whites / thoriums / stellar exposures etc. based on information in header
#imgname = '/Users/christoph/OneDrive - UNSW/simulated_spectra/blue_ghost_spectrum_20170803.fits'
#imgname = '/Users/christoph/OneDrive - UNSW/simulated_spectra/blue_ghost_spectrum_nothar_highsnr_20170906.fits'
#imgname = '/Users/christoph/OneDrive - UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib01.fit'
#template_name = '/Users/christoph/OneDrive - UNSW/simulated_spectra/ES/veloce_high_SNR_solar_template.fit'
template_name = '/Users/christoph/OneDrive - UNSW/simulated_spectra/ES/solar_template_maxsnr.fit'
imgname = '/Users/christoph/OneDrive - UNSW/simulated_spectra/ES/veloce_full_solar_image_with_2calibs.fit'
imgname2 = '/Users/christoph/OneDrive - UNSW/simulated_spectra/ES/veloce_solar_red100ms.fit'
imgname3 = '/Users/christoph/OneDrive - UNSW/simulated_spectra/ES/veloce_full_solar_image_with_2calibs_red1000ms.fit'
#gflatname = '/Users/christoph/OneDrive - UNSW/simulated_spectra/blue_ghost_flat_20170905.fits'
#flatname = '/Users/christoph/OneDrive - UNSW/simulated_spectra/veloce_flat_highsn2.fit'
#flatname = '/Users/christoph/OneDrive - UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib01.fit'
flatname = '/Users/christoph/OneDrive - UNSW/simulated_spectra/ES/veloce_flat_t70000_nfib19.fit'

flat15name = '/Users/christoph/OneDrive - UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib15.fit'

flat02name = '/Users/christoph/OneDrive - UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib02.fit'
flat03name = '/Users/christoph/OneDrive - UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib03.fit'
flat21name = '/Users/christoph/OneDrive - UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib21.fit'
flat22name = '/Users/christoph/OneDrive - UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib22.fit'
img = pyfits.getdata(imgname) + 1.
img2 = pyfits.getdata(imgname2) + 1.
img3 = pyfits.getdata(imgname3) + 1.
flat = pyfits.getdata(flatname)
flat02 = pyfits.getdata(flat02name)
flat03 = pyfits.getdata(flat03name)
flat21 = pyfits.getdata(flat21name)
flat22 = pyfits.getdata(flat22name)
img02 = flat02 + 1.
img03 = flat03 + 1.
img21 = flat21 + 1.
img22 = flat22 + 1.
flatimg = flat + 1.
temp_img = pyfits.getdata(template_name) + 1.



# #eventually this is going to be:
# path = '/Users/christoph/OneDrive - UNSW/veloce_spectra/test1/'
bias_list,dark_list,white_list,thar_list,thxe_list,laser_list,stellar_list = identify_obstypes(path)
obsnames = short_filenames(stellar_list)
dumimg = pyfits.getdata(stellar_list[0])
ny,nx = dumimg.shape
del dumimg
#####################################################################################################################################################



# (1) create/update bad pixel mask ##################################################################################################################
bpm_list = glob.glob(path + '*bad_pixel_mask*')
#read most recent bad pixel mask
bpm_dates = [x[-12:-4] for x in bpm_list]
most_recent_datestring = sorted(bpm_dates)[-1]
bad_pixel_mask = np.load(path + 'bad_pixel_mask_' + most_recent_datestring + '.npy')

# update the pixel mask
#blablabla

#save updated bad pixel mask
now = datetime.datetime.now()
dumstring = str(now)[:10].split('-')
datestring = ''.join(dumstring)
np.save(path+'bad_pixel_mask_'+datestring+'.npy', bad_pixel_mask)
#####################################################################################################################################################



# (2) subtract bias and/or dark frames ##############################################################################################################
# (i) BIAS 
offsets,rons = get_offset_and_readnoise_from_bias_frames(bias_list, debug_level=0, timit=False)
# create master bias frame
MB,ronmask = make_master_bias_and_ronmask(offsets,rons,nx,ny,savefiles=True,path=path)
## create master bias frame
#MB = create_master_img(bias_list, clip=5., imgtype='bias', remove_outliers=True, with_errors=True, savefiles=True)
# #for the Veloce lab frames???
# MB = 0.
# #maybe need to estimate the bias from overscan regions!?!?!?


# (ii) DARKS
MD,err_MD = make_master_dark(dark_list, MB, ronmask, scalable=False, savefile=True, return_errors=True, path=path, timit=False)

# #subtract master bias frame from master dark
# dum = bias_subtraction(dark_list, MB, noneg=False, savefile=True)
# #get list of bias-corrected dark frames
# dark_list_bc = glob.glob(path+"bc_dark*")
# #create master dark from bias-corrected darks; make it scalable by dividing through exposure time (ie normalize to t_exp=1s) by setting kwarg norm='exptime')
# MD = create_master_img(dark_list_bc, clip=5., RON=10., gain=1., imgtype='dark', asint=False, savefile=True, remove_outliers=True, norm=True, scalable=True)

# (iii) WHITES 
MW,err_MW = process_whites(white_list, MB=MB, ronmask=ronmask, MD=MD, err_MD=err_MD, scalable=False, fancy=False, clip=5., savefile=True, saveall=False, diffimg=False, path=None, timit=False)



#subtract master bias frame from master white
dum = bias_subtraction(white_list, MB, noneg=False, savefile=True)
#get list of bias-corrected white frames
white_list_bc = glob.glob(path+"bc_white*")
#texp_whites = 0.5       #TEMP
texp_whites = 1.0        #TEMP
#subtract scalable master dark image from bias-corrected whites:
dum = dark_subtraction(white_list_bc, texp_whites * MD, noneg=False, savefile=True)
#get list of bias-corrected and dark-corrected white frames
white_list_bc_dc = glob.glob(path+"dc_bc_white*")
#create master white from bias-corrected and dark-corrected whites
MW = create_master_img(white_list_bc_dc, clip=5., RON=10., gain=1., imgtype='white', asint=False, savefile=True, remove_outliers=True, norm=True)
#####################################################################################################################################################



# (3) order tracing #################################################################################################################################
# find orders roughly
P,tempmask = find_stripes(MW, deg_polynomial=2, min_peak=0.05, gauss_filter_sigma=3., simu=True)
# assign physical diffraction order numbers (this is only a dummy function for now) to order-fit polynomials and bad-region masks
P_id = make_P_id(P)
mask = make_mask_dict(tempmask)
# extract stripes of user-defined width from the science image, centred on the polynomial fits defined in step (1)
flat_stripes,fs_indices = extract_stripes(MW, P_id, return_indices=True, slit_height=5)
#####################################################################################################################################################



# (4) cosmic ray removal ############################################################################################################################

#####################################################################################################################################################



# (5) fit and remove background #####################################################################################################################
# extract and fit background
bg,bg_mask = extract_background(MW, P_id, return_mask=True, slit_height=10)
bg_coeffs,bg_image = fit_background(bg, deg=3, timit=True, return_full=True)
#####################################################################################################################################################



# (6) fibre-profile modelling #######################################################################################################################
# fit fibre profiles
fibre_profiles = fit_profiles_from_indices(P_id, MW, fs_indices, mask=mask, slit_height=7, model='gausslike', return_stats=True, timit=True)
# make model stripes
fitted_stripes, model_stripes = make_model_stripes_gausslike(fibre_profiles, MW, fs_indices, P_id, mask, RON=10., debug_level=1)
#####################################################################################################################################################



# (7) flat-fielding #################################################################################################################################
ny,nx = MW.shape
xx = np.arange(nx)    
yy = np.arange(ny)

#either 1D:


#or 2D:


#####################################################################################################################################################



# (8) laser-comb matching ###########################################################################################################################

#####################################################################################################################################################



# (9) extract spectra ###############################################################################################################################
# extract stripes of user-defined width from the science image, centred on the polynomial fits defined in step (3)
template_stripes,template_indices = extract_stripes(temp_img, P_id, return_indices=True, slit_height=25)

all_stripes = {}
all_stripe_indices = {}
for obsname,imgname in zip(obsnames,stellar_list):
    img = pyfits.getdata(imgname) + 1.
    stripes,stripe_indices = extract_stripes(img, P_id, return_indices=True, slit_height=25)
    all_stripes[obsname] = stripes
    all_stripe_indices[obsname] = stripe_indices

### Now extract one-dimensional spectrum via (a) quick-and dirty collapsing of stripes, (b) tramline extraction, or (c) optimal extraction ###

# (a) Quick-and-Dirty Extraction
quick_extracted_raw = {}
#pix_quick, flux_quick, err_quick = quick_extract(stripes, slit_height=25, RON=10., gain=1., verbose=False, timit=False)
#pix_quick_template, flux_quick_template, err_quick_template = quick_extract(template_stripes, slit_height=25, RON=1., gain=1., verbose=False, timit=False)
quick_extracted_raw['template'] = {}
quick_extracted_raw['template']['pix'], quick_extracted_raw['template']['flux'], quick_extracted_raw['template']['err'] = quick_extract(template_stripes, slit_height=25, RON=1., gain=1., verbose=False, timit=False)

for obsname in obsnames:
    quick_extracted_raw[obsname] = {}
    quick_extracted_raw[obsname]['pix'], quick_extracted_raw[obsname]['flux'], quick_extracted_raw[obsname]['err'] = quick_extract(all_stripes[obsname], slit_height=25, RON=1., gain=1., verbose=False, timit=False)
#extract Master White as well
#flat_pix_quick, flat_flux_quick, flat_err_quick = quick_extract(flat_stripes, slit_height=25, RON=1., gain=1., verbose=False, timit=False)
quick_extracted_raw['MW'] = {}
quick_extracted_raw['MW']['pix'], quick_extracted_raw['MW']['flux'], quick_extracted_raw['MW']['err'] = quick_extract(flat_stripes, slit_height=5, RON=1., gain=1., verbose=False, timit=False)
#this works fine as well, but is a factor of ~2 slower:
#pix_quick_fi, flux_quick_fi, err_quick_fi = quick_extract_from_indices(img, stripe_indices, slit_height=25, RON=10., gain=1., verbose=False, timit=False)
#calculate mean SNR of observation
#mean_snr = get_mean_snr(flux_quick,err=err_quick)




# (b) Tramline Extraction
# # identify tramlines for extraction
# fibre_profiles_01 = np.load('/Users/christoph/OneDrive - UNSW/fibre_profiles/sim/fibre_profiles_01.npy').item()
# laser_tramlines = find_laser_tramlines(fibre_profiles_01, mask_01)
# #-----------------------------------------------------------------------------------------------------
# ####################################################################################################################################
# # prepare the four fibres that are used for the tramline definition (only once, then saved to file for the simulations anyway) #
# # P_**,tempmask_** = find_stripes(flat**, deg_polynomial=2)                                                                        #
# # P_id_** = make_P_id(P_**)                                                                                                        #
# # mask_** = make_mask_dict(tempmask_**)                                                                                            #
# # stripes_** = extract_stripes(img**, P_id_**, slit_height=10)                                                                     #
# #                                                                                                                                  #
# # This is only needed once, ie for fitting the fibre profiles, then written to file as it takes ages...                            #
# # fibre_profiles_** = fit_profiles(img**, P_id_**, stripes_**)                                                                     #
# ####################################################################################################################################
# fibre_profiles_02 = np.load('/Users/christoph/OneDrive - UNSW/fibre_profiles/sim/fibre_profiles_02.npy').item()
# fibre_profiles_03 = np.load('/Users/christoph/OneDrive - UNSW/fibre_profiles/sim/fibre_profiles_03.npy').item()
# fibre_profiles_21 = np.load('/Users/christoph/OneDrive - UNSW/fibre_profiles/sim/fibre_profiles_21.npy').item()
# fibre_profiles_22 = np.load('/Users/christoph/OneDrive - UNSW/fibre_profiles/sim/fibre_profiles_22.npy').item()
# mask_02 = np.load('/Users/christoph/OneDrive - UNSW/fibre_profiles/masks/mask_02.npy').item()
# mask_03 = np.load('/Users/christoph/OneDrive - UNSW/fibre_profiles/masks/mask_03.npy').item()
# mask_21 = np.load('/Users/christoph/OneDrive - UNSW/fibre_profiles/masks/mask_21.npy').item()
# mask_22 = np.load('/Users/christoph/OneDrive - UNSW/fibre_profiles/masks/mask_22.npy').item()
#-----------------------------------------------------------------------------------------------------

tramlines = find_tramlines(fibre_profiles_02, fibre_profiles_03, fibre_profiles_21, fibre_profiles_22, mask_02, mask_03, mask_21, mask_22)
# for laser: pix,flux,err = collapse_extract(stripes, laser_tramlines, laser=True, RON=4., gain=1., timit=True)
pix,flux,err = collapse_extract(stripes, tramlines, laser=False, RON=4., gain=1., timit=True)


# (c) Optimal Extraction
pix,flux,err = optimal_extraction(img, P_id, stripes, stripe_indices, RON=0., gain=1., slit_height=25, timit=True, simu=False, individual_fibres=True)



### in any case, now get relative intensities in fibres
relints = get_relints(P_id, stripes, mask=mask, sampling_size=25, slit_height=25, timit=True)
#####################################################################################################################################################



# (10) wavelength calibration #######################################################################################################################
#### read dispersion solution from file (obviously this is only a temporary crutch)
####dispsol = np.load('/Users/christoph/OneDrive - UNSW/dispsol/mean_dispsol_by_orders_from_zemax.npy').item()
# get dispersion solution from laser frequency comb
#first read extracted laser comb spectrum
laserdata = np.load('/Users/christoph/OneDrive - UNSW/rvtest/laserdata.npy').item()
# now read laser_linelist
laser_ref_wl,laser_relint = readcol('/Users/christoph/OneDrive - UNSW/linelists/laser_linelist_25GHz.dat',fsep=';',twod=False)
laser_ref_wl *= 1e3


#for simulated data only
wl = get_simu_dispsol()
    
    
#finally, we are ready to call the wavelength solution routine
laser_dispsol,stats = get_dispsol_from_laser(laserdata, laser_ref_wl, verbose=True, timit=True, return_stats=True, deg_polynomial=5)


# get wavelength solution
pwl,wl = get_wavelength_solution(thflux, thflux2, poly_deg=5, laser=False, polytype='chebyshev', savetable=False, return_full=True, saveplots=False, timit=False, debug_level=0)



# add wavelength solution to extracted spectra dictionaries
for obsname in obsnames:
    quick_extracted_raw[obsname]['wl'] = wl
quick_extracted_raw['MW']['wl'] = wl
quick_extracted_raw['template']['wl'] = wl


# if flat quick-extract or collapse-extract has been performed, do the flat-fielding in 1D now
smoothed_flat, pix_sens = onedim_pixtopix_variations(quick_extracted_raw['MW']['flux'], filt='g', filter_width=25)
#quick_extracted = quick_extracted_raw.copy()
quick_extracted = copy.deepcopy(quick_extracted_raw)
#divide by the pixel-to-pixel sensitivity variations
for obs in sorted(quick_extracted.keys()):
    for o in sorted(quick_extracted[obs]['flux'].keys()):
        quick_extracted[obs]['flux'][o] = quick_extracted_raw[obs]['flux'][o] / pix_sens[o]
        
        
#####################################################################################################################################################



# (11) RADIAL VELOCITY ##############################################################################################################################
# #preparations, eg:
# f = quick_extracted['seeing1.5']['flux'].copy()
# err = quick_extracted['seeing1.5']['err'].copy()
# wl = quick_extracted['seeing1.5']['wl'].copy()
# f0 = quick_extracted['template']['flux'].copy()
# wl0 = quick_extracted['template']['wl'].copy()
# #(a) using cross-correlation
# #if using cross-correlation, we need to de-blaze the spectra first
# f_dblz, err_dblz = deblaze_orders(f, wl, smoothed_flat, mask, err=err)
# f0_dblz = deblaze_orders(f0, wl0, smoothed_flat, mask, err=None)
# #we also only want to use the central TRUE parts of the masks, ie want ONE consecutive stretch per order
# cenmask = central_parts_of_mask(mask)
# #call RV routine
# rv,rverr = get_RV_from_xcorr(f_dblz, err_dblz, wl, f0_dblz, wl0, mask=cenmask, filter_width=25, debug_level=0)     #NOTE: 'filter_width' must be the same as used in 'onedim_pixtopix_variations' above


rv,rverr = get_rvs_from_xcorr(quick_extracted, obsnames, mask, smoothed_flat, debug_level=0)

#####################################################################################################################################################



# (12) barycentric correction #######################################################################################################################
# e.g.:
lat,long,alt = get_obs_coords_from_header(fn)
bc = barycorrpy.get_BC_vel(JDUTC=JDUTC,hip_id=8102,lat=lat,longi=long,alt=float(alt),ephemeris='de430',zmeas=0.0)
bc2 = barycorrpy.get_BC_vel(JDUTC=JDUTC,hip_id=8102,lat=-31.2755,longi=149.0673,alt=1165.0,ephemeris='de430',zmeas=0.0)
bc3  = barycorrpy.get_BC_vel(JDUTC=JDUTC,hip_id=8102,obsname='AAO',ephemeris='de430')
#####################################################################################################################################################





# OUTPUT FILES???






