'''
Created on 25 Jul. 2018

@author: christoph
'''

import sys
sys.path.append("/home/astrolab/codestore/veloce_reduction/")
sys.path.append("/home/astrolab/codestore/veloce_reduction/veloce_reduction/")


import glob
import astropy.io.fits as pyfits
import numpy as np
import datetime

from veloce_reduction.veloce_reduction.get_info_from_headers import get_obstype_lists_temp
from veloce_reduction.veloce_reduction.helper_functions import short_filenames
from veloce_reduction.veloce_reduction.calibration import get_bias_and_readnoise_from_bias_frames, make_offmask_and_ronmask, make_master_bias_from_coeffs, make_master_dark, correct_orientation, crop_overscan_region
from veloce_reduction.veloce_reduction.order_tracing import find_stripes, make_P_id, make_mask_dict, extract_stripes #, find_tramlines
from veloce_reduction.veloce_reduction.spatial_profiles import fit_profiles, fit_profiles_from_indices
from veloce_reduction.process_scripts import process_whites, process_science_images




# import matplotlib.pyplot as plt
# from scipy import ndimage
# from scipy import signal
# import h5py
# import scipy.sparse as sparse
#
# import copy
# #import logging
# import time

# from scipy.optimize import curve_fit
# import collections
# from scipy import special
# from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit, Model
# from astropy.io import ascii
# from matplotlib.colors import LogNorm
# from readcol import readcol
#from mpl_toolkits.mplot3d import Axes3D

#from barycorrpy import get_BC_vel

#from veloce_reduction.helper_functions import *
# from veloce_reduction.get_info_from_headers import identify_obstypes, get_obs_coords_from_header
# from veloce_reduction.create_master_frames import create_master_img

# from veloce_reduction.cosmic_ray_removal import remove_cosmics
# from veloce_reduction.background import extract_background, fit_background
# from veloce_reduction.spatial_profiles import fit_profiles_from_indices, make_model_stripes_gausslike
# #from veloce_reduction.find_laser_peaks_2D import find_laser_peaks_2D
# from veloce_reduction.extraction import quick_extract, quick_extract_from_indices, collapse_extract, collapse_extract_from_indices, optimal_extraction, optimal_extraction_from_indices
# from veloce_reduction.wavelength_solution import get_wavelength_solution, get_simu_dispsol, fit_emission_lines_lmfit, find_suitable_peaks
# from veloce_reduction.flat_fielding import onedim_pixtopix_variations, deblaze_orders
# from veloce_reduction.relative_intensities import get_relints
# #from veloce_reduction.pseudoslit_simulations import *
# from veloce_reduction.get_radial_velocity import get_RV_from_xcorr, get_rvs_from_xcorr







# path = '/Volumes/BERGRAID/data/veloce/tests_20180723/'
# path = '/Users/christoph/data/commissioning/20180916/'
path = '/data/veloce/190919/ccd_3/'





# (0) INFO FROM FITS HEADERS ########################################################################################################################
#####TEMP#####
# bias_list,dark_list,white_list,thar_list,thxe_list,laser_list,stellar_list = identify_obstypes(path)
# bias_list = glob.glob(path + 'Bias*.fits')
# dark_list = glob.glob(path + 'Dark*.fits')
# white_list = glob.glob(path + 'Light*.fits')
# stellar_list = glob.glob(path + 'Light*.fits')
###END TEMP###
acq_list, bias_list, dark_list, flat_list, skyflat_list, domeflat_list, arc_list, thxe_list, laser_list, laser_and_thxe_list, stellar_list, unknown_list = get_obstype_lists_temp(path)
obsnames = short_filenames(stellar_list)
dumimg = crop_overscan_region(correct_orientation(pyfits.getdata(stellar_list[0])))
ny,nx = dumimg.shape
del dumimg
#####################################################################################################################################################



# (1) BAD PIXEL MASK ################################################################################################################################
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



# (2) CALIBRATION ###################################################################################################################################
gain = [0.88, 0.93, 0.99, 0.93]   # from "VELOCE_DETECTOR_REPORT_V1.PDF"
# (i) BIAS 
# get offsets and read-out noise
#either from bias frames (units: [offsets] = ADUs; [RON] = e-)
medbias,coeffs,offsets,rons = get_bias_and_readnoise_from_bias_frames(bias_list, degpol=5, clip=5., gain=gain, save_medimg=True, debug_level=0, timit=True)
#or from the overscan regions

# create MASTER BIAS frame and read-out noise mask (units = ADUs)
offmask,ronmask = make_offmask_and_ronmask(offsets, rons, nx, ny, gain=gain, savefiles=True, path=path, timit=True)
MB = make_master_bias_from_coeffs(coeffs, nx, ny, savefile=True, path=path, timit=True)
# or
# MB = offmask.copy()
# #or
# MB = medbias.copy()
#XXXalso save read-noise and offsets for all headers to write later!?!?!?


# (ii) DARKS
# create (bias-subtracted) MASTER DARK frame (units = electrons)
MD = make_master_dark(dark_list, MB=MB, gain=gain, scalable=False, savefile=True, path=path, timit=True)
MDS = make_master_dark(dark_list, MB=MB, gain=gain, scalable=True, savefile=True, path=path, timit=True)

# (iii) WHITES 
#create (bias- & dark-subtracted) MASTER WHITE frame and corresponding error array (units = electrons)
MW,err_MW = process_whites(flat_list, MB=MB, ronmask=ronmask, MD=MD, gain=gain, scalable=False, fancy=False, clip=5., savefile=True, saveall=False, diffimg=False, path=None, timit=False)
#####################################################################################################################################################



# (3) ORDER TRACING #################################################################################################################################
# find orders roughly
P,tempmask = find_stripes(MW, deg_polynomial=2, min_peak=0.05, gauss_filter_sigma=3., simu=False)
# assign physical diffraction order numbers (this is only a dummy function for now) to order-fit polynomials and bad-region masks
P_id = make_P_id(P)
mask = make_mask_dict(tempmask)
# extract stripes of user-defined width from the science image, centred on the polynomial fits defined in step (1)
MW_stripes,MW_indices = extract_stripes(MW, P_id, return_indices=True, slit_height=5)
#####################################################################################################################################################


###
#if we want to determine spatial profiles, then we should remove cosmics and background from MW like so:

# cosmic_cleaned_MW = remove_cosmics(MW, ronmask, obsname, path, Flim=3.0, siglim=5.0, maxiter=1, savemask=False, savefile=False, save_err=False, verbose=True, timit=True)
# bg_corrected_MW = remove_background(cosmic_cleaned_MW, P_id, obsname, path, degpol=5, slit_height=5, save_bg=False, savefile=False, save_err=False, exclude_top_and_bottom=True, verbose=True, timit=True)
#before doing the following:
MW_stripes,MW_stripe_indices = extract_stripes(MW, P_id, return_indices=True, slit_height=30)
err_MW_stripes = extract_stripes(err_MW, P_id, return_indices=False, slit_height=30)

fp = fit_profiles(P_id, MW_stripes, err_MW_stripes, mask=mask, stacking=True, slit_height=5, model='gausslike', return_stats=True, timit=True)
#OR
fp2 = fit_profiles_from_indices(P_id, MW, err_MW, MW_stripe_indices, mask=mask, stacking=True, slit_height=5, model='gausslike', return_stats=True, timit=True)
###



# (4) PROCESS SCIENCE IMAGES
dum = process_science_images(stellar_list, P_id, mask=mask, sampling_size=25, slit_height=30, gain=gain, MB=medbias, ronmask=ronmask, MD=MDS, scalable=True, 
                             saveall=True, path=path, ext_method='quick', from_indices=True, timit=False)



# (5) calculate RVs
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













