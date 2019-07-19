'''
Created on 5 Oct. 2018

@author: christoph
'''


'''
Created on 25 Jul. 2018

@author: christoph
'''

import glob
import astropy.io.fits as pyfits
import numpy as np
import datetime
import copy
import os

from veloce_reduction.veloce_reduction.get_info_from_headers import get_obstype_lists
from veloce_reduction.veloce_reduction.helper_functions import short_filenames
from veloce_reduction.veloce_reduction.calibration import get_bias_and_readnoise_from_bias_frames, make_ronmask, make_master_bias_from_coeffs, make_master_dark, correct_orientation, crop_overscan_region
from veloce_reduction.veloce_reduction.order_tracing import find_stripes, make_P_id, make_mask_dict, extract_stripes
from veloce_reduction.veloce_reduction.spatial_profiles import fit_profiles, fit_profiles_from_indices
from veloce_reduction.veloce_reduction.chipmasks import make_chipmask
from veloce_reduction.veloce_reduction.extraction import *
from process_scripts import process_whites, process_science_images


date = '20180917'

# desktop:
# path = '/Volumes/BERGRAID/data/veloce/raw_goodonly/' + date + '/'
# laptop:
path = '/Users/christoph/data/raw_goodonly/' + date + '/'



# (0) GET INFO FROM FITS HEADERS ####################################################################################################################
acq_list, bias_list, dark_list, flat_list, skyflat_list, domeflat_list, arc_list, thxe_list, laser_list, laser_and_thxe_list, stellar_list, unknown_list = get_obstype_lists(path)
assert len(unknown_list) == 0, "WARNING: unknown files encountered!!!"
# obsnames = short_filenames(bias_list)
dumimg = crop_overscan_region(correct_orientation(pyfits.getdata(bias_list[0])))
ny,nx = dumimg.shape
del dumimg
#####################################################################################################################################################



# # (1) BAD PIXEL MASK ##############################################################################################################################
# bpm_list = glob.glob(path + '*bad_pixel_mask*')
# #read most recent bad pixel mask
# bpm_dates = [x[-12:-4] for x in bpm_list]
# most_recent_datestring = sorted(bpm_dates)[-1]
# bad_pixel_mask = np.load(path + 'bad_pixel_mask_' + most_recent_datestring + '.npy')
# 
# # update the pixel mask
# #blablabla
# 
# #save updated bad pixel mask
# now = datetime.datetime.now()
# dumstring = str(now)[:10].split('-')
# datestring = ''.join(dumstring)
# np.save(path+'bad_pixel_mask_'+datestring+'.npy', bad_pixel_mask)
# ###################################################################################################################################################



# (2) CALIBRATION ###################################################################################################################################
# gain = [0.88, 0.93, 0.99, 0.93]   # from "VELOCE_DETECTOR_REPORT_V1.PDF"
# gain = [1., 1., 1., 1.]
gain = [1., 1.095, 1.125, 1.]   # eye-balled from extracted flat fields

# (i) BIAS and READ NOISE
# check if MEDIAN BIAS already exists
choice = 'r'
if os.path.isfile(path + 'median_bias.fits'):
    choice = raw_input("MEDIAN BIAS image for " + date + " already exists! Do you want to skip this step or recreate it? ['s' / 'r']")
if choice.lower() == 's':
    medbias = pyfits.getdata(path + 'median_bias.fits')
else:
    # get offsets and read-out noise from bias frames (units: [offsets] = ADUs; [RON] = e-)
    if len(bias_list) > 9:
        bias_list = sorted(bias_list)[0:9]
    else:
        bias_list = sorted(bias_list)
    medbias,coeffs,offsets,rons = get_bias_and_readnoise_from_bias_frames(bias_list, degpol=5, clip=5., gain=gain, save_medimg=True, debug_level=1, timit=True)

# check if read noise mask already exists
choice = 'r'
if os.path.isfile(path + 'read_noise_mask.fits'):
    choice = raw_input("READ NOISE MASK for " + date + " already exists! Do you want to skip this step or recreate it? ['s' / 'r']")
if choice.lower() == 's':
    ronmask = pyfits.getdata(path + 'read_noise_mask.fits')
else:
    ronmask = make_ronmask(rons, nx, ny, gain=gain, savefile=True, path=path, timit=True)



# MB = make_master_bias_from_coeffs(coeffs, nx, ny, savefile=True, path=path, timit=True)
# or
# MB = offmask.copy()
# #or
# MB = medbias.copy()
#XXXalso save read-noise and offsets for all headers to write later!?!?!?


# (ii) DARKS
# create (bias-subtracted) MASTER DARK frame (units = electrons)
# MD = make_master_dark(dark_list, MB=MB, gain=gain, scalable=False, savefile=True, path=path, timit=True)
# MDS = make_master_dark(dark_list, MB=medbias, gain=gain, scalable=True, savefile=True, path=path, debug_level=1, timit=True)
MDS = np.zeros(medbias.shape)



# (iii) WHITES 
#create (bias- & dark-subtracted) MASTER WHITE frame and corresponding error array (units = electrons)
choice_mw = 'r'
if os.path.isfile(path + 'master_white.fits'):
    choice_mw = raw_input("MASTER WHITE image for " + date + " already exists! Do you want to skip this step or recreate it? ['s' / 'r']")
if choice_mw.lower() == 's':
    MW = pyfits.getdata(path + 'master_white.fits', 0)
    err_MW = pyfits.getdata(path + 'master_white.fits', 1)
else:
    # this is a first iteration without background removal - just so we can do the tracing; then we come back and do it properly later
    MW,err_MW = process_whites(flat_list, MB=medbias, ronmask=ronmask, MD=MDS, gain=gain, scalable=True, fancy=False, P_id=None,
                               clip=5., savefile=False, saveall=False, diffimg=False, remove_bg=False, path=path, debug_level=1, timit=False)
#####################################################################################################################################################



# (3) ORDER TRACING #################################################################################################################################
choice = 'r'
if os.path.isfile(path + 'P_id.npy') and os.path.isfile(path + 'mask.npy'):
    choice = raw_input("ORDER TRACING has already been done for " + date + " ! Do you want to skip this step or recreate it? ['s' / 'r']")
if choice.lower() == 's':
    P_id = np.load(path + 'P_id.npy').item()
    mask = np.load(path + 'mask.npy').item()
else:
    # find rough order locations
    #P,tempmask = find_stripes(MW, deg_polynomial=2, min_peak=0.05, gauss_filter_sigma=3., simu=False)
    P,tempmask = find_stripes(MW, deg_polynomial=2, min_peak=0.05, gauss_filter_sigma=10., simu=False, maskthresh = 400)
    # if the bad pixel column is found as an order:
    # del P[5]
    # tempmask = tempmask[np.r_[0:5, 6:40],:]
    # assign physical diffraction order numbers (this is only a dummy function for now) to order-fit polynomials and bad-region masks
    P_id_dum = make_P_id(P)
    assert len(P_id_dum) == 39, 'ERROR: not exactly 39 orders found!!!'
    mask = make_mask_dict(tempmask)
    P_id = copy.deepcopy(P_id_dum)
    # for the dates where fibre 8 had a ~20% drop in throughput, do NOT subtract 2
    if (int(date) <= 20190203) or (int(date) >= 20190619):
        for o in P_id.keys():
            P_id[o][0] -= 2.
    np.save(path + 'P_id.npy', P_id)
    np.save(path + 'mask.npy', mask)
    

# now redo the master white properly, incl background removal, and save to file
if choice_mw.lower() == 'r':
    MW,err_MW = process_whites(flat_list, MB=medbias, ronmask=ronmask, MD=MDS, gain=gain, scalable=True, fancy=False, P_id=P_id,
                               clip=5., savefile=True, saveall=False, diffimg=False, remove_bg=True, path=path, debug_level=1, timit=False)
# extract stripes of user-defined width from the science image, centred on the polynomial fits defined in step (1)
MW_stripes,MW_indices = extract_stripes(MW, P_id, return_indices=True, slit_height=30)
pix,flux,err = extract_spectrum_from_indices(MW, err_MW, MW_indices, method='quick', slit_height=30, ronmask=ronmask,
                                             savefile=True, filetype='fits', obsname='master_white', path=path, timit=True)
pix,flux,err = extract_spectrum_from_indices(MW, err_MW, MW_indices, method='optimal', slope=True, offset=True, fibs='all', slit_height=30,
                                             ronmask=ronmask, savefile=True, filetype='fits', obsname='master_white', date=date, path=path, timit=True)
#####################################################################################################################################################


###
#if we want to determine spatial profiles, then we should remove cosmics and background from MW like so:

# cosmic_cleaned_MW = remove_cosmics(MW, ronmask, obsname, path, Flim=3.0, siglim=5.0, maxiter=1, savemask=False, savefile=False, save_err=False, verbose=True, timit=True)
# bg_corrected_MW = remove_background(cosmic_cleaned_MW, P_id, obsname, path, degpol=5, slit_height=5, save_bg=False, savefile=False, save_err=False, exclude_top_and_bottom=True, verbose=True, timit=True)
#before doing the following:
# MW_stripes,MW_stripe_indices = extract_stripes(MW, P_id, return_indices=True, slit_height=30)
# err_MW_stripes = extract_stripes(err_MW, P_id, return_indices=False, slit_height=30)
# pix_MW_q,flux_MW_q,err_MW_q = extract_spectrum_from_indices(MW, err_MW, MW_stripe_indices, method='quick', slit_height=30, RON=ronmask, savefile=True,
#                                                          filetype='fits', obsname='master_white', path=path, timit=True)
# pix_MW,flux_MW,err_MW = extract_spectrum_from_indices(MW, err_MW, MW_stripe_indices, method='optimal', individual_fibres=True, slit_height=30, RON=ronmask, savefile=True,
#                                                          filetype='fits', obsname='master_white', path=path, timit=True)

# fp = fit_profiles(P_id, MW_stripes, err_MW_stripes, mask=mask, stacking=True, slit_height=5, model='gausslike', return_stats=True, timit=True)
# #OR
# fp2 = fit_profiles_from_indices(P_id, MW, err_MW, MW_stripe_indices, mask=mask, stacking=True, slit_height=5, model='gausslike', return_stats=True, timit=True)
# ###

# create and save chipmask
chipmask = make_chipmask(date, savefile=True, timit=True)
# stellar_traces = make_order_traces_from_chipmask(chipmask, centre_on='stellar')   # just an idea at this stage...


### (4) PROCESS SCIENCE IMAGES
# figure out the configuration of the calibration lamps for the ARC exposures
arc_sublists = {'lfc':[], 'thxe':[], 'both':[], 'neither':[]}
for file in arc_list:
    lc = 0
    thxe = 0
    h = pyfits.getheader(file)
    if 'LCEXP' in h.keys():
        lc = 1
    if h['SIMCALTT'] > 0:
        thxe = 1
    assert lc+thxe in [0,1,2], 'ERROR: could not establish status of LFC and simultaneous ThXe for the exposures in this list!!!'    
    if lc+thxe == 0:
        arc_sublists['neither'].append(file)
    if lc+thxe == 1:
        if lc == 1:
            arc_sublists['lfc'].append(file)
        else:
            arc_sublists['thxe'].append(file)
    elif lc+thxe == 2:
        arc_sublists['both'].append(file)
        
# (4a) PROCESS ARC IMAGES
for subl in arc_sublists.keys():
    dum = process_science_images(arc_sublists[subl], P_id, chipmask, mask=mask, sampling_size=25, slit_height=30, gain=gain, MB=medbias, ronmask=ronmask, MD=MDS, scalable=True,
                                 saveall=False, path=path, ext_method='optimal', offset='True', slope='True', fibs='all', date=date, from_indices=True, timit=True)
# (4b) PROCESS STELLAR IMAGES
dum = process_science_images(stellar_list, P_id, chipmask, mask=mask, sampling_size=25, slit_height=24, gain=gain, MB=medbias, ronmask=ronmask, MD=MDS, scalable=True, 
                             saveall=False, path=path, ext_method='optimal', offset='True', slope='True', fibs='stellar', date=date, from_indices=True, timit=True)



# (5) calculate barycentric correction and append to FITS header



# (6) calculate RV













