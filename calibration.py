"""
Created on 13 Apr. 2018

@author: christoph
"""

import astropy.io.fits as pyfits
import numpy as np
from itertools import combinations
import time
import matplotlib.pyplot as plt

from veloce_reduction.helper_functions import make_quadrant_masks, binary_indices, correct_orientation, sigma_clip





def crop_overscan_region(img):
    """
    As of July 2018, Veloce uses an e2v CCD231-84-1-E74 4kx4k chip.
    Image dimensions are 4096 x 4112 pixels, but the recorded images size including the overscan region is 4202 x 4112 pixels.
    We therefore have an overscan region of size 53 x 4112 at either end. 
    
    raw_img = pyfits.getdata(filename)     -->    raw_img.shape = (4112, 4202)
    img = correct_orientation(raw_img)     -->        img.shape = (4202, 4112)
    """

    #correct orientation if needed
    if img.shape == (4112, 4202):
        img = correct_orientation(img)

    if img.shape != (4202, 4112):
        print('ERROR: wrong image size encountered!!!')
        return

    #crop overscan region
    good_img = img[53:4149,:]

    return good_img





def extract_overscan_region(img):
    """
    As of July 2018, Veloce uses an e2v CCD231-84-1-E74 4kx4k chip.
    Image dimensions are 4096 x 4112 pixels, but the recorded images size including the overscan region is 4202 x 4112 pixels.
    We therefore have an overscan region of size 53 x 4112 at either end. 
    
    raw_img = pyfits.getdata(filename)     -->    raw_img.shape = (4112, 4202)
    img = correct_orientation(raw_img)     -->        img.shape = (4202, 4112)
    """

    #correct orientation if needed
    if img.shape == (4112, 4202):
        img = correct_orientation(img)

    if img.shape != (4202, 4112):
        print('ERROR: wrong image size encountered!!!')
        return

    ny,nx = img.shape

    #crop overscan region
    os1 = img[:53,:nx//2]
    os2 = img[:53,nx//2:]
    os3 = img[ny-53:,:nx//2]
    os4 = img[ny-53:,nx//2:]

    return os1,os2,os3,os4





def get_flux_and_variance_pairs(imglist, MB, simu=False):
    """
    measure gain from a list of flatfield / dark images as described here:
    https://www.mirametrics.com/tech_note_ccdgain.php

    units = ADUs

    INPUT:
    'imglist'
    'MB'       : the master bias frame
    'simu'     : boolean - are you using simulated spectra?

    OUTPUT:
    'f_q1' : median signal for quadrant 1
    'v_q1' : variance in signal for quadrant 1
    (same for other quadrants)
    """

    img = pyfits.getdata(imglist[0])

    # do some formatting things for real observations
    if not simu:
        # bring to "correct" orientation
        img = correct_orientation(img)
        # remove the overscan region, which looks crap for actual bias images
        img = crop_overscan_region(img)

    ny,nx = img.shape

    # define four quadrants via masks
    q1, q2, q3, q4 = make_quadrant_masks(nx, ny)

    #prepare arrays containing flux and variance
    f_q1 = np.array([])
    f_q2 = np.array([])
    f_q3 = np.array([])
    f_q4 = np.array([])
    v_q1 = np.array([])
    v_q2 = np.array([])
    v_q3 = np.array([])
    v_q4 = np.array([])

    #list of all possible pairs of files
    list_of_combinations = list(combinations(imglist, 2))

    for (name1,name2) in list_of_combinations:
        #read in observations and bring to right format
        img1 = pyfits.getdata(name1)
        img2 = pyfits.getdata(name2)
        if not simu:
            img1 = correct_orientation(img1)
            img1 = crop_overscan_region(img1)
            img2 = correct_orientation(img2)
            img2 = crop_overscan_region(img2)
        #take difference and do sigma-clipping
        # diff = img1.astype(long) - img2.astype(long)
        med1_q1 = np.nanmedian(img1[q1])
        med2_q1 = np.nanmedian(img2[q1])
        r_q1 = med1_q1 / med2_q1
        diff_q1 = img1[q1].astype(long) - r_q1 * img2[q1].astype(long)
        var_q1 = (np.nanstd(sigma_clip(diff_q1, 5))/np.sqrt(2))**2
        med1_q2 = np.nanmedian(img1[q2])
        med2_q2 = np.nanmedian(img2[q2])
        r_q2 = med1_q2 / med2_q2
        diff_q2 = img1[q2].astype(long) - r_q2 * img2[q2].astype(long)
        var_q2 = (np.nanstd(sigma_clip(diff_q2, 5)) / np.sqrt(2)) ** 2
        med1_q3 = np.nanmedian(img1[q3])
        med2_q3 = np.nanmedian(img2[q3])
        r_q3 = med1_q3 / med2_q3
        diff_q3 = img1[q3].astype(long) - r_q3 * img2[q3].astype(long)
        var_q3 = (np.nanstd(sigma_clip(diff_q3, 5)) / np.sqrt(2)) ** 2
        med1_q4 = np.nanmedian(img1[q4])
        med2_q4 = np.nanmedian(img2[q4])
        r_q4 = med1_q4 / med2_q4
        diff_q4 = img1[q4].astype(long) - r_q4 * img2[q4].astype(long)
        var_q4 = (np.nanstd(sigma_clip(diff_q4, 5)) / np.sqrt(2)) ** 2

        #fill output arrays
        f_q1 = np.append(f_q1, med1_q1)
        f_q2 = np.append(f_q2, med1_q2)
        f_q3 = np.append(f_q3, med1_q3)
        f_q4 = np.append(f_q4, med1_q4)
        v_q1 = np.append(v_q1, var_q1)
        v_q2 = np.append(v_q2, var_q2)
        v_q3 = np.append(v_q3, var_q3)
        v_q4 = np.append(v_q4, var_q4)


    return f_q1,f_q2,f_q3,f_q4,v_q1,v_q2,v_q3,v_q4





def measure_gain_from_slope(signal, variance, debug_level=0):
    """
    In a graph where the y-axis is the variance, and the x-axis is the signal,
    the gain is equal to the inverse of the slope.

    INPUT:
    'signal'       : array of signal values
    'variance'     : array of variance values
    'debug_level'  : for debugging...

    OUTPUT:
    'gain'  : the gain in e-/ADU
    """

    # unweighted linear fit
    p = np.poly1d(np.polyfit(signal, variance, 1))

    #slope is p[1]
    gain = 1./p[1]

    if debug_level >= 1:
        #plot the graph
        plt.figure()
        plt.plot(signal, variance, 'kx')
        plt.xlabel('signal')
        plt.ylabel('variance')
        xr = np.max(signal) - np.min(signal)
        xx = np.linspace(np.min(signal)-0.2*xr,np.max(signal)+0.2*xr)
        plt.plot(xx,p(xx),'g--')
        plt.title('gain = 1/slope = '+str(np.round(gain,2)))
        plt.xrange(np.min(xx),np.max(xx))

    return gain





def measure_gains(filelist):
    """

    :param filelist:
    :return:
    """

    #we want to sub-group the files in 'filelist' according to their exposure times (ie brightness levels)
    texp = []
    for file in filelist:
        texp.append(pyfits.getval(file, 'exptime'))

    #find unique times
    uniq_times = np.unique(texp)

    #prepare some arrays
    signal_q1 = np.array([])
    signal_q2 = np.array([])
    signal_q3 = np.array([])
    signal_q4 = np.array([])
    variance_q1 = np.array([])
    variance_q2 = np.array([])
    variance_q3 = np.array([])
    variance_q4 = np.array([])

    #for all levels
    for t in uniq_times:
        sublist = np.array(filelist)[texp == t]
        f_q1, f_q2, f_q3, f_q4, v_q1, v_q2, v_q3, v_q4 = get_flux_and_variance_pairs(sublist, MB)
        signal_q1 = np.append(signal_q1, f_q1)
        signal_q2 = np.append(signal_q2, f_q2)
        signal_q3 = np.append(signal_q3, f_q3)
        signal_q4 = np.append(signal_q4, f_q4)
        variance_q1 = np.append(variance_q1, v_q1)
        variance_q2 = np.append(variance_q2, v_q2)
        variance_q3 = np.append(variance_q3, v_q3)
        variance_q4 = np.append(variance_q4, v_q4)

    #have to do this for each quadrant individually
    g1 = measure_gain_from_slope(signal_q1, variance_q1, debug_level=1)
    g2 = measure_gain_from_slope(signal_q2, variance_q2, debug_level=1)
    g3 = measure_gain_from_slope(signal_q3, variance_q3, debug_level=1)
    g4 = measure_gain_from_slope(signal_q4, variance_q4, debug_level=1)

    return g1,g2,g3,g4





def get_offset_and_readnoise_from_overscan(img):
    os1, os2, os3, os4 = extract_overscan_region(img)

    return





def get_offset_and_readnoise_from_bias_frames(bias_list, debug_level=0, simu=False, timit=False):

    if timit:
        start_time = time.time()

    print('Determining offset levels and read-out noise properties for 4 quadrants...')

    img = pyfits.getdata(bias_list[0])

    #do some formatting things for real observations
    if not simu:
        #bring to "correct" orientation
        img = correct_orientation(img)
        #remove the overscan region, which looks crap for actual bias images
        img = crop_overscan_region(img)


    ny,nx = img.shape

    #define four quadrants via masks
    q1,q2,q3,q4 = make_quadrant_masks(nx,ny)

    #co-add all bias frames
    #MB = create_master_img(bias_list, imgtype='bias', with_errors=False, savefiles=False, remove_outliers=True)

    #prepare arrays
    #means_q1 = []
    medians_q1 = []
    sigs_q1 = []
    #means_q2 = []
    medians_q2 = []
    sigs_q2 = []
    #means_q3 = []
    medians_q3 = []
    sigs_q3 = []
    #means_q4 = []
    medians_q4 = []
    sigs_q4 = []

    #first get mean / median for all bias images (per quadrant)
    for name in bias_list:
        img = pyfits.getdata(name)
        if not simu:
            #bring to "correct" orientation
            img = correct_orientation(img)
            #remove the overscan region, which looks crap for actual bias images
            img = crop_overscan_region(img)
        #means_q1.append(np.nanmean(img[q1]))
        medians_q1.append(np.nanmedian(img[q1]))
        #means_q2.append(np.nanmean(img[q2]))
        medians_q2.append(np.nanmedian(img[q2]))
        #means_q3.append(np.nanmean(img[q3]))
        medians_q3.append(np.nanmedian(img[q3]))
        #means_q4.append(np.nanmean(img[q4]))
        medians_q4.append(np.nanmedian(img[q4]))

    # now get sigma of RON for ALL DIFFERENT COMBINATIONS of length 2 of the images in 'bias_list'
    # by using the difference images we are less susceptible to funny pixels (hot, warm, cosmics, etc.)
    list_of_combinations = list(combinations(bias_list, 2))
    for (name1,name2) in list_of_combinations:
        # read in observations and bring to right format
        img1 = pyfits.getdata(name1)
        img2 = pyfits.getdata(name2)
        if not simu:
            img1 = correct_orientation(img1)
            img1 = crop_overscan_region(img1)
            img2 = correct_orientation(img2)
            img2 = crop_overscan_region(img2)
        #take difference and do sigma-clipping
        diff = img1.astype(long) - img2.astype(long)
        sigs_q1.append(np.nanstd(sigma_clip(diff[q1], 5))/np.sqrt(2))
        sigs_q2.append(np.nanstd(sigma_clip(diff[q2], 5))/np.sqrt(2))
        sigs_q3.append(np.nanstd(sigma_clip(diff[q3], 5))/np.sqrt(2))
        sigs_q4.append(np.nanstd(sigma_clip(diff[q4], 5))/np.sqrt(2))

    #now average over all images
    #allmeans = np.array([np.median(medians_q1), np.median(medians_q2), np.median(medians_q3), np.median(medians_q4)])
    allmedians = np.array([np.median(medians_q1), np.median(medians_q2), np.median(medians_q3), np.median(medians_q4)])
    allsigs = np.array([np.median(sigs_q1), np.median(sigs_q2), np.median(sigs_q3), np.median(sigs_q4)])

#     #convert to units of electrons rather than ADUs by muliplying with the gain (which has units of e-/ADU)
#     allmedians *= gain
#     allsigs *= gain

    if debug_level >= 1:
        #plot stuff
        print('plot the distributions for the four quadrants maybe!?!?!?')

    print('Done!!!')
    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

    return allmedians,allsigs





def make_master_bias_and_ronmask(offsets, rons, nx, ny, savefiles=False, path=None, timit=False):
    """
    This routine creates a 1-level or 4-level master bias frame of size (ny,nx)
    
    INPUT:
    'offsets'    : offset/bias levels as measured by "get_offset_and_readnoise_from_bias_frames" (either 1-element or 4-element)
    'rons'       : read-out noise level(s) from "get_offset_and_readnoise_from_bias_frames" 
    'nx'         : image dimensions
    'ny'         : image dimensions
    'savefiles'  : boolean - do you want to save the master bias to a fits file?
    'path'       : path to the output file directory (only needed if savefile is set to TRUE)
    'timit'      : boolean - do you want to measure execution run time?
    
    OUTPUT:
    'MB'       : master bias image
    'ronmask'  : read-out noise mask (or ron-image really...)
    """

    if timit:
        start_time = time.time()

    nq = len(offsets)

    if nq == 1:
        MB = np.ones((ny,nx)) * offsets
        ronmask = np.ones((ny,nx)) * rons
    elif nq == 4:
        MB = np.ones((ny,nx))
        ronmask = np.ones((ny,nx))
        q1,q2,q3,q4 = make_quadrant_masks(nx,ny)
        for q,offset in zip([q1,q2,q3,q4],offsets):
            MB[q] = MB[q] * offset
        for q,RON in zip([q1,q2,q3,q4],rons):
            ronmask[q] = ronmask[q] * RON
    else:
        print('ERROR: "offsets" must either be a scalar (for single-port readout) or a 4-element array/list (for four-port readout)!')
        return

    if savefiles:
        if path is None:
            print('ERROR: output file directory not provided!!!')
            return
        else:
            #write master bias to file
            pyfits.writeto(path+'master_bias.fits', MB, clobber=True)
#             pyfits.setval(path+'master_bias.fits', 'COMMENT', value='   Master Bias frame created from '+str(nq)+' offset/bias levels')
#             pyfits.setval(path+'master_bias.fits', 'GAIN', value=gain, comment='gain in units of e-/ADU')
#             pyfits.setval(path+'master_bias.fits', 'UNITS', value='ELECTRONS')
            pyfits.setval(path+'master_bias.fits', 'UNITS', value='ADU')
            pyfits.setval(path+'master_bias.fits', 'HISTORY', value='   master BIAS frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)')
            if nq == 1:
                pyfits.setval(path+'master_bias.fits', 'OFFSET', value=offsets)
                pyfits.setval(path+'master_bias.fits', 'RNOISE', value=rons)
            elif nq == 4:
                pyfits.setval(path+'master_bias.fits', 'OFFSET_1', value=offsets[0])
                pyfits.setval(path+'master_bias.fits', 'OFFSET_2', value=offsets[1])
                pyfits.setval(path+'master_bias.fits', 'OFFSET_3', value=offsets[2])
                pyfits.setval(path+'master_bias.fits', 'OFFSET_4', value=offsets[3])
                pyfits.setval(path+'master_bias.fits', 'RNOISE_1', value=rons[0])
                pyfits.setval(path+'master_bias.fits', 'RNOISE_2', value=rons[1])
                pyfits.setval(path+'master_bias.fits', 'RNOISE_3', value=rons[2])
                pyfits.setval(path+'master_bias.fits', 'RNOISE_4', value=rons[3])

            #now make read-out noise mask
            pyfits.writeto(path+'read_noise_mask.fits', ronmask, clobber=True)
#             pyfits.setval(path+'read_noise_mask.fits', 'COMMENT', value='   Read-noise frame created from '+str(nq)+' offset/bias levels')
#             pyfits.setval(path+'master_bias.fits', 'GAIN', value=gain, comment='gain in units of e-/ADU')
#             pyfits.setval(path+'master_bias.fits', 'UNITS', value='ELECTRONS')
            pyfits.setval(path+'read_noise_mask.fits', 'UNITS', value='ADU')
            pyfits.setval(path+'read_noise_mask.fits', 'HISTORY', value='   read-noise frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)')
            if nq == 1:
                pyfits.setval(path+'read_noise_mask.fits', 'OFFSET', value=offsets)
                pyfits.setval(path+'read_noise_mask.fits', 'RNOISE', value=rons)
            elif nq == 4:
                pyfits.setval(path+'read_noise_mask.fits', 'OFFSET_1', value=offsets[0])
                pyfits.setval(path+'read_noise_mask.fits', 'OFFSET_2', value=offsets[1])
                pyfits.setval(path+'read_noise_mask.fits', 'OFFSET_3', value=offsets[2])
                pyfits.setval(path+'read_noise_mask.fits', 'OFFSET_4', value=offsets[3])
                pyfits.setval(path+'read_noise_mask.fits', 'RNOISE_1', value=rons[0])
                pyfits.setval(path+'read_noise_mask.fits', 'RNOISE_2', value=rons[1])
                pyfits.setval(path+'read_noise_mask.fits', 'RNOISE_3', value=rons[2])
                pyfits.setval(path+'read_noise_mask.fits', 'RNOISE_4', value=rons[3])

    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

    return MB,ronmask





def make_master_dark(dark_list, MB, scalable=False, savefile=True, path=None, simu=False, timit=False):

    #CMB - I removed the 'return_errors' keyword functionality, don't think that was quite right
    #also removed 'ronmask' from INPUT

    if timit:
        start_time = time.time()

    #prepare arrays
    allimg = []
#     allerr = []

    #loop over all files in "dark_list"
    for name in dark_list:
        #read in dark image
        img = pyfits.getdata(name)
        if not simu:
            #bring to "correct" orientation
            img = correct_orientation(img)
            #remove the overscan region
            img = crop_overscan_region(img)
#         #calculate noise
#         err_img = np.sqrt(img.astype(float) + ronmask*ronmask)   #this is still in ADU
        #subtract master bias and store in list (ignoring tiny uncertainties (~<0.01 ADU) in the bias level)
        img_bc = img - MB
        allimg.append(img_bc)
#         #adjust errors and store in list
#         err_img_bc = np.sqrt(err_img*err_img + ronmask*ronmask)   #this is still in ADU
#        #NO! the RON is not the error in the bias level; so don't adjust error, just save it
#        allerr.append(err_img)

    #get median image
    MD = np.median(np.array(allimg), axis=0)
#     err_summed = np.sqrt(np.sum((np.array(allerr)**2),axis=0))
#     err_MD = err_summed / len(dark_list)

    #re-normalize to texp=1s to make it "scalable"
    texp = pyfits.getval(dark_list[0], 'exptime')
    if scalable:
        MD = MD / texp
#         #but can't have that smaller than the RON!?!?!? this is taken care of in "correct_for_bias_and_dark(_from_filename)"
#         err_MD = err_MD / texp

    if savefile:
        if path is None:
            print('WARNING: output file directory not provided!!!')
            print('Using same directory as input file...')
            dum = dark_list[0].split('/')
            path = dark_list[0][0:-len(dum[-1])]
        if scalable:
            outfn = path+'master_dark_scalable.fits'
        else:
            outfn = path+'master_dark_t'+str(int(np.round(texp,0)))+'.fits'
        #get header from master BIAS frame
        h = pyfits.getheader(path+'master_bias.fits')
        h['HISTORY'] = '   MASTER DARK frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
        h['EXPTIME'] = (texp, 'exposure time [s]')
        if scalable:
            h['COMMENT'] = 're-normalized to texp=1s to make it scalable'
            h['EXPTIME'] = (1., 'exposure time [s]; re-normalized to 1s; originally '+str(np.round(texp,2))+'s')
        pyfits.writeto(outfn, MD, h, clobber=True)
#         if return_errors:
#             h_err = h.copy()
#             h_err['HISTORY'] = 'estimated uncertainty in MASTER DARK frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
#             pyfits.append(outfn, err_MD, h_err, clobber=True)

    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

#     if return_errors:
#         return MD,err_MD
#     else:
    return MD





def correct_for_bias_and_dark(img, MB, MD, scalable=False, texp=None, timit=False):

    #CMB - I removed the 'ronmask' and 'err_MD' INPUTs

    if timit:
        start_time = time.time()

#     #(0) get error estimate for image
#     err_img = np.sqrt(img.astype(float) + ronmask*ronmask)

    #(1) BIAS SUBTRACTION
    #bias-corrected_image
    bc_img = img - MB
#     #adjust errors accordingly
#     err_bc_img = np.sqrt(err_img*err_img + ronmask*ronmask)

    #(2) DARK SUBTRACTION
    #if the darks have a different exposure time then the images you are trying to correct, we need to re-scale the master dark
    if scalable:
        if texp is not None:
            MD = MD * texp
#             #cannot have an error estimate lower than the read-out noise; this is dodgy but don't know what else to do
#             err_MD = np.maximum(err_MD * texp, ronmask)
        else:
            print('ERROR: "texp" has to be provided when "scalable" is set to TRUE')
            return
    #dark- & bias-corrected image
    dc_bc_img = bc_img - MD
#     #adjust errors accordingly
#     err_dc_bc_img = np.sqrt(err_bc_img*err_bc_img + err_MD*err_MD)

    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

#     return dc_bc_img, err_dc_bc_img
    return dc_bc_img





def correct_for_bias_and_dark_from_filename(imgname, MB, MD, scalable=False, texp=None, savefile=False, path=None, simu=False, timit=False):
    """
    
    #CMB - I removed the 'ronmask' and 'err_MD' INPUTs
    
    clone of "correct_for_bias_and_dark", but this one allows us to save output files
    """
    if timit:
        start_time = time.time()

    #(0) read in raw image and get error estimate for image
    img = pyfits.getdata(imgname)
    if not simu:
        #bring to "correct" orientation
        img = correct_orientation(img)
        #remove the overscan region
        img = crop_overscan_region(img)
#     err_img = np.sqrt(img.astype(float) + ronmask*ronmask)

    #(1) BIAS SUBTRACTION
    #bias-corrected_image
    bc_img = img - MB
#     #adjust errors accordingly
#     err_bc_img = np.sqrt(err_img*err_img + ronmask*ronmask)

    #(2) DARK SUBTRACTION
    #if the darks have a different exposure time then the images you are trying to correct, we need to re-scale the master dark
    if scalable:
        if texp is not None:
            MD = MD * texp
#             #cannot have an error estimate lower than the read-out noise; this is dodgy but don't know what else to do
#             err_MD = np.maximum(err_MD * texp, ronmask)
        else:
            print('ERROR: "texp" has to be provided when "scalable" is set to TRUE')
            return
    #dark- & bias-corrected image
    dc_bc_img = bc_img - MD
#     #adjust errors accordingly
#     err_dc_bc_img = np.sqrt(err_bc_img*err_bc_img + err_MD*err_MD)

    #if desired, write bias- & dark-corrected image and error array to fits files
    if savefile:
        dum = imgname.split('/')
        dum2 = dum[-1].split('.')
        shortname = dum2[0]
        if path is None:
            print('WARNING: output file directory not provided!!!')
            print('Using same directory as input file...')
            path = imgname[0:-len(dum[-1])]
        #outfn = path+shortname+'_bias_and_dark_corrected.fits'
        outfn = path+shortname+'_BD.fits'
        #get header from the original image FITS file
        h = pyfits.getheader(imgname)
        h['HISTORY'] = '   BIAS- & DARK-corrected image - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
        pyfits.writeto(outfn, dc_bc_img, h, clobber=True)
#         h_err = h.copy()
#         h_err['HISTORY'] = 'estimated uncertainty in BIAS- & DARK-corrected image - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
#         pyfits.append(outfn, err_dc_bc_img, h_err, clobber=True)

    if timit:
        print('Time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

#     return dc_bc_img, err_dc_bc_img
    return dc_bc_img





def process_whites(white_list, MB=None, ronmask=None, MD=None, scalable=False, fancy=False, clip=5., savefile=True, saveall=False, diffimg=False, path=None, timit=False):

    #CMB - removed err_MD keyword input

    if timit:
        start_time = time.time()

    #if the darks have a different exposure time than the whites, then we need to re-scale the master dark
    texp = pyfits.getval(white_list[0], 'exptime')

    #if INPUT arrays are not given, read them from default files
    if path is None:
        print('WARNING: output file directory not provided!!!')
        print('Using same directory as input file...')
        dum = white_list[0].split('/')
        path = white_list[0][0:-len(dum[-1])]
    if MB is None:
        #no need to fix orientation, this is already a processed file
        MB = pyfits.getdata(path+'master_bias.fits')
    if ronmask is None:
        #no need to fix orientation, this is already a processed file
        ronmask = pyfits.getdata(path+'read_noise_mask.fits')
    if MD is None:
        if scalable:
            #no need to fix orientation, this is already a processed file
            MD = pyfits.getdata(path+'master_dark_scalable.fits', 0)
#             err_MD = pyfits.getdata(path+'master_dark_scalable.fits', 1)
        else:
            #no need to fix orientation, this is already a processed file
            MD = pyfits.getdata(path+'master_dark_t'+str(int(np.round(texp,0)))+'.fits', 0)
#             err_MD = pyfits.getdata(path+'master_dark_t'+str(int(np.round(texp,0)))+'.fits', 1)


    #prepare arrays
    allimg = []
    allerr = []

    #loop over all files in "white_list"; correct for bias and darks on the fly
    for n,fn in enumerate(white_list):
        #call routine that does all the bias and dark correction stuff and proper error treatment
        img = correct_for_bias_and_dark_from_filename(fn, MB, MD, scalable=scalable, texp=texp, savefile=saveall, path=path, timit=timit)     #these are now bias- & dark-corrected images; units are still ADUs
        allimg.append(img)
#         allerr.append(err)
        allerr.append( np.sqrt(img + ronmask*ronmask) )


    #########################################################################
    ### now we do essentially what "CREATE_MASTER_IMG" does for whites... ###
    #########################################################################
    #add individual-image errors in quadrature (need it either way, not only for fancy method)
    err_summed = np.sqrt(np.sum((np.array(allerr)**2),axis=0))
    #get median image
    medimg = np.median(np.array(allimg), axis=0)

    if fancy:
        #need to create a co-added frame if we want to do outlier rejection the fancy way
        summed = np.sum((np.array(allimg)),axis=0)
        if diffimg:
            diff = np.zeros(summed.shape)

        master_outie_mask = np.zeros(summed.shape, dtype='int')

        #make sure we do not have any negative pixels for the sqrt
        medimgpos = medimg.copy()
        medimgpos[medimgpos < 0] = 0.
        med_sig_arr = np.sqrt(medimgpos + ronmask*ronmask)       #expected STDEV for the median image (from LB Eq 2.1); still in ADUs
        for n,img in enumerate(allimg):
            #outie_mask = np.abs(img - medimg) > clip*med_sig_arr
            outie_mask = (img - medimg) > clip*med_sig_arr      #do we only want HIGH outliers, ie cosmics?
            #save info about which image contributes the outlier pixel using unique binary numbers technique
            master_outie_mask += (outie_mask * 2**n).astype(int)
        #see which image(s) produced the outlier(s) and replace outies by mean of pixel value from remaining images
        n_outie = np.sum(master_outie_mask > 0)
        print('Correcting '+str(n_outie)+' outliers...')
        #loop over all outliers
        for i,j in zip(np.nonzero(master_outie_mask)[0],np.nonzero(master_outie_mask)[1]):
            #access binary numbers and retrieve component(s)
            outnum = binary_indices(master_outie_mask[i,j])   #these are the indices (within allimg) of the images that contain outliers
            dumix = np.arange(len(white_list))
            #remove the images containing the outliers in order to compute mean from the remaining images
            useix = np.delete(dumix,outnum)
            if diffimg:
                diff[i,j] = summed[i,j] - ( len(outnum) * np.mean( np.array([allimg[q][i,j] for q in useix]) ) + np.sum( np.array([allimg[q][i,j] for q in useix]) ) )
            #now replace value in master image by the sum of all pixel values in the unaffected pixels
            #plus the number of affected images times the mean of the pixel values in the unaffected images
            summed[i,j] = len(outnum) * np.mean( np.array([allimg[q][i,j] for q in useix]) ) + np.sum( np.array([allimg[q][i,j] for q in useix]) )
        #once we have finished correcting the outliers, we want to "normalize" (ie divide by number of frames) the master image and the corresponding error array
        master = summed / len(white_list)
        err_master = err_summed / len(white_list)
    else:
        #ie not fancy, just take the median image to remove outliers
        medimg = np.median(np.array(allimg), axis=0)
        #now set master image equal to median image
        master = medimg.copy()
        #estimate of the corresponding error array (estimate only!!!)
        err_master = err_summed / len(white_list)


    #now save master white to file
    if savefile:
        outfn = path+'master_white.fits'
        pyfits.writeto(outfn, master, clobber=True)
        pyfits.setval(outfn, 'HISTORY', value='   MASTER WHITE frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)')
        pyfits.setval(outfn, 'EXPTIME', value=texp, comment='exposure time [s]')
        if fancy:
            pyfits.setval(outfn, 'METHOD', value='fancy', comment='method to create master white and remove outliers')
        else:
            pyfits.setval(outfn, 'METHOD', value='median', comment='method to create master white and remove outliers')
        h = pyfits.getheader(outfn)
        h_err = h.copy()
        h_err['HISTORY'] = 'estimated uncertainty in MASTER WHITE frame - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
        pyfits.append(outfn, err_master, h_err, clobber=True)

    #also save the difference image if desired
    if diffimg:
        hdiff = h.copy()
        hdiff['HISTORY'] = '   MASTER WHITE DIFFERENCE IMAGE - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
        pyfits.writeto(path+'master_white_diffimg.fits', diff, hdiff, clobber=True)

    if timit:
        print('Total time elapsed: '+str(np.round(time.time() - start_time,1))+' seconds')

    return master, err_master





def bias_subtraction(imglist, MB, noneg=True, savefile=True):
    """
    DUMMY ROUTINE; NOT CURRENTLY USED
    """
    for file in imglist:
        img = pyfits.getdata(file)
        mod_img = img - MB
        if noneg:
            mod_img[mod_img < 0] = 0.
        if savefile:
            dum = file.split('/')
            path = file[:-len(dum[-1])]
            h = pyfits.getheader(file)
            pyfits.writeto(path+'bc_'+dum[-1], mod_img, h, clobber=True)
    return


def dark_subtraction(imglist, MD, noneg=True, savefile=True):
    """
    DUMMY ROUTINE; NOT CURRENTLY USED
    """
    for file in imglist:
        img = pyfits.getdata(file)
        mod_img = img - MD
        if noneg:
            mod_img[mod_img < 0] = 0.
        if savefile:
            dum = file.split('/')
            path = file[:-len(dum[-1])]
            h = pyfits.getheader(file)
            pyfits.writeto(path+'dc_'+dum[-1], mod_img, h, clobber=True)
    return



