'''
Created on 25 Jul. 2018

@author: christoph
'''

import astropy.io.fits as pyfits
import numpy as np

from veloce_reduction.calibration import correct_for_bias_and_dark_from_filename
from veloce_reduction.cosmic_ray_removal import remove_cosmics
from veloce_reduction.background import remove_background


def process_science_images(imglist, P_id, slit_height=25, MB=None, ronmask=None, MD=None, scalable=False, saveall=False, path=None, timit=False):
    """
    Process all science images. This includes:
    
    (1) bias and dark subtraction
    (2) cosmic ray removal 
    (3) background extraction and estimation
    
    (4) extraction of stripes
    (5) extraction of 1-dim spectra
    (6) wavelength solution
    """
    
    #####################################
    ### (1) bias and dark subtraction ###
    #####################################
    
    #if the darks have a different exposure time than the science images, then we need to re-scale the master dark
    texp = pyfits.getval(imglist[0], 'exptime')
    
    #if INPUT arrays are not given, read them from default files
    if path is None:
        print('WARNING: output file directory not provided!!!')
        print('Using same directory as input file...')
        dum = imglist[0].split('/')
        path = imglist[0][0:-len(dum[-1])]
    if MB is None:
        #no need to fix orientation, this is already a processed file [ADU]
        MB = pyfits.getdata(path+'master_bias.fits')
    if ronmask is None:
        #no need to fix orientation, this is already a processed file [e-]
        ronmask = pyfits.getdata(path+'read_noise_mask.fits')
    if MD is None:
        if scalable:
            #no need to fix orientation, this is already a processed file [e-]
            MD = pyfits.getdata(path+'master_dark_scalable.fits', 0)
#             err_MD = pyfits.getdata(path+'master_dark_scalable.fits', 1)
        else:
            #no need to fix orientation, this is already a processed file [e-]
            MD = pyfits.getdata(path+'master_dark_t'+str(int(np.round(texp,0)))+'.fits', 0)
#             err_MD = pyfits.getdata(path+'master_dark_t'+str(int(np.round(texp,0)))+'.fits', 1)
    
    
    for filename in imglist:
        #do some housekeeping with filenames
        dum = filename.split('/')
        dum2 = dum[-1].split('.')
        obsname = dum2[0]
        
        # (1) call routine that does all the bias and dark correction stuff and proper error treatment
        img = correct_for_bias_and_dark_from_filename(filename, MB, MD, gain=gain, scalable=False, texp=texp, savefile=saveall, path=path, timit=True)
        #err = np.sqrt(img + ronmask*ronmask)   #still in ADUs
        #TEMPFIX:
        err = np.sqrt(np.clip(img,0,None) + ronmask*ronmask)   #still in ADUs
        
        # (2) remove cosmic rays (ERROR REMAINS UNCHANGED)
        cosmic_cleaned_img = remove_cosmics(img, ronmask, obsname, path, Flim=3.0, siglim=5.0, maxiter=1, savemask=True, savefile=True, save_err=True, verbose=True, timit=True)
        #adjust errors?
        
        # (3) fit and remove background (ERROR REMAINS UNCHANGED)
        final_img = remove_background(cosmic_cleaned_img, P_id, obsname, path, degpol=5, slit_height=slit_height, save_bg=True, savefile=True, save_err=True, exclude_top_and_bottom=True, verbose=True, timit=True)
        #adjust errors?


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    return






























