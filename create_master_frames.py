'''
Created on 4 Apr. 2018

@author: christoph
'''

import numpy as np
import astropy.io.fits as pyfits
import time

from veloce_reduction.helper_functions import binary_indices


# #finding files (needs "import glob, os")
# path = '/Users/christoph/UNSW/veloce_spectra/temp/'
# biaslist = glob.glob(path+"*bias*")
# darklist = glob.glob(path+"*dark*")
# whitelist = glob.glob(path+"*white*")
# allobslist = glob.glob(path+'*.fits')
# dumlist = glob.glob(path+'J*')



def create_master_img(imglist, imgtype='', RON=4., gain=1., clip=5., asint=False, savefile=False, norm=True, scalable=False, remove_outliers=True, diffimg=False, noneg=False, timit=False):
    """
    This routine co-adds spectra from a given input list. It can also remove cosmics etc. by removing outlier pixels that deviate by more than a certain
    number of sigmas from the median across all images with the mean pixel value of the remaining pixels.
    
    INPUT:
    'imglist'           : list of filenames of images to co-add
    'imgtype'           : ['bias' / 'dark' / 'white'] are valid options
    'RON'               : read-out noise in electrons
    'gain'              : camera gain in electrons per ADU
    'clip'              : threshold for outlier identification (in sigmas above/below median)
    'asint'             : boolean - do you want the master image to be rounded to the nearest integer?
    'savefile'          : boolean - do you want to save the master image as a FITS file?
    'norm'              : boolean - do you want to normalize the master image (ie divide by number of images)?
    'scalable'          : boolean - do you want to make the master image scalable (ie normalize to t_exp = 1s)
    'remove_outliers'   : boolean - do you want to remove outlier pixels (eg cosmics) with the median of the remaining images?
    'diffimg'           : boolean - do you want to save the difference image to a fits file? only works if 'remove_outliers' is also set to TRUE
    'noneg'             : boolean - do you want to allow negative pixel values?
    'timit'             : boolean - measure time taken for completion of function?
    
    OUTPUT:
    'master'            : the master image
    """
    
    if timit:
        start_time = time.time()
    
    print('Creating master image from: '+str(len(imglist))+' '+imgtype.upper()+' images')
    
    if savefile or diffimg:
        normstring = ''
        intstring = ''
        outie_string = ''
        noneg_string = ''
    
    while imgtype.lower() not in ['white', 'w', 'dark', 'd', 'bias', 'b']:
                imgtype = raw_input("WARNING: Image type not specified! What kind of images are they ['(b)ias' / '(d)ark' / '(w)hite']: ")
    
    #proceed if list is not empty
    if len(imglist) > 0:
        
        if remove_outliers:
            allimg = []
            outie_string = '_outliers_removed'
        
        for n,file in enumerate(imglist):
            #img = pyfits.getdata(file).T
            img = pyfits.getdata(file)
            if remove_outliers:
                allimg.append(img)
            if n==0:
                h = pyfits.getheader(file)
                master = img.copy().astype(float)
                if remove_outliers:
                    ny,nx = img.shape
                    master_outie_mask = np.zeros((ny,nx),dtype='int')
                    if diffimg:
                        diff = np.zeros((ny,nx),dtype='float')
            else:
                master += img
        #make sure we do not have any negative pixels
        #master[master < 0] = 0.   #NO!!!!!
        
        if remove_outliers:
            medimg = np.median(np.array(allimg),axis=0)
            #for bias and dark frames just use the median image; for whitesd use something more sophisticated
            if imgtype[0] in ['b', 'd']:
                master = medimg
            #do THIS for whites
            else:
                #make sure we do not have any negative pixels for the sqrt
                medimgpos = medimg.copy()
                medimgpos[medimg < 0] = 0.
                med_sig_arr = np.sqrt(RON*RON + gain*medimgpos)       #expected STDEV for the median image (from LB Eq 2.1)
    #             rms_arr = np.std(np.array(allimg),axis=0)             #but in the very low SNR regime, this takes over, as med_sig_arr will just be RON, and flag a whole bunch of otherwise good pixels
    #             mydev = np.maximum(med_sig_arr,rms_arr)
    #             #check that the median pixel value does not deviate significantly from the minimum pixel value (unlikely!!!)
    #             minimg = np.amin(allimg,axis=0)
    #             min_sig_arr = np.sqrt(RON*RON + gain*minimg)       #expected STDEV for the minimum image (from LB Eq 2.1)
    #             fu_mask = medimg - minimg > clip*min_sig_arr
                for n,img in enumerate(allimg): 
                    #outie_mask = np.abs(img - medimg) > clip*med_sig_arr
                    outie_mask = img - medimg > clip*med_sig_arr      #do we only want HIGH outliers, ie cosmics?
                    #save info about which image contributes the outlier pixel using unique binary numbers technique
                    master_outie_mask += (outie_mask * 2**n).astype(int)          
                #see which image(s) produced the outlier(s) and replace outies by mean of pixel value from remaining images
                n_outie = np.sum(master_outie_mask > 0)
                print('Correcting '+str(n_outie)+' outliers...')
                #loop over all outliers
                for i,j in zip(np.nonzero(master_outie_mask)[0],np.nonzero(master_outie_mask)[1]):
                    #access binary numbers and retrieve component(s)
                    outnum = binary_indices(master_outie_mask[i,j])   #these are the indices (within allimg) of the images that contain outliers
                    dumix = np.arange(len(imglist))
                    #remove the images containing the outliers in order to compute mean from the remaining images
                    useix = np.delete(dumix,outnum)
                    if diffimg:
                        diff[i,j] = master[i,j] - ( len(outnum) * np.mean( np.array([allimg[q][i,j] for q in useix]) ) + np.sum( np.array([allimg[q][i,j] for q in useix]) ) )
                    #now replace value in master image by the sum of all pixel values in the unaffected pixels 
                    #plus the number of affected images times the mean of the pixel values in the unaffected images
                    master[i,j] = len(outnum) * np.mean( np.array([allimg[q][i,j] for q in useix]) ) + np.sum( np.array([allimg[q][i,j] for q in useix]) )    
                    master = master / len(imglist)       
                       
        #if not remove outliers, at least divide by number of frames
        else:
            master = master / len(imglist) 
            
#         if norm:
#             master = master / len(imglist)
#             normstring='_norm'
            
        if scalable:
#             if not norm:
#                 print('ERROR: "scalable" option has to be set together with "norm" option!!!')
#                 return
            texp = pyfits.getval(imglist[0], 'exptime')
            master = master / texp
            normstring = normstring + '_scalable'
        
        if noneg:
            master[master < 0] = 0.
            noneg_string = '_noneg'
        
        if asint:
            master = np.round(master).astype(int)    
            intstring='_int'
        
        if diffimg:
            hdiff = h.copy()
            dum = file.split('/') 
            path = file[:-len(dum[-1])]
            hdiff['modhist'] = 'DIFFERENCE IMAGE - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
            pyfits.writeto(path+'master_'+imgtype.lower()+normstring+intstring+'_diffimg.fits', diff, hdiff, clobber=True) 
            
        if savefile:
            dum = file.split('/') 
            path = file[:-len(dum[-1])]
#             while imgtype.lower() not in ['white','dark','bias']:
#                 imgtype = raw_input("WARNING: Image type not specified! What kind of images are they ['white' / 'dark' / 'bias']: ")
            h['modhist'] = 'MASTER '+imgtype.upper()+' - created '+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+' (GMT)'
            pyfits.writeto(path+'master_'+imgtype.lower()+normstring+outie_string+intstring+noneg_string+'.fits', master, h, clobber=True)                    
            
    else:
        print('WARNING: empty input list')    
        return

    print('DONE!!!')

    if timit:
        print('Elapsed time: '+str(np.round(time.time() - start_time,2))+' seconds')

    return master




##steps in a possible wrapper script:
#
# MW = create_master_img(whitelist)
# MD = create_master_img(darklist)
# MB = create_master_img(biaslist)
# bias_subtracted = bias_subtraction(dumlist,MB)
# dark_subtracted = dark_subtraction(dumlist,MD)







