'''
Created on 13 Apr. 2018

@author: christoph
'''

import astropy.io.fits as pyfits



def bias_subtraction(imglist, MB, noneg=True, savefile=True): 
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



