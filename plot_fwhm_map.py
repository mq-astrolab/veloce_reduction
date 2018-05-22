'''
Created on 28 Mar. 2018

@author: christoph
'''

import numpy as np
from mpl_toolkits import mplot3d

from veloce_reduction.helper_functions import polyfit2d,polyval2d
from veloce_reduction.find_laser_peaks_2D import *


#imgname = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_laser_comb.fit'
#imgname = '/Users/christoph/UNSW/veloce_spectra/Mar02/clean_thorium-17-12.fits'
path = '/Users/christoph/UNSW/veloce_spectra/Mar02/'
imgname = path+'master_etalon.fits'
img = pyfits.getdata(imgname).T + 1

ny,nx = img.shape

#peakshapes = find_laser_peaks_2D(img,smooth=True)
#
#or
#
#peakshapes = np.load('/Users/christoph/UNSW/fwhm_maps/peakshapes_laser_simu.npy').item()


#prepare FWHM arrays as the z-coordinates for the fit
fwhm_x = np.array(peakshapes['x_sigma']) * 2. * np.sqrt(2. * np.log(2.))     #convert sigma to FWHM
fwhm_y = np.array(peakshapes['y_sigma']) * 2. * np.sqrt(2. * np.log(2.))     #convert sigma to FWHM
#renormalize x and y to [-1,+1] so that small inaccuracies in fitted polynomial coefficients have no big effect
xnorm = (np.array(peakshapes['x0']) / ((nx-1)/2.)) - 1.
ynorm = (np.array(peakshapes['y0']) / ((ny-1)/2.)) - 1.
#now fit a 2D polynomial surface with the z-values being either x-FWHM or y-FWHM
px = polyfit2d(xnorm, ynorm, fwhm_x, order=3)
py = polyfit2d(xnorm, ynorm, fwhm_y, order=3)

#construct model
xx = np.arange(nx)          
xxn = (xx / ((nx-1)/2.)) - 1.
yy = np.arange(ny)
yyn = (yy / ((ny-1)/2.)) - 1.
X,Y = np.meshgrid(xxn,yyn)
x_map = polyval2d(X,Y,px)
y_map = polyval2d(X,Y,py)

