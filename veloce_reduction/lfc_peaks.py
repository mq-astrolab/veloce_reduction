'''
Created on 19 Dec. 2018

@author: christoph
'''

import glob
import time
import numpy as np
# import matplotlib.pyplot as plt
# import astropy.io.fits as pyfits

from readcol import readcol



# path = '/Volumes/BERGRAID/data/veloce/lfc_peaks/'
# path = '/Users/christoph/data/lfc_peaks/'
#
# files = glob.glob(path + '*olc.nst')
#
# id, y0, x0, mag, err_mag, skymod, niter, chi, sharp, y_err, x_err = readcol(files[0], twod=False, skipline=2)
# id, y, x, mag, err_mag, skymod, niter, chi, sharp, y_err, x_err = readcol(files[1], twod=False, skipline=2)
# x0 = 4112. - x0   # flipped
# x = 4112. - x
# y0 = y0 - 54.     # 53 overscan pixels either side and DAOPHOT counting from 1?
# y = y - 54.
#
# test_x0 = x0[(x0 > 1500) & (x0 < 1800) & (y0 > 1500) & (y0 < 1800)]
# test_y0 = y0[(x0 > 1500) & (x0 < 1800) & (y0 > 1500) & (y0 < 1800)]
# test_x = x[(x > 1500) & (x < 1800) & (y > 1500) & (y < 1800)]
# test_y = y[(x > 1500) & (x < 1800) & (y > 1500) & (y < 1800)]
#
# x0 = test_x0.copy()
# x = test_x.copy()
# y0 = test_y0.copy()
# y = test_y.copy()



def find_affine_transformation_matrix(x, y, x0, y0, eps=0.5, timit=False):
    '''
    Finds the affine transformation matrix that describes the co-ordinate transformation from (x0,y0) --> (x,y)
    
    x0  - reference x
    y0  - reference y
    x   - observed x
    y   - observed y
    '''
    
    if timit:
        start_time = time.time()
    
    # get numpy arrays from list of (x,y)-tuples in the shape (2,***)
    ref_peaks_xy_list = [(xpos, ypos) for xpos,ypos in zip(x0,y0)]
    ref_peaks_xy = np.array(ref_peaks_xy_list).T     # not really needed
    obs_peaks_xy_list = [(xpos, ypos) for xpos,ypos in zip(x,y)]
    obs_peaks_xy = np.array(obs_peaks_xy_list).T
    
    #now we need to match the peaks
    good_ref_peaks = []
    good_obs_peaks = []
    for n,refpeak in enumerate(ref_peaks_xy_list):
        # print(n)
        shifted_obs_peaks = obs_peaks_xy - np.expand_dims(np.array(refpeak), axis=1)
        distance = np.sqrt(shifted_obs_peaks[0,:]**2 + shifted_obs_peaks[1,:]**2)

        if np.sum(distance < eps) > 0:
            if np.sum(distance < eps) > 1:
                print('FUGANDA: ',refpeak)
                print('There is probably a cosmic really close to an LFC peak - skipping this peak...')
            else:
                good_ref_peaks.append(refpeak)
                good_obs_peaks.append((obs_peaks_xy[0,distance < eps], obs_peaks_xy[1,distance < eps]))

        # print(n, refpeak, np.sum(distance < eps))

    #go to homogeneous coordinates (ie add a z-component equal to 1, so that we can include translation into the matrix)
    good_ref_peaks_xyz = np.hstack((np.array(good_ref_peaks), np.expand_dims(np.repeat(1, len(good_ref_peaks)), axis=1)))
    good_obs_peaks_xyz = np.hstack((np.squeeze(np.array(good_obs_peaks)), np.expand_dims(np.repeat(1, len(good_obs_peaks)), axis=1)))
    
    # np.linalg.lstsq(r1,r2) solves matrix equation M*r1 = r2  (note that linalg.lstsq wants row-vectors)
    # i.e.: good_obs_peaks_xyz ~= np.dot(good_ref_peaks_xyz, M)
    M, res, rank, s = np.linalg.lstsq(good_ref_peaks_xyz, good_obs_peaks_xyz, rcond=None)

    if timit:
        print('Time elapsed: ' + str(np.round(time.time() - start_time,1)) + ' seconds')
    
    #return affine transformation matrix  (note this is the transpose of the usual form eg as listed in wikipedia)
    return M



#   if   x' = x * M
# then   x  = x' * M_inv



def divide_lfc_peaks_into_orders(x, y, tol=5):

    # read rough LFC traces
    lfc_path = '/Users/christoph/OneDrive - UNSW/lfc_peaks/'
    pid = np.load(lfc_path + 'lfc_P_id.npy').item()

    peaks = {}

    for o in sorted(pid.keys()):
        y_dist = y - pid[o](x)
        peaks[o] = zip(x[np.abs(y_dist) < tol], y[np.abs(y_dist) < tol])

    return peaks


