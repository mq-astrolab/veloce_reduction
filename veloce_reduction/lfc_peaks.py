'''
Created on 19 Dec. 2018

@author: christoph
'''

import glob
import math
import time
import numpy as np
from readcol import readcol

path = '/Volumes/BERGRAID/data/veloce/lfc_peaks/'

files = glob.glob(path + '*olc.nst')

id, y0, x0, mag, err_mag, skymod, niter, chi, sharp, y_err, x_err = readcol(files[0], twod=False, skipline=2)
id, y, x, mag, err_mag, skymod, niter, chi, sharp, y_err, x_err = readcol(files[1], twod=False, skipline=2)


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
    ref_peaks_xy = np.array(ref_peaks_xy_list).T
    obs_peak_xy_list = [(xpos, ypos) for xpos,ypos in zip(x,y)]
    obs_peaks_xy = np.array(obs_peak_xy_list).T
    
    #now we need to match the peaks
    good_ref_peaks = []
    good_obs_peaks = []
    for refpeak in ref_peaks_xy_list[:10]:
        # this works but is SUUUUPER slow!!!
        shifted_obs_peaks = obs_peaks_xy - np.expand_dims(np.array(refpeak), axis=1)
        distance = np.zeros(shifted_obs_peaks.shape[1])
        for i in range(len(distance)):
            distance[i] = math.sqrt(shifted_obs_peaks[0,i]*shifted_obs_peaks[0,i] + shifted_obs_peaks[1,i]*shifted_obs_peaks[1,i])
            
        print(np.sum(distance < eps))
        
        
    #go to homogeneous coordinates (ie add a z-component equal to 1, so that we can include translation into the matrix)
    ref_peaks_xyz = np.vstack((ref_peaks_xy, np.expand_dims(np.repeat(1, ref_peaks_xy.shape[1]), axis=0)))
    obs_peaks_xyz = np.vstack((obs_peaks_xy, np.expand_dims(np.repeat(1, obs_peaks_xy.shape[1]), axis=0)))
    
    #solve matrix equation: M*r1 = r2  (note the transpose, as linalg.lstsq wants row-vectors)
    M, res, rank, s = np.linalg.lstsq(peaks1_xyz.T, peaks2_xyz.T)
    
    if timit:
        print('Time elapsed: ' + str(np.round(time.time() - start_time,1)) + ' seconds')
    
    #return affine transformation matrix  (note the transpose again, so that you can do :  new_points = np.dot(M.T,points) )
    return M.T












