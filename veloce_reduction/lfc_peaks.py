'''
Created on 19 Dec. 2018

@author: christoph
'''

import glob
import time
import numpy as np
import matplotlib.pyplot as plt
# import astropy.io.fits as pyfits

from readcol import readcol
from veloce_reduction.veloce_reduction.helper_functions import find_nearest



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

    # go to homogeneous coordinates (ie add a z-component equal to 1, so that we can include translation into the matrix)
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



def check_transformation_scatter(lfc_files, M_list=None, nx=4112, ny=4202, n_sub=1, eps=0.5, return_residuals=True,
                                 ref_obsname='21sep30019', lfc_path = '/Users/christoph/OneDrive - UNSW/lfc_peaks/'):
    # INPUT:
    # 'lfc_files' - list of lfc files for which to check
    # 'M_list'    - corresponding list of calculated transformation matrices

    # WARNING: They obviously need to be in the same order. See how it's done at the start of "check_all_shifts_with_telemetry"

    if M_list is not None:
        assert len(lfc_files) == len(M_list), 'ERROR: list of files and list of matrices have different lengths!!!'

    # read reference LFC peak positions
    _, yref, xref, _, _, _, _, _, _, _, _ = readcol(lfc_path + ref_obsname + 'olc.nst', twod=False, skipline=2)
    xref = nx - xref
    yref = yref - 54.  # or 53??? but does not matter for getting the transformation matrix
    ref_peaks_xy_list = [(xpos, ypos) for xpos, ypos in zip(xref, yref)]

    all_delta_x_list = []
    all_delta_y_list = []

    # loop over all files
    for i,lfc_file in enumerate(lfc_files):

        print('Processing observation ' + str(i+1) + '/' + str(len(lfc_files)) + '...')

        # read observation LFC peak positions
        try:
            _, y, x, _, _, _, _, _, _, _, _ = readcol(lfc_file, twod=False, skipline=2)
        except:
            _, y, x, _, _, _, _, _, _ = readcol(lfc_file, twod=False, skipline=2)
        del _
        x = nx - x
        y = y - 54.  # or 53??? but does not matter for getting the transformation matrix
        obs_peaks_xy_list = [(xpos, ypos) for xpos, ypos in zip(x, y)]
        obs_peaks_xy = np.array(obs_peaks_xy_list).T

        if M_list is None:
            M = find_affine_transformation_matrix(xref, yref, x, y, timit=True, eps=2.)   # note that within "wavelength_solution" this is called Minv
        else:
            M = M_list[i]

        # now we need to match the peaks so we can compare the reference peaks with the (back-)transformed obs peaks
        good_ref_peaks = []
        good_obs_peaks = []
        for n, refpeak in enumerate(ref_peaks_xy_list):
            # print(n)
            shifted_obs_peaks = obs_peaks_xy - np.expand_dims(np.array(refpeak), axis=1)
            distance = np.sqrt(shifted_obs_peaks[0, :] ** 2 + shifted_obs_peaks[1, :] ** 2)

            if np.sum(distance < eps) > 0:
                if np.sum(distance < eps) > 1:
                    print('FUGANDA: ', refpeak)
                    print('There is probably a cosmic really close to an LFC peak - skipping this peak...')
                else:
                    good_ref_peaks.append(refpeak)
                    good_obs_peaks.append((obs_peaks_xy[0, distance < eps], obs_peaks_xy[1, distance < eps]))
            # print(n, refpeak, np.sum(distance < eps))

        # divide good_ref_peaks into several subsections for a more detailed investigation
        x_step = nx / np.sqrt(n_sub).astype(int)
        y_step = ny / np.sqrt(n_sub).astype(int)
        x_centres = np.arange(0.5 * x_step, (np.sqrt(n_sub).astype(int) + 0.5) * x_step, x_step)
        y_centres = np.arange(0.5 * y_step, (np.sqrt(n_sub).astype(int) + 0.5) * y_step, y_step)
        peak_subsection_id = []
        for refpeak in good_ref_peaks:
            # first, figure out which subsection this particular peak falls into
            xpos = refpeak[0]
            ypos = refpeak[1]
            nearest_x_ix = find_nearest(x_centres, xpos, return_index=True)
            nearest_y_ix = find_nearest(y_centres, ypos, return_index=True)
            # then save that information
            peak_subsection_id.append((nearest_x_ix, nearest_y_ix))

        # give each subsection a label
        subsection_id = []
        for j in range(np.sqrt(n_sub).astype(int)):
            for i in range(np.sqrt(n_sub).astype(int)):
                subsection_id.append((i, j))  # (x,y)

        # # divide chip into several subsections for a more detailed investigation
        # section_masks = []
        # section_indices = []
        # x_step = nx / np.sqrt(n_sub).astype(int)
        # y_step = ny / np.sqrt(n_sub).astype(int)
        # for j in range(np.sqrt(n_sub).astype(int)):
        #     for i in range(np.sqrt(n_sub).astype(int)):
        #         q = np.zeros((ny, nx), dtype='bool')
        #         q[j * y_step : (j+1) * y_step, i * x_step : (i+1) * x_step] = True
        #         section_masks.append(q)
        #         section_indices.append((i,j))   # (x,y)

        # go to homogeneous coordinates (ie add a z-component equal to 1, so that we can include translation into the matrix)
        good_ref_peaks_xyz = np.hstack((np.array(good_ref_peaks), np.expand_dims(np.repeat(1, len(good_ref_peaks)), axis=1)))
        good_obs_peaks_xyz = np.hstack((np.squeeze(np.array(good_obs_peaks)), np.expand_dims(np.repeat(1, len(good_obs_peaks)), axis=1)))

        # calculate transformed co-ordinates (ie the observed peaks transformed back to match the reference peaks)
        xyz_prime = np.dot(good_obs_peaks_xyz, M)
        delta_x = good_ref_peaks_xyz[:, 0] - xyz_prime[:, 0]
        delta_y = good_ref_peaks_xyz[:, 1] - xyz_prime[:, 1]

        delta_x_list = []
        delta_y_list = []
        # loop over all subsections
        for tup in subsection_id:
            # find indices of peaks falling in each subsection
            ix = [i for i, x in enumerate(peak_subsection_id) if x == tup]
            if return_residuals:
                delta_x_list.append(delta_x[ix])
                delta_y_list.append(delta_y[ix])
            else:
                # return difference between ref and obs
                delta_x_list.append(good_ref_peaks_xyz[ix,0] - good_obs_peaks_xyz[ix,0])
                delta_y_list.append(good_ref_peaks_xyz[ix,1] - good_obs_peaks_xyz[ix,1])

        # append to all-files list
        all_delta_x_list.append(delta_x_list)
        all_delta_y_list.append(delta_y_list)

    return all_delta_x_list, all_delta_y_list



def vector_plot(dx, dy, plotval='med'):
    '''
    dx, dy are outputs from "check_transformation_scatter()"
    'plotval' : either 'med(ian)' or 'mean'
    '''
    
    n_sub = len(dx[0])
    
    if plotval.lower() in ['med', 'median']:
        print('OK, plotting the median of the residuals per subsection...')
        x = [np.median(dx[0][i]) for i in range(n_sub)]
        y = [np.median(dy[0][i]) for i in range(n_sub)]
    elif plotval.lower() == 'mean':
        print('OK, plotting the mean of the residuals per subsection...')
        x = [np.mean(dx[0][i]) for i in range(n_sub)]
        y = [np.mean(dy[0][i]) for i in range(n_sub)]
    else:
        print('FAIL!!!')
        return
    
    # give each subsection a label
    subsection_id = []
    for j in range(np.sqrt(n_sub).astype(int)):
        for i in range(np.sqrt(n_sub).astype(int)):
            subsection_id.append((i, j))  # (x-axis, y-axis)
    
    plt.figure()
    X = np.arange(np.sqrt(n_sub).astype(int)) + 0.5
    Y = np.arange(np.sqrt(n_sub).astype(int)) + 0.5
    U = np.reshape(x, (np.sqrt(n_sub).astype(int), np.sqrt(n_sub).astype(int)))
    V = np.reshape(y, (np.sqrt(n_sub).astype(int), np.sqrt(n_sub).astype(int)))
    plt.quiver(X, Y, U, V)
    
    return
    
    
    
    
    
