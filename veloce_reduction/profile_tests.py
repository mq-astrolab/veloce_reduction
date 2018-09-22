def get_fibre_profiles_single_order(sc, sr, err_sc, ordpol, ordmask=None, nfib=24, sampling_size=25, step_size=None,
                                    varbeta=True, return_snr=True, debug_level=0, timit=False):
    """
    INPUT:
    'sc'             : the flux in the extracted, flattened stripe
    'sr'             : row-indices (ie in spatial direction) of the cutouts in 'sc'
    'err_sc'         : the error in the extracted, flattened stripe
    'ordpol'         : set of polynomial coefficients from P_id for that order (ie p = P_id[ord])
    'fppo'           : Fibre Profile Parameters by Order (dictionary containing the fitted fibre profile parameters)
    'ordmask'        : gives user the option to provide a mask (eg from "find_stripes")
    'nfib'           : number of fibres for which to retrieve the relative intensities (default is 24, b/c there are 19 object fibres plus 5 sky fibres for VeloceRosso)
    'slit_height'    : height of the 'extraction slit' is 2*slit_height
    'sampling_size'  : how many pixels (in dispersion direction) either side of current i-th pixel do you want to consider?
                       (ie stack profiles for a total of 2*sampling_size+1 pixels in dispersion direction...)
    'step_size'      : only calculate the relative intensities every so and so many pixels (should not change, plus it takes ages...)
    'return_full'    : boolean - do you want to also return the full model (if FALSE, then only the relative intensities are returned)
    'return_snr'     : boolean - do you want to return the SNR of the collapsed super-pixel at each location in 'userange'?
    'debug_level'    : for debugging...

    OUTPUT:
    'relints'        : relative intensities in the fibres
    'relints_norm'   : relative intensities in the fibres re-normalized to a sum of 1
    'full_model'     : a list containing the full model of every cutout (only if 'return_full' is set to TRUE)
    'modgrid'        : a list containing the grid on which the full model is evaluated (only if 'return_full' is set to TRUE)


    MODHIST:
    14/06/2018 - CMB create (essentially a clone of "determine_spatial_profiles_single_order", but removed all the parts concerning the testing of different models)
    25/06/2018 - CMB added call to "linalg_extract_column" rather than naively fitting some model
    28/06/2018 - CMB added 'return_snr' keyword
    03/08/2018 - CMB added proper error treatment
    """

    #TODO: add offset and/or slope to fit

    if timit:
        start_time = time.time()
    if debug_level >= 1:
        print('Fitting fibre profiles for one order...')

    npix = sc.shape[1]

    if ordmask is None:
        ordmask = np.ones(npix, dtype='bool')

    if step_size is None:
        step_size = 2 * sampling_size
    userange = np.arange(np.arange(npix)[ordmask][0] + sampling_size, np.arange(npix)[ordmask][-1], step_size)
    #don't use pixel columns 200 pixels from either end of the chip
    userange = userange[np.logical_and(userange > 200, userange < npix - 200)]

    # prepare output dictionary
    fibre_profiles_ord = {}
    fibre_profiles_ord['mu'] = np.zeros((len(userange),nfib))
    fibre_profiles_ord['sigma'] = np.zeros((len(userange), nfib))
    fibre_profiles_ord['amp'] = np.zeros((len(userange), nfib))
    if varbeta:
        fibre_profiles_ord['beta'] = np.zeros((len(userange), nfib))
    if return_snr:
            fibre_profiles_ord['SNR'] = []

    # loop over all columns for one order and do the profile fitting
    for i, pix in enumerate(userange):

        if debug_level >= 1:
            print('pix = ', str(pix))


        # calculate SNR of collapsed super-pixel at this location (including RON)
        if return_snr:
            #snr.append(np.sqrt(np.sum(sc[:,pix])))
            #snr.append(np.sum(sc[:, pix]) / np.sqrt(np.sum(err_sc[:, pix] ** 2)))
            fibre_profiles_ord['SNR'].append(np.sum(sc[:, pix]) / np.sqrt(np.sum(err_sc[:, pix] ** 2)))

        # check if that particular cutout falls fully onto CCD
        checkprod = np.product(sr[1:, pix].astype(float))  # exclude the first row number, as that can legitimately be zero
        # NOTE: This also covers row numbers > ny, as in these cases 'sr' is set to zero in "flatten_single_stripe(_from_indices)"
        if ordmask[pix] == False:
            fibre_profiles_ord['mu'][i, :] = np.repeat(-1., nfib)
            fibre_profiles_ord['sigma'][i, :] = np.repeat(-1., nfib)
            fibre_profiles_ord['amp'][i, :] = np.repeat(-1., nfib)
            if varbeta:
                fibre_profiles_ord['beta'][i, :] = np.repeat(-1., nfib)
        elif checkprod == 0:
            checksum = np.sum(sr[:, pix])
            if checksum == 0:
                print('WARNING: the entire cutout lies outside the chip!!!')
                # best_values = {'amp':-1., 'beta':-1., 'mu':-1., 'sigma':-1.}
            else:
                print('WARNING: parts of the cutout lie outside the chip!!!')
                # best_values = {'amp':-1., 'beta':-1., 'mu':-1., 'sigma':-1.}
            fibre_profiles_ord['mu'][i, :] = np.repeat(-1., nfib)
            fibre_profiles_ord['sigma'][i, :] = np.repeat(-1., nfib)
            fibre_profiles_ord['amp'][i, :] = np.repeat(-1., nfib)
            if varbeta:
                fibre_profiles_ord['beta'][i, :] = np.repeat(-1., nfib)
        else:
            # this is the NORMAL case, where the entire cutout lies on the chip
            grid = np.array([])
            # data = np.array([])
            normdata = np.array([])
            # errors = np.array([])
            weights = np.array([])
            refpos = ordpol(pix)
            for j in np.arange(np.max([0, pix - sampling_size]), np.min([npix - 1, pix + sampling_size]) + 1):
                grid = np.append(grid, sr[:, j] - ordpol(j) + refpos)
                # data = np.append(data,sc[:,j])
                normdata = np.append(normdata, sc[:, j] / np.sum(sc[:, j]))
                # assign weights for flux (and take care of NaNs and INFs)
                # normerr = np.sqrt(sc[:,j] + RON**2) / np.sum(sc[:,j])
                normerr = err_sc[:, j] / np.sum(sc[:, j])
                pix_w = 1. / (normerr * normerr)
                pix_w[np.isinf(pix_w)] = 0.
                weights = np.append(weights, pix_w)
                ### initially I thought this was clearly rubbish as it down-weights the central parts
                ### and that we really want to use the relative errors, ie w_i = 1/(relerr_i)**2
                ### HOWEVER: this is not true, and the optimal extraction linalg routine requires absolute errors!!!
                # weights = np.append(weights, 1./((np.sqrt(sc[:,j] + RON**2)) / sc[:,j])**2)
                if debug_level >= 2:
                    # plt.plot(sr[:,j] - ordpol(j),sc[:,j],'.')
                    plt.plot(sr[:, j] - ordpol(j), sc[:, j] / np.sum(sc[:, j]), '.')
                    # plt.xlim(-5,5)
                    plt.xlim(-sc.shape[0] / 2, sc.shape[0] / 2)

            # data = data[grid.argsort()]
            normdata = normdata[grid.argsort()]
            weights = weights[grid.argsort()]
            grid = grid[grid.argsort()]

            # if debug_level >= 2:
            #     plt.plot(grid,normdata)

            dynrange = np.max(normdata) - np.min(normdata)
            goodpeaks, mostpeaks, allpeaks = find_suitable_peaks(normdata, thresh=np.min(normdata)+0.5*dynrange, bgthresh=np.min(normdata)+0.25*dynrange,
                                                                 clip_edges=False, gauss_filter_sigma=10, slope=1e-8)

            if debug_level >=1 :
                print('Number of fibres found: ',len(goodpeaks))

            if len(goodpeaks) != nfib :
                print('FUUUUU!!! pix = ',pix)

                fibre_profiles_ord['mu'][i, :] = np.repeat(-1., nfib)
                fibre_profiles_ord['sigma'][i, :] = np.repeat(-1., nfib)
                fibre_profiles_ord['amp'][i, :] = np.repeat(-1., nfib)
                if varbeta:
                    fibre_profiles_ord['beta'][i, :] = np.repeat(-1., nfib)

            else:

                #do the fitting
                peaks = np.r_[grid[goodpeaks]]

                npeaks = len(peaks)

                guess = []
                lower_bounds = []
                upper_bounds = []
                for n in range(npeaks):
                    if varbeta:
                        guess.append(np.array([peaks[n], 1., normdata[goodpeaks[n]], 2.]))
                        lower_bounds.append([peaks[n] - 1, 0, 0, 1])
                        upper_bounds.append([peaks[n] + 1, np.inf, np.inf, 4])
                    else:
                        guess.append(np.array([peaks[n], 1., normdata[goodpeaks[n]]]))
                        lower_bounds.append([peaks[n] - 1, 0, 0])
                        upper_bounds.append([peaks[n] + 1, np.inf, np.inf])
                guess = np.array(guess).flatten()
                lower_bounds = np.array(lower_bounds).flatten()
                upper_bounds = np.array(upper_bounds).flatten()
                if varbeta:
                    popt, pcov = op.curve_fit(multi_fibmodel_with_amp, grid, normdata, p0=guess,
                                              bounds=(lower_bounds, upper_bounds))
                else:
                    popt, pcov = op.curve_fit(CMB_multi_gaussian, grid, normdata, p0=guess,
                                              bounds=(lower_bounds, upper_bounds))



                if varbeta:
                    popt_arr = np.reshape(popt, (24, 4))
                else:
                    popt_arr = np.reshape(popt, (24, 3))

                fibre_profiles_ord['mu'][i, :] = popt_arr[:,0]
                fibre_profiles_ord['sigma'][i, :] = popt_arr[:, 1]
                fibre_profiles_ord['amp'][i, :] = popt_arr[:, 2]
                if varbeta:
                    fibre_profiles_ord['beta'][i, :] = popt_arr[:, 3]


                # full_model = np.zeros(grid.shape)
                # if varbeta:
                #     full_model += multi_fibmodel_with_amp(grid,*popt)
                # else:
                #     full_model += CMB_pure_gaussian(grid, *popt)




        # fill output array
        positions[i,:] = line_pos_fitted
        relints[i, :] = line_amp_fitted
        fnorm = line_amp_fitted / np.sum(line_amp_fitted)
        relints_norm[i, :] = fnorm

    if timit:
        print('Elapsed time for retrieving relative intensities: ' + np.round(time.time() - start_time, 2).astype(str) + ' seconds...')

    if return_snr:
        return relints, relints_norm, positions, snr
    else:
        return relints, relints_norm, positions