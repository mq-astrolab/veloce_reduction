import glob
import time
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from readcol import readcol

from veloce_reduction.veloce_reduction.wavelength_solution import get_dispsol_for_all_fibs, get_dispsol_for_all_fibs_2
from veloce_reduction.veloce_reduction.get_radial_velocity import get_RV_from_xcorr, get_RV_from_xcorr_2, make_ccfs, \
    old_make_ccfs
from veloce_reduction.veloce_reduction.helper_functions import get_snr, short_filenames
from veloce_reduction.veloce_reduction.flat_fielding import onedim_pixtopix_variations, deblaze_orders
from veloce_reduction.veloce_reduction.barycentric_correction import get_barycentric_correction
from veloce_reduction.veloce_reduction.cosmic_ray_removal import onedim_medfilt_cosmic_ray_removal

########################################################################################################################
# HOUSEKEEPING
path = '/Users/christoph/data/reduced/tauceti/tauceti_with_LFC/'
# path = '/Volumes/BERGRAID/data/veloce/reduced/tauceti/tauceti_with_LFC/'

file_list = glob.glob(path + '*optimal*.fits')
wl_list = glob.glob(path + '*wl*')
assert len(file_list) == len(wl_list), 'ERROR: number of wl-solution files does not match the number of reduced spectra!!!'

obsname_list = [fn.split('_')[-3] for fn in file_list]
object_list = [pyfits.getval(fn, 'OBJECT').split('+')[0] for fn in file_list]
bc_list = [pyfits.getval(fn, 'BARYCORR') for fn in file_list]
texp_list = [pyfits.getval(fn, 'ELAPSED') for fn in file_list]
utmjd_start = np.array([pyfits.getval(fn, 'UTMJD') for fn in file_list]) + 2.4e6 + 0.5   # the fits header has 2,400,000.5 subtracted!!!!!
utmjd = utmjd_start + (np.array(texp_list)/2.)/86400.
sortix = np.argsort(utmjd)
all_obsnames = np.array(obsname_list)[sortix]
files = np.array(file_list)[sortix]
wls = np.array(file_list)[sortix]
all_bc = np.array(bc_list)[sortix]
all_jd = utmjd[sortix]



########################################################################################################################
########################################################################################################################
########################################################################################################################

all_snr = readcol(path + 'tauceti_all_snr.dat', twod=False)[0]

# # get mean SNR per collapsed pixel
# all_snr = []
# for i, file in enumerate(files):
#     print('Estimating mean SNR for tau Ceti observation ' + str(i + 1) + '/' + str(len(files)))
#     flux = pyfits.getdata(file, 0)
#     err = pyfits.getdata(file, 1)
#     all_snr.append(get_snr(flux, err))
# np.savetxt(path + 'tauceti_all_snr.dat', np.array(all_snr))

########################################################################################################################
########################################################################################################################
########################################################################################################################

# calculating the CCFs for one order / 11 orders
# (either with or without the LFC shifts applied, comment out the 'wl' and 'wl0' you don't want)

vers = 'v1c'

all_xc = []
all_rv = np.zeros((len(files), 11, 19))
all_sumrv = []
# all_sumrv = np.zeros(len(files))
# all_sumrv_2 = np.zeros(len(files))
xcsums = np.zeros((len(files), 301))

# TEMPLATE:
ix0 = 7
f0 = pyfits.getdata(files[ix0], 0)
err0 = pyfits.getdata(files[ix0], 1)
obsname_0 = '19sep30377'  # the highest SNR obs
date_0 = '20180919'
wl0 = pyfits.getdata(wl_list[ix0])

maskdict = np.load('/Volumes/BERGRAID/data/veloce/reduced/' + date_0 + '/mask.npy').item()

# wl0 = pyfits.getdata('/Users/christoph/OneDrive - UNSW/dispsol/individual_fibres_dispsol_poly7_21sep30019.fits')

mw_flux = pyfits.getdata('/Volumes/BERGRAID/data/veloce/reduced/' + date_0 + '/master_white_optimal3a_extracted.fits')
smoothed_flat, pix_sens = onedim_pixtopix_variations(mw_flux, filt='gaussian', filter_width=25)

f0_clean = f0.copy()

# loop over orders
for o in range(f0.shape[0]):
    # loop over fibres (but exclude the simThXe and LFC fibres for obvious reasons!!!)
    for fib in range(1,f0.shape[1]-1):
        print('Order ', o + 1)
        if (o == 0) and (fib == 0):
            start_time = time.time()
        f0_clean[o, fib, :], ncos = onedim_medfilt_cosmic_ray_removal(f0[o, fib, :], err0[o, fib, :], w=31, thresh=5., low_thresh=3.)
        if (o == 38) and (fib == 18):
            print('time elapsed ', time.time() - start_time, ' seconds')

f0_dblz, err0_dblz = deblaze_orders(f0_clean[:,3:22,:], smoothed_flat[:,3:22,:], maskdict, err=err0[:,3:22,:], combine_fibres=True,
                                    degpol=2, gauss_filter_sigma=3., maxfilter_size=100)
f0_dblz, err0_dblz = deblaze_orders(f0_clean, smoothed_flat, maskdict, err=err0, combine_fibres=True,
                                    degpol=2, gauss_filter_sigma=3., maxfilter_size=100)


# use a synthetic template?
# wl0, f0 = readcol('/Users/christoph/OneDrive - UNSW/synthetic_templates/' + 'synth_teff5250_logg45.txt', twod=False)
# wl0 = pyfits.getdata('/Users/christoph/OneDrive - UNSW/synthetic_templates/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
# f0 = pyfits.getdata('/Users/christoph/OneDrive - UNSW/synthetic_templates/phoenix/lte05400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')


for i, filename in enumerate(files[73:83]):
    # for i,filename in enumerate(files):
    i += 73
    print('Processing RV for tau Ceti observation ' + str(i + 1) + '/' + str(len(files)))
    # get obsname and date
    dum = filename.split('/')
    dum2 = dum[-1].split('.')
    dum3 = dum2[0].split('_')
    obsname = dum3[1]
    day = obsname[:2]
    mon = obsname[2:5]
    if mon == 'jan':
        year = '2019'
        mondig = '01'
    elif mon == 'sep':
        year = '2018'
        mondig = '09'
    elif mon == 'nov':
        year = '2018'
        mondig = '11'
    date = year + mondig + day
    # read in spectrum
    f = pyfits.getdata(filename, 0)
    err = pyfits.getdata(filename, 1)
    # f_clean = f.copy()
    # for o in range(f.shape[0]):
    #     for fib in range(f.shape[1]):
    #         f_clean[o,fib,:],ncos = onedim_medfilt_cosmic_ray_removal(f[o,fib,:], err[o,fib,:], w=31, thresh=5., low_thresh=3.)
    #     wl = pyfits.getdata(filename, 2)
    #     wl = pyfits.getdata('/Users/christoph/OneDrive - UNSW/dispsol/individual_fibres_dispsol_poly7_21sep30019.fits')
    #     wldict,wl = get_dispsol_for_all_fibs(obsname, date=date, fibs='stellar', refit=False, fibtofib=True, nightly_coeffs=True)
    wldict, wl = get_dispsol_for_all_fibs_2(obsname, refit=True, eps=2)
    #     f_dblz, err_dblz = deblaze_orders(f_clean, smoothed_flat[:,2:21,:], maskdict, err=err, combine_fibres=True, skip_first_order=True, degpol=2, gauss_filter_sigma=3., maxfilter_size=100)
    #     all_xc.append(old_make_ccfs(f, wl, f0, wl0, bc=all_bc[i], bc0=all_bc[6], mask=None, smoothed_flat=None, delta_log_wl=1e-6, relgrid=False,
    #                             flipped=False, individual_fibres=False, debug_level=1, timit=False))
    #     rv,rverr,xcsum = get_RV_from_xcorr_2(f, wl, f0, wl0, bc=all_bc[i], bc0=all_bc[6], individual_fibres=True, individual_orders=True, old_ccf=True, debug_level=1)
    sumrv, sumrverr, xcsum = get_RV_from_xcorr_2(f, wl, f0, wl0, bc=all_bc[i], bc0=all_bc[ix0],
                                                 smoothed_flat=smoothed_flat, fitrange=35, individual_fibres=False,
                                                 individual_orders=False,
                                                 fit_slope=True, old_ccf=False, debug_level=1)
    #     sumrv,sumrverr,xcsum = get_RV_from_xcorr_2(f_dblz, wl, f0_dblz, wl0, bc=all_bc[i], bc0=all_bc[6], individual_fibres=False, individual_orders=False, old_ccf=True, debug_level=1)
    #     all_rv[i,:,:] = rv
    all_sumrv.append(sumrv)
    #     all_sumrv[i] = sumrv
    xcsums[i, :] = xcsum
xcsums = np.array(xcsums)

np.save('/Users/christoph/OneDrive - UNSW/tauceti/rvtest/july_2019/rvs_' + vers + '.npy', all_sumrv)

# PLOT
plt.plot(all_sumrv, 'x')
plt.xlabel('# obs')
plt.ylabel('dRV [m/s]')
plt.title('Tau Ceti (N=151)   --   ' + vers)
plt.text(100, 100, 'RMS = ' + str(np.round(np.std(all_sumrv), 1)) + ' m/s', size='large')
plt.savefig('/Users/christoph/OneDrive - UNSW/tauceti/rvtest/july_2019/rvs_' + vers + '.eps')

########################################################################################################################
########################################################################################################################
########################################################################################################################

