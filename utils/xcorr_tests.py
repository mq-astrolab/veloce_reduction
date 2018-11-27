########################################################################################################################
# HOUSEKEEPING
path = '/Users/christoph/data/reduced/tauceti/tauceti_with_LFC/'
files = glob.glob(path + 'HD10700*')
# sort list of files
all_shortnames = []
for i,filename in enumerate(files):
    dum = filename.split('/')
    dum2 = dum[-1].split('.')
    dum3 = dum2[0]
    dum4 = dum3.split('_')
    shortname = dum4[1]
    all_shortnames.append(shortname)
files = files[np.argsort(all_shortnames)]

########################################################################################################################
########################################################################################################################
########################################################################################################################

# calculating BARYCORR

all_jd = []
all_bc = []
outfn = '/Users/christoph/data/reduced/tauceti/tauceti_with_LFC/tauceti_all_info.dat'
outfile = open(outfn, 'w')
outfn_jd = '/Users/christoph/data/reduced/tauceti/tauceti_with_LFC/tauceti_all_jds.dat'
outfile_jd = open(outfn_jd, 'w')
outfn_bc = '/Users/christoph/data/reduced/tauceti/tauceti_with_LFC/tauceti_all_bcs.dat'
outfile_bc = open(outfn_bc, 'w')
outfn_names = '/Users/christoph/data/reduced/tauceti/tauceti_with_LFC/tauceti_all_obsnames.dat'
outfile_names = open(outfn_names, 'w')

for i,filename in enumerate(files):
    print('Processing tau Ceti observation '+str(i+1)+'/'+str(len(files)))
    # do some housekeeping with filenames
    dum = filename.split('/')
    dum2 = dum[-1].split('.')
    dum3 = dum2[0]
    dum4 = dum3.split('_')
    shortname = dum4[1]
    bc = get_barycentric_correction(filename)
    pyfits.setval(filename, 'BARYCORR', value=bc[0], comment='barycentric velocity correction [m/s]')
    all_bc.append(bc[0])
    jd = pyfits.getval(filename, 'UTMJD') + 2.4e6 + 0.5
    all_jd.append(jd)
    outfile_names.write(shortname + ' \n')
    outfile_jd.write('%14.6f \n' % (jd))
    outfile_bc.write('%14.6f \n' % (bc))
    outfile.write(shortname + '     %14.6f     %14.6f \n' % (jd, bc))

outfile.close()
outfile_jd.close()
outfile_bc.close()
outfile_names.close()

########################################################################################################################
########################################################################################################################
########################################################################################################################

# calculate wl-solution for all fibres, including the LFC shifts and slopes; also append to reduced spectrum FITS file
signflip_slope = True
for i,filename in enumerate(files):
    print('Processing tau Ceti observation ' + str(i+1) + '/' + str(len(files)))
    dum = filename.split('/')
    dum2 = dum[-1].split('.')
    dum3 = dum2[0].split('_')
    obsname = dum3[1]
    wldict,wl = get_dispsol_for_all_fibs(obsname, signflip_slope=signflip_slope)
    pyfits.append(filename, wl, clobber=True)

########################################################################################################################
########################################################################################################################
########################################################################################################################

# calculating the CCFs for one order / 11 orders
# (either with or without the LFC shifts applied, comment out the 'wl' and 'wl0' you don't want)

all_xc = []
all_rv = []
for file in files:
    f = pyfits.getdata(file, 0)
    err = pyfits.getdata(file, 1)
    wl = pyfits.getdata(file, 2)
    # wl = pyfits.getdata('/Users/christoph/OneDrive - UNSW/dispsol/individual_fibres_dispsol_poly7_21sep30019.fits')
    f0 = pyfits.getdata(files[2], 0)
    err0 = pyfits.getdata(files[2], 1)
    wl0 = pyfits.getdata(files[2], 2)
    # wl0 = pyfits.getdata('/Users/christoph/OneDrive - UNSW/dispsol/individual_fibres_dispsol_poly7_21sep30019.fits')
    all_xc.append(get_RV_from_xcorr_combined_fibres(f, err, wl, f0, err0, wl0, return_xcs=True, individual_fibres=False,
                                                    individual_orders=False))
    rv,rverr = get_RV_from_xcorr_combined_fibres(f, err, wl, f0, err0, wl0, return_xcs=False, individual_fibres=False,
                                                    individual_orders=False)
    all_rv.append(rv)

########################################################################################################################
########################################################################################################################
########################################################################################################################

# plotting the CCFs on RV axis with and without BC applied

# WHEN THERE IS ONLY ONE CCF PER OBSERVATION

plt.figure()
plt.title('OLD ; CCF flipped = False')
plt.xlim(-2e4,2e4)
plt.ylim(0.9980,1.0005)
for i in range(len(files)):
    plt.plot(c * (np.arange(len(old_all_xc[i])) - (len(old_all_xc[i]) // 2)) * delta_log_wl,
             old_all_xc[i] / np.max(old_all_xc[i]), 'k.-')
    plt.plot(c * (np.arange(len(old_all_xc[i])) - (len(old_all_xc[i]) // 2)) * delta_log_wl + all_bc[i],
             old_all_xc[i] / np.max(old_all_xc[i]), 'r.-')
    plt.plot(c * (np.arange(len(old_all_xc[i])) - (len(old_all_xc[i]) // 2)) * delta_log_wl - all_bc[i],
             old_all_xc[i] / np.max(old_all_xc[i]), 'b.-')


plt.figure()
plt.title('NEW ; CCF flipped = False')
plt.xlim(-2e4,2e4)
plt.ylim(0.9980,1.0005)
for i in range(len(files)):
    plt.plot(c * (np.arange(len(all_xc[i])) - (len(all_xc[i]) // 2)) * delta_log_wl,
             all_xc[i] / np.max(all_xc[i]), 'k.-')
    plt.plot(c * (np.arange(len(all_xc[i])) - (len(all_xc[i]) // 2)) * delta_log_wl + all_bc[i],
             all_xc[i] / np.max(all_xc[i]), 'r.-')
    plt.plot(c * (np.arange(len(all_xc[i])) - (len(all_xc[i]) // 2)) * delta_log_wl - all_bc[i],
             all_xc[i] / np.max(all_xc[i]), 'b.-')


plt.figure()
plt.title('NEW ; CCF flipped = True')
plt.xlim(-2e4,2e4)
plt.ylim(0.9980,1.0005)
for i in range(len(files)):
    plt.plot(c * (np.arange(len(all_xc[i])) - (len(all_xc[i]) // 2)) * delta_log_wl,
             all_xc_flipped[i] / np.max(all_xc_flipped[i]), 'k.-')
    plt.plot(c * (np.arange(len(all_xc[i])) - (len(all_xc[i]) // 2)) * delta_log_wl + all_bc[i],
             all_xc_flipped[i] / np.max(all_xc_flipped[i]), 'r.-')
    plt.plot(c * (np.arange(len(all_xc[i])) - (len(all_xc[i]) // 2)) * delta_log_wl - all_bc[i],
             all_xc_flipped[i] / np.max(all_xc_flipped[i]), 'b.-')


# WHEN THERE ARE MULTIPLE CCFs (for different orders) PER OBSERVATION
addrange = 40
# index = [4,5,6,25,26,33,34,35]
index = [5,6,17,25,26,27,31,34,35,36,37]

# # this was for all obs at once
# xcarr = np.zeros((len(all_xc), len(all_xc[0]), 2*addrange+1))
# for i in range(xcarr.shape[0]):
#     for j in range(xcarr.shape[1]):
#         dum = np.array(all_xc[i][j])
#         xcarr[i,j,:] = dum[len(dum)//2 - addrange : len(dum)//2 + addrange +1]
# xcsums = np.sum(xcarr,axis=1)


plt.figure()
plt.title('tau Ceti - CCFs for 11 orders added up')
plt.xlim(-2e4,2e4)
plt.ylim(0.9980,1.0005)
for i in range(len(files)):
    plt.plot(c * (np.arange(len(xcsums[i,:])) - (len(xcsums[i,:]) // 2)) * delta_log_wl,
             xcsums[i,:] / np.max(xcsums[i,:]), 'k.-')
    plt.plot(c * (np.arange(len(xcsums[i,:])) - (len(xcsums[i,:]) // 2)) * delta_log_wl + all_bc[i],
             xcsums[i,:] / np.max(xcsums[i,:]), 'r.-')
    plt.plot(c * (np.arange(len(xcsums[i,:])) - (len(xcsums[i,:]) // 2)) * delta_log_wl - all_bc[i],
             xcsums[i,:] / np.max(xcsums[i,:]), 'b.-')

########################################################################################################################

