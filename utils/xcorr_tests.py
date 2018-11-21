########################################################################################################################

# calculating BARYCORR

for i,filename in enumerate(sorted(files)):
    # do some housekeeping with filenames
    dum = filename.split('/')
    dum2 = dum[-1].split('.')
    dum3 = dum2[0]
    dum4 = dum3.split('_')
    shortname = dum4[1]
    bc = get_barycentric_correction(filename)
    pyfits.setval(filename, 'BARYCORR', value=bc, comment='barycentric velocity correction [m/s]')
    all_bc.append(bc)
    jd = pyfits.getval()
    all_jd.append(jd)
########################################################################################################################


########################################################################################################################

# plotting the CCFs on RV axis with and without BC applied

plt.figure()
plt.title('CCF flipped = False')
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
plt.title('CCF flipped = True')
plt.xlim(-2e4,2e4)
plt.ylim(0.9980,1.0005)
for i in range(len(files)):
    plt.plot(c * (np.arange(len(all_xc[i])) - (len(all_xc[i]) // 2)) * delta_log_wl,
             all_xc_flipped[i] / np.max(all_xc_flipped[i]), 'k.-')
    plt.plot(c * (np.arange(len(all_xc[i])) - (len(all_xc[i]) // 2)) * delta_log_wl + all_bc[i],
             all_xc_flipped[i] / np.max(all_xc_flipped[i]), 'r.-')
    plt.plot(c * (np.arange(len(all_xc[i])) - (len(all_xc[i]) // 2)) * delta_log_wl - all_bc[i],
             all_xc_flipped[i] / np.max(all_xc_flipped[i]), 'b.-')
########################################################################################################################

