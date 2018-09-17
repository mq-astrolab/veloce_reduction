


thflux = pyfits.getdata('/Users/christoph/data/commissioning/20180814/14aug30044_extracted.fits',0)
#thflux2 = pyfits.getdata('/Users/christoph/data/commissioning/20180814/14aug30064_extracted.fits',0)
thflux2 = pyfits.getdata('/Users/christoph/data/commissioning/20180916/working/16sep30009_extracted.fits',0)


linenum, ordnum, m, pix, wl, model_wl, res = readcol('/Users/christoph/OneDrive - UNSW/linelists/AAT_folder/5th_order_'+
                                                   'clipped_lines_used_in_fit_as_of_2018-08-16.dat',
                                                   twod=False,skipline=2)
#H-alpha:
#new_ordnum = 29



#METHOD 1 - find new line positions by cross-correlation
shift = []
wshift = []

for new_ordnum in np.arange(2,40):

    print('order ',new_ordnum)

    raw_data = thflux[new_ordnum - 1, :]
    raw_data2 = thflux2[new_ordnum - 1, :]

    data = quick_bg_fix(raw_data)
    data2 = quick_bg_fix(raw_data2)

    scale = 300
    wdata = np.arcsinh(data / scale)
    wdata2 = np.arcsinh(data2 / scale)

    xc = np.correlate(data,data2, mode='same')
    wxc = np.correlate(wdata, wdata2, mode='same')

    xrange = np.arange(np.argmax(xc)-6, np.argmax(xc)+6+1, 1)

    #fit Gaussisn to central part of CCF
    mod = GaussianModel()
    parms = mod.guess(xc[xrange],x=xrange)
    result = mod.fit(xc[xrange],parms,x=xrange,weights=None)
    wparms = mod.guess(wxc[xrange], x=xrange)
    wresult = mod.fit(wxc[xrange], parms, x=xrange, weights=None)

    # #plot it?
    # plot_osf = 10
    # plot_os_grid = np.linspace(xrange[0],xrange[-1],plot_osf * (len(xrange)-1) + 1)
    # guessmodel = mod.eval(result.init_params,x=plot_os_grid)
    # bestmodel = mod.eval(result.params,x=plot_os_grid)
    # plt.legend()

    mu = result.best_values['center']
    wmu = wresult.best_values['center']

    shift.append(mu - (len(xc)//2))
    wshift.append(wmu - (len(wxc) // 2))





#METHOD 2 - find new line positions by checking a small region around each line in the mastertable
lam = np.array([])
x = np.array([])
order = np.array([])
ref_wl = np.array([])
search_region_size = 1
for new_ordnum in np.arange(2,40):

    print('order ', new_ordnum)

    #fix fails
    if new_ordnum == 17:
        thresh = 2000
        bgthresh = 1500
    elif new_ordnum == 25:
        thresh = 4000
        bgthresh = 2000
    else:
        thresh = 1500
        bgthresh = 1000

    mord = 64 + new_ordnum
    ix = np.argwhere(m==mord).flatten()
    ord_pix = pix[ix]
    ord_wl = wl[ix]

    raw_data = thflux[new_ordnum - 1, :]
    raw_data2 = thflux2[new_ordnum - 1, :]

    data = quick_bg_fix(raw_data)
    data2 = quick_bg_fix(raw_data2)

    # fix more fails, ie mask out regions around super bright / saturated lines
    # TODO: make this a predefined set of masks, so that the if statement does not have to be evaluated all the time
    if new_ordnum == 2:
        mask = np.zeros(len(data), dtype='bool')
        mask[3040:3110] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 3:
        mask = np.zeros(len(data),dtype='bool')
        mask[2300:2600] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 8:
        mask = np.zeros(len(data),dtype='bool')
        mask[1670:1860] = True
        mask[3560:3660] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 9:
        mask = np.zeros(len(data),dtype='bool')
        mask[1180:1420] = True
        mask[1560:1850] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 10:
        mask = np.zeros(len(data),dtype='bool')
        mask[2270:2430] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 11:
        mask = np.zeros(len(data),dtype='bool')
        mask[3000:3400] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 12:
        mask = np.zeros(len(data),dtype='bool')
        mask[540:1100] = True
        mask[2930:3220] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 13:
        mask = np.zeros(len(data),dtype='bool')
        mask[470:830] = True
        mask[2130:2330] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 15:
        mask = np.zeros(len(data), dtype='bool')
        mask[2630:2810] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 16:
        mask = np.zeros(len(data), dtype='bool')
        mask[2400:2800] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 17:
        mask = np.zeros(len(data),dtype='bool')
        mask[3060:3160] = True
        mask[3260:3370] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 18:
        mask = np.zeros(len(data),dtype='bool')
        mask[800:960] = True
        mask[1090:1320] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 19:
        mask = np.zeros(len(data), dtype='bool')
        mask[1850:2080] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 23:
        mask = np.zeros(len(data), dtype='bool')
        mask[1325:1400] = True
        data[mask] = 0
        data2[mask] = 0
    if new_ordnum == 24:
        mask = np.zeros(len(data), dtype='bool')
        mask[1900:2010] = True
        data[mask] = 0
        data2[mask] = 0


    fitted_line_pos2 = fit_emission_lines(data2, return_all_pars=False, varbeta=False, timit=False, verbose=False,
                                          thresh=thresh, bgthresh=bgthresh)

    for i,checkval in enumerate(fitted_line_pos2):
        print(i)
        q = np.logical_and(ord_pix > checkval - search_region_size, ord_pix < checkval + search_region_size)
        found = np.sum(q)
        if found > 0:
            if found == 1:
                print('OK')
                lam = np.append(lam, ord_wl[q])
                x = np.append(x, ord_pix[q])
                order = np.append(order, new_ordnum)
                ref_wl = np.append(ref_wl, ord_wl[np.argwhere(q==True)])
            else:
                print('ERROR')





def quick_bg_fix(raw_data, npix=4112):
    left_xx = np.arange(npix/2)
    right_xx = np.arange(npix/2, npix)
    left_bg = ndimage.minimum_filter(ndimage.gaussian_filter(raw_data[:npix/2],3), size=100)
    right_bg = ndimage.minimum_filter(ndimage.gaussian_filter(raw_data[npix/2:],3), size=100)
    data = raw_data.copy()
    data[left_xx] = raw_data[left_xx] - left_bg
    data[right_xx] = raw_data[right_xx] - right_bg
    return data




#re-normalize arrays to [-1,+1]
x_norm = (x / ((len(data)-1)/2.)) - 1.
order_norm = ((order-2) / (37./2.)) - 1.       #TEMP, TODO, FUGANDA, PLEASE FIX ME!!!!!
#order_norm = ((m-1) / ((len(P_id)-1)/2.)) - 1.

#call the fitting routine
p = fit_poly_surface_2D(x_norm, order_norm, ref_wl, weights=None, polytype='chebyshev', poly_deg=5, debug_level=0)

#if return_full:
xx = np.arange(4112)
xxn = (xx / ((len(xx) - 1) / 2.)) - 1.
oo = np.arange(1, len(thflux))
oon = ((oo - 1) / (38. / 2.)) - 1.  # TEMP, TODO, FUGANDA, PLEASE FIX ME!!!!!
# oon = ((oo-1) / ((len(thflux)-1)/2.)) - 1.
X, O = np.meshgrid(xxn, oon)
p_wl = p(X, O)

