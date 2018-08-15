


thflux = pyfits.getdata('/Users/christoph/data/commissioning/20180814/14aug30044_extracted.fits',0)
thflux2 = pyfits.getdata('/Users/christoph/data/commissioning/20180814/14aug30064_extracted.fits',0)

linenum, ordnum, m, pix, wl, model_wl, res = readcol('/Users/christoph/OneDrive - UNSW/linelists/AAT_folder/5th_order_'+
                                                   'clipped_lines_used_in_fit_as_of_2018-08-16.dat',
                                                   twod=False,skipline=2)

new_ordnum = 29



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
lam = []
x = []
search_region_size = 1
for new_ordnum in np.arange(2,40):

    print('order ', new_ordnum)

    #fix fails
    if new_ordnum == 25:
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

    # fix more fails (eventually we want to mask out regions around super bright / saturated lines so we don't need to fix the fails)
    if new_ordnum == 16:
        data = data[:2400]
    if new_ordnum == 3:
        data = data[:2000]

    fitted_line_pos2 = fit_emission_lines(data2, return_all_pars=False, varbeta=False, timit=False, verbose=False,
                                          thresh=thresh, bgthresh=bgthresh)

    for i,checkval in enumerate(fitted_line_pos2):
        print(i)
        q = np.logical_and(ord_x > checkval - search_region_size, ord_x < checkval + search_region_size)
        found = np.sum(q)
        if found > 0:
            if found == 1:
                lam.append(ord_wl[q])
                x.append(ord_pix[q])
            else:
                print('ERROR')





def quick_bg_fix(raw_data):
    left_xx = np.arange(2056)
    right_xx = np.arange(2056, 4112)
    left_bg = ndimage.minimum_filter(ndimage.gaussian_filter(raw_data[:2056],3), size=100)
    right_bg = ndimage.minimum_filter(ndimage.gaussian_filter(raw_data[2056:],3), size=100)
    data = raw_data.copy()
    data[left_xx] = raw_data[left_xx] - left_bg
    data[right_xx] = raw_data[right_xx] - right_bg
    return data

