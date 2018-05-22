'''
Created on 11 Aug. 2017

@author: christoph
'''

#how to measure program run-time quick-and-dirty style
start_time = time.time()
i=1
x=123456.789
while i < 10000000:
    np.float_power(x,2)
    i += 1
print(time.time() - start_time)

start_time = time.time()
#medimg = np.median(allimg,axis=0)
medimg2 = np.median(np.array(allimg),axis=0)
# while i < 10:
#     medimg = np.median(allimg,axis=0)
#     i += 1
print(time.time() - start_time)


#locations for individual peaks
peaklocs = np.arange(316,1378+59,59)
peaks = np.zeros(2048)
peaks[peaklocs]=1
xgrid = x_ix - sim.x_map[i,j] - nx//2

offsets_peaklocs = offsets[peaklocs]

g=[]
for ii in range(19):
    g.append(fibmodel(offsets,offsets_peaklocs[ii],2,norm=0))
g = np.array(g)

ginterp = []
for ii in range(19):
    ginterp.append(np.interp(xgrid, offsets, g[ii,:]))
myphi = np.array(ginterp).T

result = eta * myphi
result_sum = np.sum(result,axis=1)

#then just plug into linalg_extract_column




#how I created the temporary P_id dictionary for JS's code
P_id = {}
Ptemp = {}
ordernames = []
for i in range(1,10):
    ordernames.append('order_0%i' % i)
for i in range(10,34):
    ordernames.append('order_%i' % i)
#the array parms comes from the "find_stripes" function
for i in range(33):
    Ptemp.update({ordernames[i]:parms[i]})
P_id = {'fibre_01': Ptemp}



#timing tests
start_time = time.time()
for i in range(10):
    mask2 = stripe.copy()
    mask2[:,:] = 1
print(time.time() - start_time)






#from POLYSPECT
def spectral_format_with_matrix_rechts(input_arm):
        """Create a spectral format, including a detector to slit matrix at
           every point.

        Returns
        -------
        x: (nm, ny) float array
            The x-direction pixel co-ordinate corresponding to each y-pixel
            and each order (m).
        w: (nm, ny) float array
            The wavelength co-ordinate corresponding to each y-pixel and each
            order (m).
        blaze: (nm, ny) float array
            The blaze function (pixel flux divided by order center flux)
            corresponding to each y-pixel and each order (m).
        matrices: (nm, ny, 2, 2) float array
            2x2 slit rotation matrices, mapping output co-ordinates back
            to the slit.
        """
        x, w, b, ccd_centre = spectral_format(input_arm)
        matrices = np.zeros((x.shape[0], x.shape[1], 2, 2))
        amat = np.zeros((2, 2))

        for i in range(x.shape[0]):  # i is the order
            for j in range(x.shape[1]):
                # Create a matrix where we map input angles to output
                # coordinates.
                slit_microns_per_det_pix = input_arm.slit_microns_per_det_pix_first + \
                    float(i) / x.shape[0] * (input_arm.slit_microns_per_det_pix_last - \
                                             input_arm.slit_microns_per_det_pix_first)
                amat[0, 0] = 1.0 / slit_microns_per_det_pix
                amat[0, 1] = 0
                amat[1, 0] = 0
                amat[1, 1] = 1.0 / slit_microns_per_det_pix
                # Apply an additional rotation matrix. If the simulation was
                # complete, this wouldn't be required.
                r_rad = np.radians(input_arm.extra_rot)
                dy_frac = (j - x.shape[1] / 2.0) / (x.shape[1] / 2.0)
                extra_rot_mat = np.array([[np.cos(r_rad * dy_frac),
                                           np.sin(r_rad * dy_frac)],
                                          [-np.sin(r_rad * dy_frac),
                                           np.cos(r_rad * dy_frac)]])
                amat = np.dot(extra_rot_mat, amat)
                # We actually want the inverse of this (mapping output
                # coordinates back onto the slit.
                matrices[i, j, :, :] = np.linalg.inv(amat)
        return x, w, b, matrices




#from GHOSTSIM
def spectral_format_with_matrix_links(input_arm):
    """Create a spectral format, including a detector to slit matrix at every point.
    
    Returns
    -------
    x: (nm, ny) float array
        The x-direction pixel co-ordinate corresponding to each y-pixel and each
        order (m).    
    w: (nm, ny) float array
        The wavelength co-ordinate corresponding to each y-pixel and each
        order (m).
    blaze: (nm, ny) float array
        The blaze function (pixel flux divided by order center flux) corresponding
        to each y-pixel and each order (m).
    matrices: (nm, ny, 2, 2) float array
        2x2 slit rotation matrices.
    """        
    x,w,b,ccd_centre = spectral_format(input_arm)
    x_xp,w_xp,b_xp,dummy = spectral_format(input_arm,xoff=-1e-3,ccd_centre=ccd_centre)
    x_yp,w_yp,b_yp,dummy = spectral_format(input_arm,yoff=-1e-3,ccd_centre=ccd_centre)
    dy_dyoff = np.zeros(x.shape)
    dy_dxoff = np.zeros(x.shape)
    #For the y coordinate, spectral_format output the wavelength at fixed pixel, not 
    #the pixel at fixed wavelength. This means we need to interpolate to find the 
    #slit to detector transform.
    isbad = w*w_xp*w_yp == 0
    for i in range(x.shape[0]):
        ww = np.where(isbad[i,:] == False)[0]
        dy_dyoff[i,ww] =     np.interp(w_yp[i,ww],w[i,ww],np.arange(len(ww))) - np.arange(len(ww))
        dy_dxoff[i,ww] =     np.interp(w_xp[i,ww],w[i,ww],np.arange(len(ww))) - np.arange(len(ww))
        #Interpolation won't work beyond the end, so extrapolate manually (why isn't this a numpy
        #option???)
        dy_dyoff[i,ww[-1]] = dy_dyoff[i,ww[-2]]
        dy_dxoff[i,ww[-1]] = dy_dxoff[i,ww[-2]]
                
    #For dx, no interpolation is needed so the numerical derivative is trivial...
    dx_dxoff = x_xp - x
    dx_dyoff = x_yp - x

    #flag bad data...
    x[isbad] = np.nan
    w[isbad] = np.nan
    b[isbad] = np.nan
    dy_dyoff[isbad] = np.nan
    dy_dxoff[isbad] = np.nan
    dx_dyoff[isbad] = np.nan
    dx_dxoff[isbad] = np.nan
    matrices = np.zeros( (x.shape[0],x.shape[1],2,2) )
    amat = np.zeros((2,2))

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            ## Create a matrix where we map input angles to output coordinates.
            amat[0,0] = dx_dxoff[i,j]
            amat[0,1] = dx_dyoff[i,j]
            amat[1,0] = dy_dxoff[i,j]
            amat[1,1] = dy_dyoff[i,j]
            ## Apply an additional rotation matrix. If the simulation was complete,
            ## this wouldn't be required.
            r_rad = np.radians(input_arm.extra_rot)
            dy_frac = (j - x.shape[1]/2.0)/(x.shape[1]/2.0)
            extra_rot_mat = np.array([[np.cos(r_rad*dy_frac),np.sin(r_rad*dy_frac)],[-np.sin(r_rad*dy_frac),np.cos(r_rad*dy_frac)]])
            amat = np.dot(extra_rot_mat,amat)
            ## We actually want the inverse of this (mapping output coordinates back
            ## onto the slit.
            matrices[i,j,:,:] =  np.linalg.inv(amat)
    return x,w,b,matrices



#finding files (needs "import glob, os")
path = '/Users/christoph/UNSW/simulated_spectra/ES/'
files = glob.glob(path+"veloce_flat_t70000_single*.fit")
images={}
allfibflat = np.zeros((4096,4096))
for i,file in enumerate(files):
    fibname = file[-9:-4]
    images[fibname] = pyfits.getdata(file)
    allfibflat += images[fibname]
#read one dummy header
h = pyfits.getheader(files[0])
#write output file
pyfits.writeto(path+'veloce_flat_t70000_nfib19.fit', allfibflat, h)



#read table from file
import re
non_decimal = re.compile(r'[^\d.]+')
#f = open('/Users/christoph/UNSW/linelists/ThXe_linelist_raw2.txt', 'r')
f = open('/Users/christoph/UNSW/linelists/ThAr_linelist_raw2.txt', 'r')
species = []
wl = []
relint = []
for i,line in enumerate(f):
    print('Line ',str(i+1),': ',repr(line))
    line = line.strip()
    cols = line.split("|")
    if (cols[0].strip() != '') and (cols[1].strip() != '') :
        species.append(cols[0].strip())
        wl.append(float(cols[1].strip()))
        if non_decimal.sub('', cols[2].strip()) == '':
            relint.append(0)
        else:
            relint.append(float(non_decimal.sub('', cols[2].strip())))
f.close()
linelist = np.array([species,wl,relint])
#np.savetxt('/Users/christoph/UNSW/linelists/ThXe_linelist.dat', np.c_[wl, relint], delimiter=';', fmt='%10.8f; %i')
np.savetxt('/Users/christoph/UNSW/linelists/ThAr_linelist.dat', np.c_[wl, relint], delimiter=';', fmt='%10.8f; %i')


#make fake laser-comb line-list
delta_f = 2.5E10    #final frequency spacing is 25 Ghz
c = 2.99792458E8    #speed of light in m/s
f0 = c / 1E-6           #do for a wavelength range of 550nm-1000nm, ie 1000nm is ~300Thz
fmax = c / 5.5E-7
ct=0
frange = np.arange(9812) * delta_f + f0
wl = np.flip((c / frange) / 1e-6, axis=0)     #wavelength in microns (from shortest to longest wavelength)
relint = [1]*len(wl)
np.savetxt('/Users/christoph/UNSW/linelists/laser_linelist_25GHz.dat', np.c_[wl, relint], delimiter=';', fmt='%10.8f; %i')




#write arrays to output file
outfn = open('/Users/christoph/UNSW/rvtest/solar_0ms.txt', 'w')
for ord in sorted(pix.iterkeys()):
    for i in range(4096):
        outfn.write("%s     %f     %f     %f\n" %(pix[ord][i],wl[ord][i],flux[ord][i],err[ord][i]))

outfn.close()




#timing tests for matrix inversion
start_time = time.time()
for i in range(10000):
    C_inv = np.linalg.inv(C)
    eta = np.matmul(C_inv,b)
    
    #np.linalg.solve(C, b)
print(time.time() - start_time,' seconds')


#finding peaks when
for ord in sorted(laserdata['flux'].iterkeys())[:-1]:
    data = laserdata['flux'][ord].copy()
    #data[::2] += 1e-3
    filtered_data = ndimage.gaussian_filter(data.astype(np.float), 1)
    filtered_data2 = ndimage.gaussian_filter(data.astype(np.float), 2)
    allpeaks = signal.argrelextrema(filtered_data, np.greater)[0]
    allpeaks2 = signal.argrelextrema(filtered_data2, np.greater)[0]
    print(ord,' : ',len(allpeaks),' | ',len(allpeaks2))
    if len(allpeaks) != len(allpeaks2):
        print('fuganda')
        
        
        
        
# co-add spectra
path = '/Users/christoph/UNSW/veloce_spectra/Mar02/'
for n,file in enumerate(glob.glob(path+'*etalon*')):
    img = pyfits.getdata(file)
    if n==0:
        h = pyfits.getheader(file)
        master = img.copy().astype(float)
    else:
        master += img
master = master / 3.
pyfits.writeto(path+'master_dark.fits', master, h)
#and as integer:
master_int = np.round(master).astype(int)     
pyfits.writeto(path+'master_dark_int.fits', master_int, h)



#check if the prelim (guessed from previous order) wl of a fitted peak position corresponds to a known wavelength from ThAr atlas
linelist = np.array(readcol('/Users/christoph/UNSW/linelists/thar_mm.arc',skipline=8,fixedformat=[13]))     #in Angstroms
ll15 = linelist[(linelist > 6820.) & (linelist < 6970.)]
#from Zemax as a starting point
dispsol = np.load('/Users/christoph/UNSW/dispsol/mean_dispsol_by_orders_from_zemax.npy').item()   
plt.plot(xx,np.array(dispsol['order90']['model'](xx)*10)[::-1] - thar_fit_14(xx))
plt.plot(xx,np.array(dispsol['order89']['model'](xx)*10)[::-1] - thar_fit_15(xx))

ord = 'order_40'
ordnum = ord[-2:]
m = 105 - int(ordnum)
#data = thflux[ord]
data = thflux2[ord]
fitted_line_pos = fit_emission_lines(data,varbeta=False,timit=True,verbose=True,thresh=400.,bgthresh=200.,maxthresh=5e5)
goodpeaks,mostpeaks,allpeaks = find_suitable_peaks(data,thresh=400.,bgthresh=200.,debug_level=3)    
plt.title(ord)
#prelim_wl = thar_fit(fitted_line_pos) + 76.5 - 0.7 + (1.4/4112.)*fitted_line_pos
# prelim_wl = np.array(np.array(dispsol['order'+str(m)]['model'](4112.-fitted_line_pos)*10)[::-1] + 2. - (1.2/4112.)*(4112.-fitted_line_pos))[::-1]
# prelim_wl_xx = np.array(np.array(dispsol['order'+str(m)]['model'](4112.-xx)*10)[::-1] + 2. - (1.2/4112.)*(4112.-xx))[::-1]
prelim_wl = np.array(np.array(dispsol['order'+str(m)]['model'](4112.-fitted_line_pos)*10) + 2. - (1.2/4112.)*(4112.-fitted_line_pos))
prelim_wl_xx = np.array(np.array(dispsol['order'+str(m)]['model'](4112.-xx)*10) + 2. - (1.2/4112.)*(4112.-xx))
plt.plot(prelim_wl_xx,data)
plt.scatter(prelim_wl, data[goodpeaks], marker='x', color='r', s=40)
plt.title(ord+'  (prelim; collapsed)')

line_number, refwlord = readcol('/Users/christoph/UNSW/linelists/AAT_folder/ThAr_linelist_order_'+ordnum+'.dat',fsep=';',twod=False)
lam = refwlord.copy()  

mask_order = np.array([0,1,2,3,6,8,11,12,14])
x = fitted_line_pos[mask_order]

thar_fit = np.poly1d(np.polyfit(x, lam, 3))
resid = thar_fit(x) - lam
single_rverr = 3e8 * (np.std(resid) / np.mean(lam))
plt.figure()
plt.plot(x,lam,'bo')
plt.plot(xx,thar_fit(xx),'r')
plt.plot(xx,prelim_wl_xx,'y--')
plt.figure()
plt.plot(x,resid,'bo')
plt.title('single RV err = '+str(single_rverr)+' m/s')


line_number, dumlist = readcol('/Users/christoph/UNSW/linelists/AAT_folder/dumlist'+ordnum+'.dat',fsep=';',twod=False)
# prelim_wl = np.array([ 6145.24431076,  6151.84836198,  6155.12784905,  6155.46986366,
#         6161.29164742,  6164.4444754 ,  6165.09659353,  6169.83366727,
#         6170.19022628,  6172.31165711,  6173.13884062,  6178.51872573,
#         6180.81244083,  6182.74533812,  6184.9202941 ,  6188.29736647,
#         6192.1107481 ,  6198.48289355,  6201.38852608,  6203.79920293,
#         6207.56108194,  6212.89250428,  6216.35753654,  6225.02558645,
#         6226.88563639])
# dumlist = np.array([ 6145.4411,  6151.9929,  6154.0682,  6157.0878,  6161.3534,
#         6164.4796,  6173.0964,  6178.4315,  6180.705 ,  6182.6217,
#         6188.1251,  6191.9053,  6198.2227,  6203.4925,  6207.2201,
#         6212.503 ,  6215.9383,  6220.0112,  6221.3192,  6224.5272,
#         6226.3697])
# do the cross-checking
wlmask = np.zeros(len(dumlist),dtype='bool')
posmask = []
for i in range(len(prelim_wl)):
    print('Trying to match line '+str(i+1)+'/'+str(len(prelim_wl))+':')
    checkarr = np.abs(dumlist - prelim_wl[i]) < 0.5
    if np.sum(checkarr) == 0:
        print('Unknown line...')
    elif np.sum(checkarr) == 1:
        posmask.append(i)
        wlmask += checkarr
    else:
        print('FUGANDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
#check the two masks have the same length
if len(posmask) != np.sum(wlmask):
    print('FUGANDA')
testpos_order = fitted_line_pos[posmask]
x = testpos_order.copy()
refwlord = dumlist[wlmask]
lam = refwlord.copy()  
thar_fit = np.poly1d(np.polyfit(x, lam, 5))
resid = thar_fit(x) - lam
single_rverr = 3e8 * (np.std(resid) / np.mean(lam))
