import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import astropy.io.fits as pyfits
import barycorrpy



def get_barycentric_correction(fn, h=0.01, w=0.01):

    height = u.Quantity(h, u.deg)
    width = u.Quantity(w, u.deg)
    r = Gaia.query_object_async(coordinate=coord, width=width, height=height)

    utmjd = pyfits.getval(fn, 'UTMJD') + 2.4e6
    ra = pyfits.getval(fn, 'MEANRA')
    dec = pyfits.getval(fn, 'MEANDEC')

    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')

    gaia_data = Gaia.query_object_async(coordinate=coord, width=width, height=height)

    bc = barycorrpy.get_BC_vel(JDUTC=utmjd, ra=ra, dec=dec, obsname='AAO', ephemeris='de430')

    #         # HMMM...using hip_id=xxx and actual coordinates from header makes a huge difference (~11m/s for the tau Ceti example I tried)!!!
    #         bc1 = barycorrpy.get_BC_vel(JDUTC=utmjd, hip_id=8102, obsname='AAO', ephemeris='de430')
    #         bc2 = barycorrpy.get_BC_vel(JDUTC=utmjd, ra=ra, dec=dec, obsname='AAO', ephemeris='de430')
    #
    #         #now append relints, wl-solution, and barycorr to extracted FITS file header
    #         outfn = path + obsname + '_extracted.fits'
    #         if os.path.isfile(outfn):
    #             #relative fibre intensities
    #             dum = append_relints_to_FITS(relints, outfn, nfib=19)
    #             #wavelength solution
    #             #pyfits.setval(fn, 'RELINT' + str(i + 1).zfill(2), value=relints[i], comment='fibre #' + str(fibnums[i]) + ' - ' + fibinfo[i] + ' fibre')
    #             #barycentric correction
    #             pyfits.setval(outfn, 'BARYCORR', value=np.array(bc[0])[0], comment='barycentric correction [m/s]')