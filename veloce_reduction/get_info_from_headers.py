'''
Created on 7 May 2018

@author: christoph
'''


import glob
import astropy.io.fits as pyfits



#path = '/Users/christoph/UNSW/veloce_spectra/test1/'



def identify_obstypes(path):
    """
    Identify the type of observation from the card in the FITS header, and create lists of filename for the different observation types.
    """
    
    
    file_list = glob.glob(path+"*.fits")
    
    bias_list = []
    dark_list = []
    sky_list = []
    # lflat_list = []
    skyflat_list = []
    # arc_list = []
    fibre_flat_list = []
    unknown_list = []
    # white_list = []
    thar_list = []
    thxe_list = []
    laser_list = []
    stellar_list = []
    
    for file in file_list:
        h = pyfits.getheader(file)
        # type = h['exptype']
        obstype = h['NDFCLASS']
        
        if obstype.upper() == 'BIAS':
            bias_list.append(file)
        elif obstype.upper() == 'DARK':
            dark_list.append(file)
        elif obstype.upper() == 'MFSKY':
            sky_list.append(file)
        # elif obstype.upper() == 'LFLAT':
        #     lflat_list.append(file)
        elif obstype.upper() == 'SFLAT':
            skyflat_list.append(file)
        # elif obstype.upper() == 'MFARC':
        #     arc_list.append(file)
        elif obstype.upper() == 'MFFFF':
            fibre_flat_list.append(file)
        elif obstype.upper() == 'MFOBJECT':
            if h['OBJECT'].lower() == 'thar':
                thar_list.append(file)
            elif h['OBJECT'].lower() == 'thxe':
                thxe_list.append(file)
            elif h['OBJECT'].lower() == 'laser':
                laser_list.append(file)
            else:
                stellar_list.append(file)
        # elif obstype.upper() in ('FLAT', 'WHITE'):
        #     white_list.append(file)
        # elif obstype.upper() == 'THAR':
        #     thar_list.append(file)
        # elif obstype.upper() == 'THXE':
        #     thxe_list.append(file)
        # elif obstype.upper() == 'LASER':
        #     laser_list.append(file)
        # elif obstype.upper() == 'STELLAR':
        #     stellar_list.append(file)
        else:
            print('WARNING: unknown exposure type encountered for',file)
            unknown_list.append(file)
    
    return bias_list,dark_list,sky_list,skyflat_list,fibre_flat_list,thar_list,thxe_list,laser_list,stellar_list,unknown_list




def get_obstype_lists_temp(path, pattern=None, weeding=True):

    if pattern is None:
        file_list = glob.glob(path + "*.fits")
    else:
        file_list = glob.glob(path + '*' + pattern + '*.fits')
    
    
    # first weed out binned observations
    if weeding:
        unbinned = []
        binned = []
        for file in file_list:
            xdim = pyfits.getval(file, 'NAXIS2')
            if xdim == 4112:
                unbinned.append(file)
            else:
                binned.append(file)
    else:
        unbinned = file_list

    # prepare output lists
    if weeding:
        acq_list = binned[:]
    else:
        acq_list = []
    bias_list = []
    dark_list = []
    flat_list = []
    skyflat_list = []
    domeflat_list = []
    arc_list = []
    thxe_list = []
    laser_list = []
    laser_and_thxe_list = []
    stellar_list = []
    unknown_list = []

    for file in unbinned:
        obj_type = pyfits.getval(file, 'OBJECT')

        if obj_type.lower() == 'acquire':
            if not weeding:
                acq_list.append(file)
        elif obj_type.lower().startswith('bias'):
            bias_list.append(file)
        elif obj_type.lower().startswith('dark'):
            dark_list.append(file)
        elif obj_type.lower().startswith('flat'):
            flat_list.append(file)
        elif obj_type.lower().startswith('skyflat'):
            skyflat_list.append(file)
        elif obj_type.lower().startswith('domeflat'):
            domeflat_list.append(file)
        elif obj_type.lower().startswith('arc'):
            arc_list.append(file)
        elif obj_type.lower() in ["thxe","thxe-only", "simth"]:
            thxe_list.append(file)
        elif obj_type.lower() in ["lc","lc-only","lfc","lfc-only", "simlc"]:
            laser_list.append(file)
        elif obj_type.lower() in ["thxe+lfc","lfc+thxe","lc+simthxe","lc+thxe"]:
            laser_and_thxe_list.append(file)
        elif obj_type.lower().startswith(("wasp","proxima","kelt","toi","tic","hd","hr","hip","gj","gl","ast","alpha","beta","gamma",
                                          "delta","tau","ksi","ach","zeta","ek",'1', '2', '3', '4', '5', '6', '7', '8', '9')):
            stellar_list.append(file)
        else:
            unknown_list.append(file)

    return acq_list, bias_list, dark_list, flat_list, skyflat_list, domeflat_list, arc_list, thxe_list, laser_list, laser_and_thxe_list, stellar_list, unknown_list





def get_obs_coords_from_header(fn):
    h = pyfits.getheader(fn)
    lat = h['LAT_OBS']
    long = h['LONG_OBS']
    alt = h['ALT_OBS']
    return lat,long,alt
    




