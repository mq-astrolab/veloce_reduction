import numpy as np
import time
from readcol import readcol
import os
import glob
import astropy.io.fits as pyfits


def create_PT0_dict(path='/Users/christoph/OneDrive - UNSW/observing/AAT/', savefile=True):

    # read input file
    targets, T0, P = readcol(path + 'toi_PT0_list.txt', twod=False)

    PT0_dict = {}
    for i in range(len(targets)):
        PT0_dict[targets[i]] = {'P':P[i], 'T0':T0[i]}

    if savefile:
        np.save(path + 'toi_PT0_dict.npy', PT0_dict)

    return PT0_dict



def calculate_orbital_phase(toi, jd=None, PT0_dict=None, t_ref = 2457000.):

    """
    jd can be an array
    """

    if PT0_dict is None:
        PT0_dict = np.load('/Users/christoph/OneDrive - UNSW/observing/AAT/toi_PT0_dict.npy').item()

    P = PT0_dict[toi.upper()]['P']
    T0 = PT0_dict[toi.upper()]['T0']

    if jd is None:
        # jd = jdnow()
        jd = time.time() / 86400. + 2440587.5

    phase = np.mod(jd - (t_ref + T0), P) / P

    return phase





def create_velobs_dict(path='/Users/christoph/OneDrive - UNSW/observing/AAT/', savefile=True):

    # read input file
    targets, T0, P = readcol(path + 'toi_PT0_list.txt', twod=False)

    # initialise dictionary
    vo = {}

    # loop over all TOIs
    for toi, t0, per in zip(targets, T0, P):

        # initialise sub-dictionary for this object
        vo[toi] = {}

        # loop over all "names"
        name = '200'  # eventually get this from toi
        # target_matches = [s for s in all_target_list if name in s]
        fn_list = [fn for fn in all_obs_list if name in ((fn.split('/')[-1]).split('_')[0]).split('+')[0]]
        fn_list.sort()

        # prepare dictionary entry for this target
        vo[toi]['filenames'] = fn_list
        vo[toi]['nobs'] = len(fn_list)
        vo[toi]['P'] = per
        vo[toi]['T0'] = t0
        vo[toi]['JD'] = []
        vo[toi]['texp'] = []
        vo[toi]['obsnames'] = []
        vo[toi]['phase'] = []
        # vo[toi]['snr'] = []     # would be nice as an upgrade in the future
        # vo[toi]['cal'] = []     # would be nice as an upgrade in the future (calibration source: LFC, ThXe, interp?)
        # vo[toi]['rv'] = []     # would be nice as an upgrade in the future
        days = []
        seq = []
        # fill dictionary
        # loop over all observations for this target
        for file in fn_list:
            h = pyfits.getheader(file)
            vo[toi]['JD'].append(
                h['UTMJD'] + 2.4e6 + 0.5 + (0.5 * h['ELAPSED'] / 86400.))  # use plain JD here, in order to avoid confusion
            vo[toi]['texp'].append(h['ELAPSED'])
            obsname = (file.split('/')[-1]).split('_')[1]
            vo[toi]['obsnames'].append(obsname)
            days.append(obsname[:5])
            seq.append(obsname[5:])
        vo[toi]['phase'].append(calculate_orbital_phase(toi, vo[toi]['JD']))
        # check which exposures are adjacent to determine the number of epochs
        vo[toi]['nepochs'] = vo[toi]['nobs']
        for d in set(days):
            ix = [i for i, day in enumerate(days) if day == d]
            rundiff = np.diff(np.array(seq)[ix].astype(int))
            vo[toi]['nepochs'] -= np.sum(rundiff == 1)

    if savefile:
        np.save(path + 'velobs.npy', vo)

    return vo



###############
# representative plot of a normalized circular orbit with the orbital phases of the obstimes indicated
x = np.linspace(0,1,1000)
plt.plot(x,np.sin(2.*np.pi*x),'k')
phi = np.squeeze(vo[toi]['phase'])
plt.plot(phi, np.sin(2.*np.pi*phi),'ro')
plt.xlabel('orbital phase')
plt.ylabel('dRV / K')
plt.title(toi)



PT0_dict = np.load('/Users/christoph/OneDrive - UNSW/observing/AAT/toi_PT0_dict.npy').item()

if laptop:
    redpath = '/Users/christoph/data/reduced/'
else:
    redpath = '/Volumes/BERGRAID/data/veloce/reduced/'

# count all the nightly directories only (ie not the other ones like "tauceti")
datedir_list = glob.glob(redpath + '20*')
datedir_list.sort()
print('Searching for reduced spectra in ' + str(len(datedir_list)) + ' nights of observations...')

# all_target_list = []
all_obs_list = []

for datedir in datedir_list:
    datedir += '/'
    obs_list = glob.glob(datedir + '*optimal*')
    all_obs_list.append(obs_list)
    # target_set = set([(fn.split('/')[-1]).split('_')[0] for fn in obs_list])
    # all_target_list.append(list(target_set))

all_obs_list = [item for sublist in all_obs_list for item in sublist]
all_target_list = [((fn.split('/')[-1]).split('_')[0]).split('+')[0] for fn in all_obs_list]
# all_targets = set([item for sublist in all_target_list for item in sublist])


def get_obslist_dict(targets, laptop=False):

    if laptop:
        redpath = '/Users/christoph/data/reduced/'
    else:
        redpath = '/Volumes/BERGRAID/data/veloce/reduced/'

    # count all the nightly directories only (ie not the other ones like "tauceti")
    datedir_list = glob.glob(redpath + '20*')
    datedir_list.sort()
    print('Searching for reduced spectra in ' + str(len(datedir_list)) + ' nights of observations...')

    # all_target_list = []
    all_obs_list = []

    for datedir in datedir_list:
        datedir += '/'
        obs_list = glob.glob(datedir + '*optimal*')
        all_obs_list.append(obs_list)
        # target_set = set([(fn.split('/')[-1]).split('_')[0] for fn in obs_list])
        # all_target_list.append(list(target_set))

    all_obs_list = [item for sublist in all_obs_list for item in sublist]
    all_target_list = [((fn.split('/')[-1]).split('_')[0]).split('+')[0] for fn in all_obs_list]
    # all_targets = set([item for sublist in all_target_list for item in sublist])

    unique_targets = set(all_target_list)




    # now this is tedious, but weed out and remove duplicates
    unique_targets.remove('master')


    obslists_dict = {}

    return




# class veloce_observations

