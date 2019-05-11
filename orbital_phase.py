import numpy as np
import time
from readcol import readcol


def create_PT0_dict(path='/Users/christoph/OneDrive - UNSW/observing/AAT/', savefile=True):

    # read input file
    target, T0, P = readcol(path + 'toi_PT0_list.txt', twod=False)

    PT0_dict = {}
    for i in range(len(target)):
        PT0_dict[target[i]] = {'P':P[i], 'T0':T0[i]}

    if savefile:
        np.save(path + 'toi_PT0_dict.npy', PT0_dict)

    return PT0_dict



def calculate_orbital_phase(toi, jdnow=None, PT0_dict=None, t_ref = 2457000.):

    if PT0_dict is None:
        PT0_dict = np.load('/Users/christoph/OneDrive - UNSW/observing/AAT/toi_PT0_dict.npy').item()

    P = PT0_dict[toi.upper()]['P']
    T0 = PT0_dict[toi.upper()]['T0']

    if jdnow is None:
        # jdnow = jdnow()
        jdnow = time.time() / 86400. + 2440587.5

    phase = np.mod(jdnow - (t_ref + T0), P) / P

    return phase
