import pandas as pd
import numpy as np
import pyfits
import glob
from astroquery.vizier import Vizier
import time

def stats(directories):

    not_source_arr = ["test", "ThAr", "Flat", "Dark Frame", "Bias Frame", "Nothing", "ThXe", "ThXe test", "CuAr", "FLAT", "ARC - ThAr", "FOCUS", "Flat Field - Quartz", "EpsSGRTelluric", "Archenar", "testdark", "alphaGru", "Archenar Acq"]

    objectArr = []
    magArr = []
    expArr = []
    rArr = []

    filterArr = ["Umag", "Vmag", "Bmag", "Imag", "Rmag", "mpg"]

    for i in directories:
    	for j in glob.glob(i):
            object_name = pyfits.getheader(j)["OBJECT"]
            print(object_name)
            if object_name not in not_source_arr:
                result_table = Vizier.query_object(object_name)

                for i in result_table[2].keys():
                    if i in filterArr:
                        mag = result_table[2][0][i]
                        objectArr.append(pyfits.getheader(j)["OBJECT"])

                        magArr.append(mag)
                        expArr.append(pyfits.getheader(j)["EXPOSED"])
                        rArr.append(pyfits.getheader(j)["SPEED"])       	
        
    fluxArr = np.zeros(len(objectArr))
    rsArr = np.zeros(len(objectArr))
    bArr = np.zeros(len(objectArr))
    rvArr = np.zeros(len(objectArr))
    rvErrArr = np.zeros(len(objectArr))
    
    df = pd.DataFrame({'Object': objectArr, 'Magnitude': magArr, 'Total Flux': fluxArr, 'Exposure Time': expArr, 'Readout Speed': rArr, 'Binning': bArr, 'Radial Velocity': rvArr, 'RV Uncertainty': rvErrArr}, columns=['Object', 'Magnitude', 'Total Flux', 'Exposure Time', 'Readout Speed', 'Binning', 'Radial Velocity', 'RV Uncertainty'])
    
    print(df)

    df.to_csv("name"+"stats")

start = time.clock()

stats(["/Users/Brendan/Dropbox/Brendan/Veloce/Data/veloce/180814/ccd_3/14aug*", "/Users/Brendan/Dropbox/Brendan/Veloce/Data/veloce/180815/ccd_3/15aug*"])

end = time.clock()

print("Time Elapsed: ",end-start)