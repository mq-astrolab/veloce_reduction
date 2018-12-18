'''
Created on 13 Dec. 2018

@author: christoph
'''


from readcol import readcol


path = '/Users/christoph/OneDrive - UNSW/telemetry/'

# read detector temperature file(s)
with open(path + 'arccamera.txt') as det_temp_file:
    for line in det_temp_file:
        if line[0] == '#':
            column_names = line.lstrip('#').split()
        else:
            timestamp, det_temp, cryohead_temp, perc_heater = line.rstrip('\n')
        

timestamp, det_temp, cryohead_temp, perc_heater = readcol(path + 'arccamera.txt', skipline=1, twod=False)

timestamp2, temp_internal, temp_external, temp_room, temp_elec_cabinet, p_internal, p_room, p_regulator, p_set_point, p_rosso_cryo, h_internal, h_external, h_room, focus = readcol(path + 'velocemech.txt', skipline=1, twod=False)