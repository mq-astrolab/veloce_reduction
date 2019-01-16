'''
Created on 13 Dec. 2018

@author: christoph
'''


from readcol import readcol
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np

path = '/Users/christoph/OneDrive - UNSW/telemetry/'

# tau Ceti observation times
jd_sep = readcol('/Volumes/BERGRAID/data/veloce/reduced/tauceti/tauceti_with_LFC/' + 'tauceti_all_jds.dat')
jd_nov = readcol('/Volumes/BERGRAID/data/veloce/reduced/tauceti/tauceti_with_LFC/nov2018/' + 'tauceti_all_jds.dat')
jd = list(jd_sep) + list(jd_nov)

# # read detector temperature file(s)
# with open(path + 'arccamera.txt') as det_temp_file:
#     for line in det_temp_file:
#         if line[0] == '#':
#             column_names = line.lstrip('#').split()
#         else:
#             timestamp, det_temp, cryohead_temp, perc_heater = line.rstrip('\n')
        

timestamp_cam, det_temp, cryohead_temp, perc_heater = readcol(path + 'arccamera.txt', skipline=1, twod=False)
timestamp_cam_0, det_temp_0, cryohead_temp_0, perc_heater_0 = readcol(path + 'arccamera.txt.0', skipline=1, twod=False)
# timestamp_cam_1, det_temp_1, cryohead_temp_1, perc_heater_1 = readcol(path + 'arccamera.txt.1', skipline=1, twod=False)
cam_time = Time(timestamp_cam, format='unix')         # from 17/10/2018 - 27/11/2018
cam_time_0 = Time(timestamp_cam_0, format='unix')     # from 19/04/2018 - 17/10/2018
# cam_time_1 = Time(timestamp_cam_1, format='unix')

timestamp_mech, temp_internal, temp_external, temp_room, temp_elec_cabinet, p_internal, p_room, p_regulator, p_set_point, p_rosso_cryo, h_internal, h_external, h_room, focus = readcol(path + 'velocemech.txt', skipline=1, twod=False)
timestamp_mech_0, temp_internal_0, temp_external_0, temp_room_0, temp_elec_cabinet_0, p_internal_0, p_room_0, p_regulator_0, p_set_point_0, p_rosso_cryo_0, h_internal_0, h_external_0, h_room_0, focus_0 = readcol(path + 'velocemech.txt.0', skipline=1, twod=False)
timestamp_mech_1, temp_internal_1, temp_external_1, temp_room_1, temp_elec_cabinet_1, p_internal_1, p_room_1, p_regulator_1, p_set_point_1, p_rosso_cryo_1, h_internal_1, h_external_1, h_room_1, focus_1 = readcol(path + 'velocemech.txt.1', skipline=1, twod=False)
mech_time = Time(timestamp_mech, format='unix')       # from 26/10/2018 - 27/11/2018
mech_time_0 = Time(timestamp_mech_0, format='unix')   # from 23/09/2018 - 26/10/2018
mech_time_1 = Time(timestamp_mech_1, format='unix')   # from 13/08/2018 - 23/09/2018  

timestamp_therm, temp_mc_setpoint, temp_mc_int, temp_extencl, temp_extencl_setpoint, state_mc = readcol(path + 'velocetherm.txt', skipline=1, twod=False)
therm_time = Time(timestamp_therm, format='unix')

timestamp_int_therm, temp_enc_setpoint, temp_enc_target, temp_cryo_setpoint, temp_sensor_1, temp_sensor_2, temp_sensor_3, temp_sensor_4, temp_sensor_5, temp_sensor_6, temp_sensor_7,\
    pwm_1, pwm_2, pwm_3, pwm_4, pwm_5 = readcol(path + 'veloceinttherm.txt', skipline=1, twod=False)
timestamp_int_therm_0, temp_enc_setpoint_0, temp_enc_target_0, temp_cryo_setpoint_0, temp_sensor_1_0, temp_sensor_2_0, temp_sensor_3_0, temp_sensor_4_0, temp_sensor_5_0, temp_sensor_6_0, temp_sensor_7_0,\
    pwm_1_0, pwm_2_0, pwm_3_0, pwm_4_0, pwm_5_0 = readcol(path + 'veloceinttherm.txt.0', skipline=1, twod=False)
timestamp_int_therm_1, temp_enc_setpoint_1, temp_enc_target_1, temp_cryo_setpoint_1, temp_sensor_1_1, temp_sensor_2_1, temp_sensor_3_1, temp_sensor_4_1, temp_sensor_5_1, temp_sensor_6_1, temp_sensor_7_1,\
    pwm_1_1, pwm_2_1, pwm_3_1, pwm_4_1, pwm_5_0 = readcol(path + 'veloceinttherm.txt.1', skipline=1, twod=False)
int_therm_time = Time(timestamp_int_therm, format='unix')      # from 26/11/2018 - 27/11/2018
int_therm_time_0 = Time(timestamp_int_therm_0, format='unix')  # from 25/11/2018 - 26/11/2018
int_therm_time_1 = Time(timestamp_int_therm_1, format='unix')  # from 24/11/2018 - 25/11/2018



