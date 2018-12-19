'''
Created on 13 Dec. 2018

@author: christoph
'''


from readcol import readcol


path = '/Users/christoph/OneDrive - UNSW/telemetry/'

# # read detector temperature file(s)
# with open(path + 'arccamera.txt') as det_temp_file:
#     for line in det_temp_file:
#         if line[0] == '#':
#             column_names = line.lstrip('#').split()
#         else:
#             timestamp, det_temp, cryohead_temp, perc_heater = line.rstrip('\n')
        

timestamp_cam, det_temp, cryohead_temp, perc_heater = readcol(path + 'arccamera.txt', skipline=1, twod=False)

timestamp_mech, temp_internal, temp_external, temp_room, temp_elec_cabinet, p_internal, p_room, p_regulator, p_set_point, p_rosso_cryo, h_internal, h_external, h_room, focus = readcol(path + 'velocemech.txt', skipline=1, twod=False)

timestamp_therm, temp_mc_setpoint, temp_mc_int, temp_extencl, temp_extencl_setpoint, state_mc = readcol(path + 'velocetherm.txt', skipline=1, twod=False)

timestamp_int_therm, temp_enc_setpoint, temp_enc_target, temp_cryo_setpoint, temp_sensor_1, temp_sensor_2, temp_sensor_3, temp_sensor_4, temp_sensor_5, temp_sensor_6, temp_sensor_7,\
    pwm_1, pwm_2, pwm_3, pwm_4, pwm_5 = readcol(path + 'veloceinttherm.txt', skipline=1, twod=False)




