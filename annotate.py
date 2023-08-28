
import pickle
from glob import glob
import os
import pandas as pd
import csv
from csv import writer
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from natsort import natsorted

img_array=['1661551105864531986', '1661551107062699310', '1661551109092851870', '1661551109758633129','1661551109825180523']
laser_array=[]

path_joint = '/home/christos_sevastopoulos/Desktop/berdematex/Extracted_Data/heracleia2/joints/'
path_laser = '/home/christos_sevastopoulos/Desktop/berdematex/Extracted_Data/heracleia2/front_laser/'
path_img = '/home/christos_sevastopoulos/Desktop/berdematex/Extracted_Data/heracleia2/img/'
csv_name = 'csv_name'
threshold = 1
laser_threshold = 1.2 #(metra)
dataset_name = 'new'

if not os.path.exists(dataset_name):
    print("It doesnt exist! Lets create it ")
    os.makedirs(dataset_name)

# Creating list for scans and images
scans = glob(path_joint + '*pkl')
scans.sort()

# Laser part
lasers = glob(path_laser + '*pkl')
lasers.sort()

images =  glob(path_img + '*jpg')
sorted_images = natsorted(images)

output_path = "%s" % dataset_name + os.sep
limit= 1.0

new_index=[sorted_images.index(x) for x in [path_img+x+'.jpg' for x in img_array]]
scans_for_plot=[lasers[x] for x in new_index]
print(scans_for_plot)


for s in scans_for_plot:

    file_to_read_laser = open(s, "rb")
    loaded_dictionary_laser = pickle.load(file_to_read_laser)
    ranges = loaded_dictionary_laser["ranges"]
    print(loaded_dictionary_laser)
    print(ranges)
    ranges_reshaped = np.array(ranges).reshape(1, 1081)
    ranges_200_400 = ranges_reshaped[0][200:400]
    #print(ranges_200_400)
   # print((ranges))

    plt.plot(ranges)
    plt.show()




def label_maker(fn,laser_flag):   ### for heracleia

    if  (fn[0]<= threshold and fn[0]> -threshold) and (fn[1]<= threshold and fn[1]> -threshold) and (fn[2]<= threshold and fn[2]> -threshold) and (fn[3]<= 0.01 and fn[3]> -threshold):
        return 0

    if  (fn[0]<= -threshold ) and (fn[1]<= -threshold) and (fn[2]<= -threshold) and (fn[3]<= -threshold):
        return 0

    if  ((fn[0]> threshold+limit ) and (fn[1]> threshold+limit) and (fn[2]> threshold+limit) and (fn[3]> threshold+limit)) and (laser_flag == 1):

        return 1

    else:
        return 0


velocity_array = np.empty([1, 4])
csv_final = output_path + csv_name + '.csv'

with open(csv_final, 'w') as final_file:
    print(final_file)
    for i, s in enumerate(scans):
        file_to_read = open(s, "rb")
        loaded_dictionary = pickle.load(file_to_read)
        velocity = loaded_dictionary["velocity"]
        output_filename = os.path.basename(s).split('.')[0]
        print('joint_name: '+ output_filename)
        # Finding the laser data file that matches the joint data file based on their name order
        for j, l in enumerate(lasers):
            if  i == j:
                    for z, k in enumerate(images):
                         if  i == z:
                             output_filename = os.path.basename(k).split('.')[0]
                             print('image_name: '+ output_filename)
                    file_to_read_laser = open(l, "rb")
                    loaded_dictionary_laser = pickle.load(file_to_read_laser)
                    ranges = loaded_dictionary_laser["ranges"]
                    output_filename = os.path.basename(l).split('.')[0]
                    print('laser_name: '+ output_filename)
                    print('image_number: '+ str(i))
                    ranges_reshaped = np.array(ranges).reshape(1, 1081)
                    ranges_200_400 = ranges_reshaped[0][200:400]
                    if (i == 35):
                        print(ranges_200_400)
                    laser_flag = 1
                    if np.any(ranges_200_400 <= laser_threshold):
                        laser_flag = 0
                    print('laser_flag: '+ str (laser_flag))
                    print('velocity: '+ str (velocity))
                    vel_out = label_maker(velocity,laser_flag)
                    print('output: '+  str (vel_out))
                    tmp = np.array(velocity).reshape(1, 4)
                    velocity_array = np.append(velocity_array, tmp, axis=0)
                    final_file.write(output_filename)
                    final_file.write('\t' + ',' + str(vel_out) + '\n')
                    print('')
