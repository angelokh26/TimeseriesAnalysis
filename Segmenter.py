# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 12:47:48 2020

@author: Angelo Hollett angelokh26@gmail.com A00419149

Breaks a time series into specified duration chunks

"""

import numpy as np
import os
import matplotlib.pyplot as plt

directory = os.getcwd()

Lengths = []

ends_with = "5760.csv"

path = './raw_ident_BLS1NLS1'
os.chdir(path)
path = os.getcwd()

os.chdir('C:/Users/****/')
path = os.getcwd()

for file in os.listdir(path):
    if file.endswith(ends_with):
    
        data = np.genfromtxt(file, delimiter=",", skip_header=1)
    
        time = data[:,0]
        counts = data[:,1]     #count rate
        counts_err = data[:,2]

        max_time = max(time)
        Lengths.append(max_time)

LengthsKS = np.divide(Lengths, 1000)


############################### SEGMENTATION ##################################

for file in os.listdir(path):
    if file.endswith(ends_with):
        file_name = os.path.splitext(file)[0]
        
        data = np.genfromtxt(file, delimiter=",", skip_header=1)
    
        time = data[:,0]
        counts = data[:,1]     #count rate
        counts_err = data[:,2]
        
        data_mat = np.column_stack((time,counts,counts_err))
        
        
        seg = 250000
        dur = max(time)
        N = np.floor(dur/seg)
        
        rem = dur % seg    # remainder after dividing the duration by the seg
        rem_rat = rem/seg  # this remainder as a fractin (b/w 0 and 1)
        

        save_name = file_name + "_" + str(seg)
        
        seg_num_list = np.arange(0,(N), 1)
        if dur > seg:
            if rem_rat < 0.25:
                for i in seg_num_list:
                    
                    save_name_seg_start = int((0 +i*seg))
                    save_name_seg_end = int((seg + i*seg))
                    
                    data_mat_seg = data_mat[data_mat[:,0]<=(seg+i*seg),:]
                    data_mat_upper = data_mat[data_mat[:,0]>=(i*seg),:]
                    data_mat_range = data_mat_upper[data_mat_upper[:,0]<=(seg+i*seg),:]
                    
                    #np.savetxt("../Segments_250ks_Ident_NLS1BLS1/%s_%s_%s.csv" % (save_name, save_name_seg_start,save_name_seg_end ), data_mat_range, delimiter=",", fmt='%s', header='time, counts, counts_err')
                
            else:
                for i in seg_num_list:
                
                    save_name_seg_start = int((0 +i*seg))
                    save_name_seg_end = int((seg + i*seg))
                    
                    data_mat_seg = data_mat[data_mat[:,0]<=(seg+i*seg),:]
                    data_mat_upper = data_mat[data_mat[:,0]>=(i*seg),:]
                    data_mat_range = data_mat_upper[data_mat_upper[:,0]<=(seg+i*seg),:]
                    
                    #np.savetxt("../Segments_250ks_Ident_NLS1BLS1/%s_%s_%s.csv" % (save_name, save_name_seg_start,save_name_seg_end ), data_mat_range, delimiter=",", fmt='%s', header='time, counts, counts_err')   

        #np.savetxt("./Segments/%s.csv" % save_name, data_mat_seg, delimiter=",", fmt='%s', header='time, counts, counts_err')

#np.savetxt("log_x.csv", np.column_stack((y, x_log)), delimiter=",", fmt='%s', header='x')