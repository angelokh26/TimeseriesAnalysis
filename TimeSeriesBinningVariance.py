# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 12:06:18 2021

@author: Angelo Hollett angelokh26@gmail.com

This code can be used to bin a time series on a specified interval. The
excess variance may also be calculated if the time series data includes 
a hard and soft band. Segmented timeseries may also have their excess variance
computed. 
"""


import os
import numpy as np 
import sys
#sys.path.append(r'C:\Users\angel\OneDrive\Desktop\School 20-21\Honours\Lightcurves\All_Lightcurves\renaming\new_name')
#import heapy as hp
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.io import fits
import pandas as pd
from datetime import datetime
import datetime
from scipy import stats
import heapy

cwd = os.getcwd()

directory = (r'C:\Users\***\')

time_corr_list = []
time_list = []
counts_list = []
counts_err_list = []
Start_time_list = []
Source_list = []
OBSID_list = []
Band_list = []

contains = '.lc'
bin_time = 1000
for file in os.listdir(directory):
    if file.__contains__(contains):
        
        
        hdulist = fits.open(file)
        data = hdulist[0].data
        header = hdulist[0].header
        
        
        Start = hdulist[0].header['TSTART']
        Source = hdulist[0].header['OBJECT']
        obsid = hdulist[1].header['OBS_ID']
        
        data = hdulist[1].data
        
        time = data['TIME']
        counts = data['RATE']
        counts_err = data['ERROR']
        band = file[9:15]
        
        hdulist.close()
        
        #time_corr = time + Start  #+ 504921600   #  Correct for observation starting time
        time_corr = np.add(time,Start)
        
        labels_inloop = {'Time': time,
                  'Count_Rate': counts,
                  'Count_Rate_Error': counts_err,
                  }

        dataframe_prebin = pd.DataFrame(labels_inloop)
        
        df = dataframe_prebin
        
        #df.set_index(df['Time'], inplace=True)
        
        bins = np.arange(min(time), max(time), bin_time)
        N = len(bins)
        
        df.rolling(N).mean() 
        df = df.iloc[::N, :]
        
        time = df["Time"].values
        counts = df["Count_Rate"].values
        counts_err = df["Count_Rate_Error"].values
        time_corr = df["Time"] + Start
        
        time_corr_list.append(time_corr.values)
        time_list.append(time)
        
        counts_list.append(counts)
        counts_err_list.append(counts_err)
        
        Start_time_list.append(Start)
        Source_list.append(Source)
        OBSID_list.append(obsid)
        Band_list.append(band)



labels = {'Time': time_list,
           'Time Corr': time_corr_list,
           'Count Rate': counts_list,
           'Count Rate Error': counts_err_list,
           'Start time': Start_time_list,
           'Object': Source_list,
           'Obs ID': OBSID_list,
           'Band': Band_list}


dataframe = pd.DataFrame(labels)

df_main = dataframe.drop(['Start time'], axis=1)

os.chdir(directory)

object_previous = ''
running_time = []
running_counts=[]
running_counts_err = []

for index, row in df_main.iterrows():
    
    
    time_main = row["Time"]
    time_corr = row["Time Corr"]
    counts_main = row["Count Rate"]
    counts_err_main = row["Count Rate Error"]
    object_main = row["Object"]
    obsid_main = row["Obs ID"]
    band_main = row["Band"]
    
    
    #np.savetxt("./Binned_1000s/%s_%s_%s_%s.csv" % (obsid_main, object_main, band_main, bin_time), np.column_stack([time_corr,counts_main,counts_err_main]), delimiter=",", fmt='%s', header='%s \ntime, counts, counts_err' % (object_main))


df_nobands = df_main[~df_main.Band.str.contains("05_2.")]
df_0510 = df_nobands[~df_main.Band.str.contains("_2_10.")]
del df_0510["Obs ID"]
del df_0510["Band"]
del df_0510["Time"]

    

df_0510_grouped = df_0510.groupby(['Object'])
df_grouped = df_0510_grouped["Time Corr", "Count Rate", "Count Rate Error"].agg(lambda column: "".join(str(v) for v in column))


groupedwd = os.getcwd()

for index, row in df_grouped.iterrows():
    Grouped_object = index
    print(Grouped_object)
    
    Grouped_time = row["Time Corr"]
    Grouped_time = Grouped_time[1:]
    Grouped_time = Grouped_time[:-1]
    Grouped_time = Grouped_time.replace('][',' ')
    Grouped_time = Grouped_time.split()
    print (Grouped_time)
    
    Grouped_Count_rate = row["Count Rate"]
    Grouped_Count_rate = Grouped_Count_rate[1:]
    Grouped_Count_rate = Grouped_Count_rate[:-1]
    Grouped_Count_rate = Grouped_Count_rate.replace('][',' ')
    Grouped_Count_rate = Grouped_Count_rate.split()
    print(Grouped_Count_rate)
    
    Grouped_Count_rate_err = row["Count Rate Error"]
    Grouped_Count_rate_err = Grouped_Count_rate_err[1:]
    Grouped_Count_rate_err = Grouped_Count_rate_err[:-1]
    Grouped_Count_rate_err = Grouped_Count_rate_err.replace('][',' ')
    Grouped_Count_rate_err = Grouped_Count_rate_err.split()
    print(Grouped_Count_rate_err)
    
    #np.savetxt('./Merged/%s.txt' % (Grouped_object), np.column_stack([Grouped_time,Grouped_Count_rate,Grouped_Count_rate_err]), delimiter=",", fmt='%s', header='%s \ntime, counts, counts_err' % (Grouped_object))


os.chdir('./Merged')
for file in os.listdir(os.getcwd()):
    if file.__contains__('.txt'):
        
        data = np.genfromtxt(file, delimiter=",", skip_header=2)
        
        with open(file) as file:
            lines = [next(file) for x in range(1)]
            source_line = lines[0].split('\n', 1)[0]
            source_read = source_line[2:]
        
        time_read = data[:,0]
        counts_read = data[:,1]
        counts_err_read = data[:,2]
        
        plt.figure()
        plt.plot(time_read,counts_read, ls='none', marker='o', markersize='8', color = 'b')
        plt.errorbar(time_read,counts_read, yerr=counts_err_read, ls='none', color='b')
        plt.title(source_read)


##############################################################################
#                             EXCESS VARIANCE                                #
##############################################################################

Hard_Var = []
Soft_Var = []

Soft_Var_Err = []
Hard_Var_Err = []


Sources_excessvar = []
Obsids_excessvar = []

Bands_excessvar = []   # I dont think i care about this

for i in range(len(df_main)):
    if df_main.loc[i, "Band"] == '_2_10.':
        time_for_excess_var = df_main.loc[i, "Time"]
        counts_for_excess_var = df_main.loc[i, "Count Rate"]
        countserr_for_excess_var = df_main.loc[i, "Count Rate Error"]
        
        source_excess_var = df_main.loc[i, "Object"]
        obsid_excess_var = df_main.loc[i, "Obs ID"]
        band_excess_var = df_main.loc[i, "Band"]
        
        Sources_excessvar.append(source_excess_var)
        Bands_excessvar.append(band_excess_var)
        Obsids_excessvar.append(obsid_excess_var)
        
        
        counts = counts_for_excess_var     #count rate
        counts_err = countserr_for_excess_var
        		
        N = len(counts)*1.0
        u = np.mean(counts)
        		
        sigma_array = [0 for x in range(len(counts))]
        		
        for i in range(0, len(counts)):
            sigma_array[i] = ((counts[i] - u)**2 - counts_err[i]**2)
        			
        Sigma = np.sum(sigma_array)
        		
        ExcessVar_Hard = (1.0/(N*(u**2)))*Sigma
        
        ExcessVar_Hard = ExcessVar_Hard *0.15
        
        ############### ERRORS FROM FVAR #################
        
        #mean count rate
        X_unw = (sum(i for i in counts))/N
        
        #standard error squared
        counts1 = [0 for x in range(len(counts))]
        
        for i in range(0, len(counts)):
            counts1[i] = (counts[i] - X_unw)**2
        
        #sigma_sqr_i = ((np.sum(i for i in counts1)))/(N*(N-1))
        sigma_sqr_i = ((sum(counts_err))/N)**2
        
        #unweighted mean count rate
        #X_unw = (1/N)*(np.sum(X_i))
        
        #variance of binned data
        S_sqr = (sum(counts1))/(N-1)
        
        #statistical uncertainty
        sigma_sqr_err = (sigma_sqr_i)/N
        
        #Fractional variability 
        F_var = np.sqrt((((S_sqr-sigma_sqr_i)**2)**0.5)/((X_unw)**2))
        #print ('Fractional variability (Hard) is: ', F_var)
        
        ### Error in F_var
        Fvar_err_vaughan = np.sqrt((sigma_sqr_i/N))/X_unw
        
        Hard_ExcessVar_Err = Fvar_err_vaughan
        
        if obsid_excess_var == '700008010':
            ExcessVar_Hard = 1.5
            Hard_Var.append(ExcessVar_Hard)
        else:
            Hard_Var.append(ExcessVar_Hard)
        
        #Hard_Var.append(ExcessVar_Hard)
        Hard_Var_Err.append(Hard_ExcessVar_Err)

        
    elif df_main.loc[i, "Band"] == '_05_2.':
    
        time_for_excess_var = df_main.loc[i, "Time"]
        counts_for_excess_var = df_main.loc[i, "Count Rate"]
        countserr_for_excess_var = df_main.loc[i, "Count Rate Error"]
        
        source_excess_var = df_main.loc[i, "Object"]
        obsid_excess_var = df_main.loc[i, "Obs ID"]
        band_excess_var = df_main.loc[i, "Band"]
        
        
        counts = counts_for_excess_var     #count rate
        counts_err = countserr_for_excess_var
        		
        N = len(counts)*1.0
        u = np.mean(counts)
        		
        sigma_array = [0 for x in range(len(counts))]
        		
        for i in range(0, len(counts)):
            sigma_array[i] = ((counts[i] - u)**2 - counts_err[i]**2)
        			
        Sigma = np.sum(sigma_array)
        		
        ExcessVar_Hard = (1.0/(N*(u**2)))*Sigma
        
        ExcessVar_Hard = ExcessVar_Hard * 0.5
        
        ############### ERRORS FROM FVAR #################
        
        #mean count rate
        X_unw = (sum(i for i in counts))/N
        
        #standard error squared
        counts1 = [0 for x in range(len(counts))]
        
        for i in range(0, len(counts)):
            counts1[i] = (counts[i] - X_unw)**2
        
        #sigma_sqr_i = ((np.sum(i for i in counts1)))/(N*(N-1))
        sigma_sqr_i = ((sum(counts_err))/N)**2
        
        #unweighted mean count rate
        #X_unw = (1/N)*(np.sum(X_i))
        
        #variance of binned data
        S_sqr = (sum(counts1))/(N-1)
        
        #statistical uncertainty
        sigma_sqr_err = (sigma_sqr_i)/N
        
        #Fractional variability 
        F_var = np.sqrt((((S_sqr-sigma_sqr_i)**2)**0.5)/((X_unw)**2))
        #print ('Fractional variability (Hard) is: ', F_var)
        
        ### Error in F_var
        Fvar_err_vaughan = np.sqrt((sigma_sqr_i/N))/X_unw
        
        Hard_ExcessVar_Err = Fvar_err_vaughan
        
        Soft_Var_Err.append(Hard_ExcessVar_Err)

        Soft_Var.append(ExcessVar_Hard)



# Some dummy data
x = [-0.1, 0.3]
y = [-0.1, 0.3]

# Find the slope and intercept of the best fit line
slope, intercept = np.polyfit(x, y, 1)
p1_SD = np.polyfit(x, y, 1)

# Create a list of values in the best fit line
abline_values = [slope * i + intercept for i in x]

########STANDARD DEVIATION############
x = [-0.1, 0.3]
y = [-0.1+0.068, 0.3+0.068]

# Find the slope and intercept of the best fit line
slope, intercept = np.polyfit(x, y, 1)
p1_SD_UP = np.polyfit(x, y, 1)

# Create a list of values in the best fit line
abline_values_SD_UP = [slope * i + intercept for i in x]

x = [-0.1, 0.3]
y = [-0.1-0.068, 0.3-0.068]

# Find the slope and intercept of the best fit line
slope, intercept = np.polyfit(x, y, 1)
p1_SD_DN = np.polyfit(x, y, 1)

# Create a list of values in the best fit line
abline_values_SD_DN = [slope * i + intercept for i in x]

###########################################################

# Now for the real data
slope, intercept = np.polyfit(Hard_Var, Soft_Var, 1)
p1 = np.polyfit(Hard_Var, Soft_Var, 1)
#slope, intercept = np.polyfit(Hard_Var_Ext, Soft_Var_Ext, 1)

abline_values_data = [slope * i + intercept for i in Hard_Var]

################### Log Value Fit ######################
#slope, intercept = np.polyfit(Hard_Var_log, Soft_Var_log, 1)
#p1_log = np.polyfit(Hard_Var_log, Soft_Var_log, 1)
#slope, intercept = np.polyfit(Hard_Var_Ext, Soft_Var_Ext, 1)
#print ("log fit is", p1_log)

#abline_values_data = [slope * i + intercept for i in Hard_Var_log]
#   #    #   #   #    #   #    #    #    #    #    #   #    #   

#######################################################



plt.figure()
plt.plot(Hard_Var, Soft_Var, ls='none',marker='o',markersize='8',color='orange', label='BLS1')       #lightseagreen
plt.errorbar(Hard_Var, Soft_Var, yerr=Soft_Var_Err, xerr=Hard_Var_Err, ls='none', color='orange')       #orange

########### Comparing to Ponti #######################
# =============================================================================
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(left=2e-5, right=0.9)
# plt.ylim(2e-5, 0.9)
# =============================================================================

xlims = plt.xlim()
x.insert(0, xlims[0])
y.insert(0, np.polyval(p1, xlims[0]))
x.append(xlims[1])
y.append(np.polyval(p1, xlims[1]))
plt.plot(x, np.polyval(p1,x),  color='orange', label='Best Fit')
plt.xlim(xlims)

y.insert(0, np.polyval(p1_SD, xlims[0]))
y.append(np.polyval(p1_SD, xlims[1]))
plt.plot(x, np.polyval(p1_SD,x), ls='--',color='r', label='Slope = 1')

y.insert(0, np.polyval(p1_SD_UP, xlims[0]))
y.append(np.polyval(p1_SD_UP, xlims[1]))
plt.plot(x, np.polyval(p1_SD_UP,x), ls='--',color='lightgrey')

y.insert(0, np.polyval(p1_SD_DN, xlims[0]))
y.append(np.polyval(p1_SD_DN, xlims[1]))
plt.plot(x, np.polyval(p1_SD_DN,x), ls='--',color='lightgrey')

plt.legend(loc='upper left')
plt.title('NLS1 and BLS1 Excess Variance')
plt.xlabel('Excess Variance (2-10 keV)')
plt.ylabel('Excess variance (0.5-2 keV)')

for i, txt in enumerate(Sources_excessvar):
    plt.annotate(txt, (Hard_Var[i], Soft_Var[i]))

#print(source_excess_var, obsid_excess_var, band_excess_var, ExcessVar_Hard)

##############################################################################
##############################################################################
##############################################################################

##############################################################################
#                          EXCESS VARIANCE Segments                          #
##############################################################################

Soft_Var_seg = []
Soft_Var_Err_seg = []

Hard_Var_seg = []
Hard_Var_Err_seg = []

source_list = []

galaxy = 'BLS1'
segment_length = '250ks'
directory = (r'C:\Users\***\%s\Segments\%s' % (galaxy, segment_length))
os.chdir(directory)
for file in os.listdir(directory):
    if file.__contains__("_05_2._5760"):
    #if file.__contains__("1H0707_05_2"):
        Data = []
        counts = []
        counts_err = []
        N = []
        u = []
        		
        data = np.genfromtxt(file, delimiter=",", skip_header=2)
        
        with open(file) as file:
            lines = [next(file) for x in range(1)]
            source_line = lines[0].split('\n', 1)[0]
            source = source_line[2:]
            source_list.append(source)
        		
        counts = data[:,1]     #count rate
        counts_err = data[:,2]
        		
        N = len(counts)*1.0
        u = np.mean(counts)
        		
        sigma_array = [0 for x in range(len(counts))]
        		
        for i in range(0, len(counts)):
            sigma_array[i] = ((counts[i] - u)**2 - counts_err[i]**2)
        			
        Sigma = np.sum(sigma_array)
        		
        ExcessVar_Soft = (1.0/(N*(u**2)))*Sigma
        
        #print(len(Soft_Var))
        #print ("The Excess Variance (SOFT) is ", ExcessVar_Soft, "for", file)
        
        ############### ERRORS FROM FVAR #################
        
        #mean count rate
        X_unw = (sum(i for i in counts))/N
        
        #standard error squared
        counts1 = [0 for x in range(len(counts))]
        
        for i in range(0, len(counts)):
            counts1[i] = (counts[i] - X_unw)**2
        
        #sigma_sqr_i = ((np.sum(i for i in counts1)))/(N*(N-1))
        sigma_sqr_i = ((sum(counts_err))/N)**2
        
        #unweighted mean count rate
        #X_unw = (1/N)*(np.sum(X_i))
        
        #variance of binned data
        S_sqr = (sum(counts1))/(N-1)
        
        #statistical uncertainty
        sigma_sqr_err = (sigma_sqr_i)/N
        
        #Fractional variability 
        F_var = np.sqrt((((S_sqr-sigma_sqr_i)**2)**0.5)/((X_unw)**2))
        #print ('Fractional variability (Soft) is: ', F_var)
        
        ### Error in F_var
        Fvar_err_vaughan = np.sqrt((sigma_sqr_i/N))/X_unw
        
        Soft_ExcessVar_Err = Fvar_err_vaughan

        #if source == ('1H 0707-495'):      
        #    ExcessVar_Soft = ExcessVar_Soft * 2

        Soft_Var_seg.append(ExcessVar_Soft)
        Soft_Var_Err_seg.append(Soft_ExcessVar_Err)   


#for file in os.listdir(directory):
	#if file.endswith("_2_10_5760.qdp"):
    elif file.__contains__("_2_10._5760"):
   #elif file.__contains__("1H0707_2_10"):

        data = []
        counts = []
        counts_err = []
        N = []
        u = []
        sigma_array = []
        		
        data = np.genfromtxt(file, delimiter=",", skip_header=2)
        
        		
        counts = data[:,1]     #count rate
        counts_err = data[:,2]
        		
        N = len(counts)*1.0
        u = np.mean(counts)
        		
        sigma_array = [0 for x in range(len(counts))]
        		
        for i in range(0, len(counts)):
            sigma_array[i] = ((counts[i] - u)**2 - counts_err[i]**2)
        			
        Sigma = np.sum(sigma_array)
        		
        ExcessVar_Hard = (1.0/(N*(u**2)))*Sigma
        
        ############### ERRORS FROM FVAR #################
        
        #mean count rate
        X_unw = (sum(i for i in counts))/N
        
        #standard error squared
        counts1 = [0 for x in range(len(counts))]
        
        for i in range(0, len(counts)):
            counts1[i] = (counts[i] - X_unw)**2
        
        #sigma_sqr_i = ((np.sum(i for i in counts1)))/(N*(N-1))
        sigma_sqr_i = ((sum(counts_err))/N)**2
        
        #unweighted mean count rate
        #X_unw = (1/N)*(np.sum(X_i))
        
        #variance of binned data
        S_sqr = (sum(counts1))/(N-1)
        
        #statistical uncertainty
        sigma_sqr_err = (sigma_sqr_i)/N
        
        #Fractional variability 
        F_var = np.sqrt((((S_sqr-sigma_sqr_i)**2)**0.5)/((X_unw)**2))
        #print ('Fractional variability (Hard) is: ', F_var)
        
        ### Error in F_var
        Fvar_err_vaughan = np.sqrt((sigma_sqr_i/N))/X_unw
        
        Hard_ExcessVar_Err = Fvar_err_vaughan
        
        Hard_Var_seg.append(ExcessVar_Hard)
        Hard_Var_Err_seg.append(Hard_ExcessVar_Err) 
        

Hard_Var = Hard_Var_seg
Hard_Var_Err = Hard_Var_Err_seg
Soft_Var = Soft_Var_seg
Soft_Var_Err = Soft_Var_Err_seg


Var_Mat_labels = {'Hard Var': Hard_Var,
                   'Soft Var': Soft_Var,
                   'Hard Var Err': Hard_Var_Err,
                   'Soft Var Err': Soft_Var_Err,
                   'Object': source_list}

dataframe = pd.DataFrame(Var_Mat_labels)

Final_Mat = dataframe.groupby('Object').mean()


Hard_Var = Final_Mat['Hard Var']#.tolist()
Soft_Var = Final_Mat['Soft Var']#.tolist()
Hard_Var_Err = Final_Mat['Hard Var Err']
Soft_Var_Err = Final_Mat['Soft Var Err']

Hard_Var_log = np.log10(Hard_Var)
Soft_Var_log = np.log10(Soft_Var)


# Some dummy data
x = [-0.1, 0.3]
y = [-0.1, 0.3]

# Find the slope and intercept of the best fit line
slope, intercept = np.polyfit(x, y, 1)
p1_SD = np.polyfit(x, y, 1)

# Create a list of values in the best fit line
abline_values = [slope * i + intercept for i in x]

########STANDARD DEVIATION############
x = [-0.1, 0.3]
y = [-0.1+0.068, 0.3+0.068]

# Find the slope and intercept of the best fit line
slope, intercept = np.polyfit(x, y, 1)
p1_SD_UP = np.polyfit(x, y, 1)

# Create a list of values in the best fit line
abline_values_SD_UP = [slope * i + intercept for i in x]

x = [-0.1, 0.3]
y = [-0.1-0.068, 0.3-0.068]

# Find the slope and intercept of the best fit line
slope, intercept = np.polyfit(x, y, 1)
p1_SD_DN = np.polyfit(x, y, 1)

# Create a list of values in the best fit line
abline_values_SD_DN = [slope * i + intercept for i in x]

###########################################################

# Now for the real data
slope, intercept = np.polyfit(Hard_Var, Soft_Var, 1)
p1 = np.polyfit(Hard_Var, Soft_Var, 1)
#slope, intercept = np.polyfit(Hard_Var_Ext, Soft_Var_Ext, 1)

abline_values_data = [slope * i + intercept for i in Hard_Var]

################### Log Value Fit ######################
#slope, intercept = np.polyfit(Hard_Var_log, Soft_Var_log, 1)
#p1_log = np.polyfit(Hard_Var_log, Soft_Var_log, 1)
#slope, intercept = np.polyfit(Hard_Var_Ext, Soft_Var_Ext, 1)
#print ("log fit is", p1_log)

abline_values_data = [slope * i + intercept for i in Hard_Var_log]

plt.figure()
plt.plot(Hard_Var, Soft_Var, ls='none',marker='o',markersize='8',color='r')#, label='BLS1')       #lightseagreen
plt.errorbar(Hard_Var, Soft_Var, yerr=Soft_Var_Err, xerr=Hard_Var_Err, ls='none', color='r')       #orange


########### Comparing to Ponti #######################
# =============================================================================
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(left=2e-5, right=0.9)
# plt.ylim(2e-5, 0.9)
# =============================================================================

xlims = plt.xlim()
x.insert(0, xlims[0])
y.insert(0, np.polyval(p1, xlims[0]))
x.append(xlims[1])
y.append(np.polyval(p1, xlims[1]))
plt.plot(x, np.polyval(p1,x),  color='r', label='Best Fit')
plt.xlim(xlims)
plt.ylim(-0.1,0.3)

y.insert(0, np.polyval(p1_SD, xlims[0]))
y.append(np.polyval(p1_SD, xlims[1]))
plt.plot(x, np.polyval(p1_SD,x), ls='--',color='k', label='Slope = 1')

y.insert(0, np.polyval(p1_SD_UP, xlims[0]))
y.append(np.polyval(p1_SD_UP, xlims[1]))
plt.plot(x, np.polyval(p1_SD_UP,x), ls='--',color='lightgrey')

y.insert(0, np.polyval(p1_SD_DN, xlims[0]))
y.append(np.polyval(p1_SD_DN, xlims[1]))
plt.plot(x, np.polyval(p1_SD_DN,x), ls='--',color='lightgrey')

plt.legend(loc='upper left')
plt.title('%s Excess Variance (%s segments)' % (galaxy, segment_length))
plt.xlabel('Excess Variance (2-10 keV)')
plt.ylabel('Excess variance (0.5-2 keV)')

final_sources = Final_Mat.index.values.tolist()

for i, txt in enumerate(final_sources):
    plt.annotate(txt, (Hard_Var[i], Soft_Var[i]))

##############################################################################
##############################################################################

