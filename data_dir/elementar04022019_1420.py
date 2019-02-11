 # Quality Assurance and Control Algorithm for Automatic Weather System (AWS) 
# in the Hong Kong Observatory (HKO)
# Author: Li Lok Hang, Alan (R3b); 
# Project started on 15/1/2019

###############################################################################
# Version History:

# 04/02/2019: Simplified code structures;

# 01/02/2019: Implemented threshold multiplier;

# 31/01/2019: Implemented cold front regime;

# 30/01/2019: Implemented random forest algorithm Q5; Spatial test moved to Q6

# 29/01/2019: Fixed NaN problem; Implemented spatial interpolation; Verified SRT

# 25/01/2019: Implemented data frame to show QC statistics

# 24/01/2019: Cast flags into str; add skewnorm fitting method to step test ;
# RMSE and MAE

# 23/01/2019: Implemented persistency test for general variables using moving
# variance method

# 22/01/2019: Implemented SRT (ii), input file and log system;

# 21/01/2019: Implemented the time elapsed to run; SRT (i)

# 18/01/2019: Implemented range test using skewnorm fitting method

# 17/01/2019: Implemented persistency test, consistency test;

# 16/01/2019: Implemented spatial-range test; 
# allowed loading multi-station multi-element data via csv

# 15/01/2019: Implemented range test, step test; 
# defined class station and element
###############################################################################

# Import modules
from flask import Flask, flash, redirect, render_template, request, session, abort
import math
import numpy as np
import operator
import os
import pandas as pd
from scipy import spatial
from scipy.stats import skewnorm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import time
import datetime
import sys

###############################################################################
#-----------------------------------------------------------------------------#
# PART I: INITALIZATION
#-----------------------------------------------------------------------------#
###############################################################################
# Get the time now to start timer
now = datetime.datetime.now()
t0 = time.perf_counter()

# Set working directory
os.chdir("c:/Users/snr3/Desktop/alanli")

# Set log directory
log_dir = "C:/Users/snr3/Desktop/alanli/log_dir/"

# Set data directory
data_dir = "C:/Users/snr3/Desktop/alanli/data_dir/"

# Set output directory
output_dir = "C:/Users/snr3/Desktop/alanli/output_dir/"

###############################################################################
# Toggle printing
    # 1 = print QA flags and file error
    # 2 = print QA flags only
    # 3 = print final QA flags only (casted to str)
print_level = 1

# Toggle file input
    # 1 = .csv (for debug)
    # 2 = real-time data
file_input = 1

# Toggle log file output
    # 1 = use console
    # 2 = use log
log_out = 1

# Set infinite threshold for printing np.array
np.set_printoptions(threshold=1000)

###############################################################################
# Save log from console to a .log file
if log_out == 1:
    sys.stdout = sys.stderr
elif log_out == 2:
    log_path = log_dir + "log_" + now.strftime("%Y%m%d_%H%M") + ".log"
    sys.stdout = open(log_path, "w")

###############################################################################
# Set up server
app = Flask(__name__)
 
@app.route("/")
def index():
    return "Welcome to the test server!"
 
@app.route("/hello/<string:name>/")
def autoqc(name):
    return render_template('test.html', list_obj = elementlist[2][1].flaglist)
 
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=80)

###############################################################################

# Define a class "station"
class station:
    def __init__(self, sid, lon, lat, baro_height, anemo_height, ground_height):
        self.sid = sid; self.lon = lon; self.lat = lat
        self.baro_height = baro_height
        self.anemo_height = anemo_height
        self.ground_height = ground_height

# Read station data and store them into data frame
df = pd.read_csv(data_dir + 'AWS_STATION_INFO.csv', header=None, skipinitialspace = True)
df.columns = ['station_id', 'lat', 'lon', 'type', 'baro_height', 'anemo_height', 'ground_height']
sid = df['station_id']; slat = df['lat']; slon = df['lon']
sbh = df['baro_height']; sah = df['anemo_height']; sgh = df['ground_height']

# Define the station list
stationlist = []
for i in range(86):
    stationlist.append(station(sid[i],slat[i],slon[i],sbh[i],sah[i],sgh[i]))

###############################################################################
# Define threshold matrix based on monthly min max values
def tarray(mmin,mmax,a,b,c,d):
    tarr = np.array([0]*6)
    # a,b: first allowance
    # c,d: second allowance
    return tarr[mmin-d,mmin-b,mmin,mmax,mmax+a,mmax+c]

pthreshold_nodata = 3

###############################################################################
#-----------------------------------------------------------------------------#
# PART II: PERFORMING QUALITY ASSURANCE TESTS 
#-----------------------------------------------------------------------------#
###############################################################################
# Define the element name
elementname = ['1-min scalar mean wind direction','1-min scalar mean wind speed','1-min max wind speed (3 sec)','GPW(A) 10-min mean wind direction',
               '10-min mean wind speed','10-min maximum wind speed (3 sec)','60-min prevailing wind direction','60-min prevailing wind speed',
               '60-min max wind speed (3 sec)','1-min mean dry bulb temperature','1-min mean wet bulb temperature','Dew point from TT and TW',
               'Relative humidity from TT and TW','Maximum of TT since midnight (HKT)','Minimum of TT since midnight (HKT)','Time of maximum TT (nearest min)',
               'Time of minimum TT (nearest min)','1-min mean station level pressure','1-mean mean MSL pressure','1-min total rainfall',
               'GPR(A) 15-min total rainfall','60-min total rainfall','Total rainfall since mid-night','1-min mean tide level',
               '1-min mean sea temperature','1-min visibility']

# Initialize an element id
elementid = ['A1','B1','C1','D1','E1','F1','G1','H1','I1','J1','K1','L1','M1','N1','O1','P1','Q1','R1','S1','T1','U1','V1','W1','X1','Y1','Z1']

# Create an element list
n = len(elementname); m = len(stationlist)
elementlist = [[0] * n for i in range(m)]

###############################################################################
# Q1 - Q3: INTRA-ELEMENT TESTS
###############################################################################
# Define a class "element"
class element:
     def __init__(self, name, elementid, datelist, valuelist, flaglist, threshold_multiplier):
         self.name = name
         self.elementid = elementid
         self.datelist = datelist
         self.valuelist = valuelist
         self.flaglist = flaglist
         self.threshold_multiplier = threshold_multiplier
         
     # Q1: Perform range test
     def range_test(self, threshold):
         for k in range(len(self.datelist)):        
             # Based on weather conditions, use a scaled threshold:
             threshold_f = threshold
             threshold_f = [n*self.threshold_multiplier[k] for n in threshold_f]

             # Check for missing data or threshold
             if (np.isnan(self.valuelist[k]) | np.any(np.isnan(threshold))):
                 self.flaglist[k] += 0
                 continue
             
             # Check for impossible value (Only two thresholds)
             if(self.elementid in ('A1','D1','G1')):
                 if (self.valuelist[k] <= threshold_f[0] or threshold_f[5] <= self.valuelist[k]):
                     self.flaglist[k] += 4e9
                 else:
                     self.flaglist[k] += 1e9
 
             # Check based on thresholds and monthly min max              
             else:
                 if (self.valuelist[k] < threshold_f[0]):
                     self.flaglist[k] += 4e9
                     
                 elif (threshold_f[0] <= self.valuelist[k] < threshold_f[1]):
                     self.flaglist[k] += 3e9
                     
                 elif (threshold_f[1] <= self.valuelist[k] < threshold_f[2]):
                     self.flaglist[k] += 2e9
                     
                 elif (threshold_f[2] <= self.valuelist[k] < threshold_f[3]):
                     self.flaglist[k] += 1e9
                     
                 elif (threshold_f[3] <= self.valuelist[k] < threshold_f[4]):
                     self.flaglist[k] += 2e9
                     
                 elif (threshold_f[4] <= self.valuelist[k] < threshold_f[5]):
                     self.flaglist[k] += 3e9
    
                 elif (threshold_f[5] <= self.valuelist[k]):
                     self.flaglist[k] += 4e9
             
     # Q2: Perform step test 
     def step_test(self, threshold):
         for k in range(1,len(self.datelist)):
             # Based on weather conditions, use a scaled threshold:
             threshold_f = threshold
             threshold_f = [n*self.threshold_multiplier[k] for n in threshold_f]
             
             # Check for missing data or threshold
             if (np.isnan(self.valuelist[k]) | np.all(np.isnan(threshold_f))):
                 self.flaglist[k] += 0
                 continue
             else:
                 # Initialize the counter and local variable step
                 counter = 1
                 step = 0
                 while (counter <= 5 and k - counter >= 0):
                     # Define a step, if the last element does not exist,
                     # search for the latest one with time interval < 5 min
                     step = abs(self.valuelist[k] - self.valuelist[k - counter])
                     # If latest element exists, skip to step test
                     if(~np.isnan(step)):
                         break
                     counter += 1
                 if (counter == 5):
                     self.flaglist[k] += 0
                     continue

             # Perform step test
             if (step < threshold_f[0]):
                 self.flaglist[k] += 1e8

             elif (threshold_f[0] <= step < threshold_f[1]):
                 self.flaglist[k] += 2e8

             elif (threshold_f[1] <= step < threshold_f[2]):
                 self.flaglist[k] += 3e8
                 
             elif (threshold_f[2] <= step):
                 self.flaglist[k] += 4e8

     # Q3: Perform persistency test
     def persistency_test(self, threshold):
         # Check for elements
         if (self.elementid in ('A1','B1','C1','D1','E1','F1','G1','H1','I1','J1','K1','L1','M1')):    
             # Normalize for wind direction only
             # Normalization for other variables not needed (2018/01/25)
             if (self.elementid in ('A1','D1','G1')):
                 value = np.sin(np.pi/360*self.valuelist)
             else:
                 value = self.valuelist
             # Computing moving variance 
             rolling_variance = pd.Series(value).rolling(window=180).std()         
             
             # Define the threshold for low variances
             pindex = []
             var_threshold = [0.02,0.01,0.005]
             
             if (self.elementid in ('B1','C1','E1','F1','H1','I1')):
                 if(np.mean(value[-60:]) < 1):
                 # For calm winds, a lower threshold is adopted
                     pindex.append(np.where(np.isnan(rolling_variance)))
                     pindex.append(np.where(~np.isnan(rolling_variance)))
                     pindex.append(np.where((rolling_variance > var_threshold[1]) & (rolling_variance < var_threshold[0])))
                     pindex.append(np.where((rolling_variance > var_threshold[2]) & (rolling_variance < var_threshold[1])))
                     pindex.append(np.where(rolling_variance < var_threshold[2]))
                 else:
                 # For non-calm winds, a higher threshold is adopted
                     var_threshold[:] = [i * 2 for i in var_threshold]
                     pindex.append(np.where(np.isnan(rolling_variance)))
                     pindex.append(np.where(~np.isnan(rolling_variance)))
                     pindex.append(np.where((rolling_variance > var_threshold[1]) & (rolling_variance < var_threshold[0])))
                     pindex.append(np.where((rolling_variance > var_threshold[2]) & (rolling_variance < var_threshold[1])))
                     pindex.append(np.where(rolling_variance < var_threshold[2]))
             else:
                 # For other non-wind variables:
                 pindex.append(np.where(np.isnan(rolling_variance)))
                 pindex.append(np.where(~np.isnan(rolling_variance)))
                 pindex.append(np.where((rolling_variance > var_threshold[1]) & (rolling_variance < var_threshold[0])))
                 pindex.append(np.where((rolling_variance > var_threshold[2]) & (rolling_variance < var_threshold[1])))
                 pindex.append(np.where(rolling_variance < var_threshold[2]))
                
             # Flag time steps for low variances; use np.add for casting datatypes
             self.flaglist[pindex[0]] = np.add(self.flaglist[pindex[0]], 0, casting = "unsafe")
             self.flaglist[pindex[1]] = np.add(self.flaglist[pindex[1]], 1e7, casting = "unsafe")
             self.flaglist[pindex[2]] = np.add(self.flaglist[pindex[2]], 2e7, casting = "unsafe")
             self.flaglist[pindex[3]] = np.add(self.flaglist[pindex[3]], 3e7, casting = "unsafe")
             self.flaglist[pindex[4]] = np.add(self.flaglist[pindex[4]], 4e7, casting = "unsafe")

         # Check if the element is X1
         elif (self.elementid in ('X1')):
             X_0 = None
             X_1 = self.valuelist
             cnt = 1
             nodatat = 0
             
             for k in range(len(self.datelist)):
                 #Check for missing threshold
                 if (np.isnan(threshold)):
                     self.flaglist[k] += 0
                     cnt = 1
                     continue
                 
                 #Check for missing data
                 if (np.isnan(self.valuelist[k])):
                     nodatat += 1
                     if (nodatat == pthreshold_nodata):
                         self.flaglist[k] += 0
                         cnt = 1
                     else:
                         continue
                     
                 #Check if X0 is present
                 if (X_0 is None):
                     X_0 = X_1[k]
                     self.flaglist[k] += 0
                     cnt = 1
                     
                 #Carry out persistency test                    
                 if (cnt <= 180):    
                     if (X_1 == X_0):
                         cnt += 1
                         self.flaglist[k] += 0
                     else:
                         cnt = 1
                         self.flaglist[k] += 0
                 elif (cnt > 180 and cnt < 210):
                     if (abs(X_1 - X_0) <= 5):
                         cnt += 1
                     else:
                         X_0 = X_1[k]
                         self.flaglist[k] += 0
                 elif (cnt > 210):
                     self.flaglist[k] += 2e7

     # Evaluate the composite flag based on other flags
     def get_composite_flag(self):
         for k in range(len(self.datelist)):
             # Reset flag
             self.flaglist[k] -= (self.flaglist[k] // 1e3 % 1e1) * 1e3
             
             # Cast flag to string
             flag = str(self.flaglist[k]).zfill(10)
             
             # Only all bypassed gives a bypassed flag
             if ('0' in flag[0] and '0' in flag[1] and '0' in flag[2] and '0' in flag[3] and '0' in flag[4] and '0' in flag[5]):
                 self.flaglist[k] += 0
         
            # Any error flag gives a composite error flag
             elif ('4' in flag):
                 self.flaglist[k] += 4e3

            # Any highly suspcious flag gives a composite highly suspcious flag                 
             elif ('3' in flag):
                 self.flaglist[k] += 3e3                 

            # Any suspcious flag gives a composite suspcious flag                 
             elif ('2' in flag):
                 self.flaglist[k] += 2e3
                 
            # In other cases, the element passed the composite test             
             else:
                 self.flaglist[k] += 1e3
    
     # Unflag elements
     def unflag(self,q,start,end):
         q = 10 - q
         self.flaglist[start:end] -= (self.flaglist[start:end] // (10 ** q) % 10) * (10 ** q)

     # Change thresholds
     def change_threshold(self,k,multiplier):
         self.threshold_multiplier[k] = max(self.threshold_multiplier[k],multiplier)

###############################################################################
# Q4: INTER-ELEMENT TESTS
###############################################################################  
# Consistency test
# Consistency among J1, N1, O1, and K1
def consistency_J1N1O1K1(i):
    # Initialize the variables
    J1 = elementlist[i][9]
    K1 = elementlist[i][10]
    L1 = elementlist[i][11]
    M1 = elementlist[i][12]
    N1 = elementlist[i][13]
    O1 = elementlist[i][14]

    for k in range(len(J1.datelist)):
        if (np.isnan(J1.valuelist[k]) or np.isnan(N1.valuelist[k])):
            N1.flaglist[k] += 0
        elif (N1.valuelist[k] <= J1.valuelist[k]):
            N1.flaglist[k] += 4e6
        else:
            N1.flaglist[k] += 1e6
            if (np.isnan(J1.valuelist[k]) or np.isnan(O1.valuelist[k])):
                O1.flaglist[k] += 0
            elif (O1.valuelist[k] > J1.valuelist[k]):
                O1.flaglist[k] += 4e6
            else:
                O1.flaglist[k] += 1e6
                
        if (np.isnan(J1) or np.isnan(K1)):
            J1.flaglist[k] += 0
            K1.flaglist[k] += 0
        
        # Check for wet-bulb temperature sensor problem
        elif (J1.valuelist[k] - K1.valuelist[k] >= 0):
            if (J1.valuelist[k] - K1.valuelist[k] < 14):
                J1.flaglist[k] += 1e6; K1.flaglist[k] += 1e6
                L1.flaglist[k] += 1e6; M1.flaglist[k] += 1e6
            elif (J1.valuelist[k] - K1.valuelist[k] < 16):
                J1.flaglist[k] += 2e6; K1.flaglist[k] += 2e6
                L1.flaglist[k] += 2e6; M1.flaglist[k] += 2e6
            elif (J1.valuelist[k] - K1.valuelist[k] < 20):
                J1.flaglist[k] += 3e6; K1.flaglist[k] += 3e6
                L1.flaglist[k] += 3e6; M1.flaglist[k] += 3e6
            else:
                J1.flaglist[k] += 3e6; K1.flaglist[k] += 4e6
                L1.flaglist[k] += 4e6; M1.flaglist[k] += 4e6
                
        # Check for dry-bulb temperature sensor problem
        else:
            if (-0.2 < J1.valuelist[k] - K1.valuelist[k] <= -0.1):
                J1.flaglist[k] += 2e6; K1.flaglist[k] += 2e6
                L1.flaglist[k] += 2e6; M1.flaglist[k] += 2e6
            elif (-0.5 < J1.valuelist[k] - K1.valuelist[k] <= -0.2):
                J1.flaglist[k] += 3e6; K1.flaglist[k] += 3e6
                L1.flaglist[k] += 3e6; M1.flaglist[k] += 3e6
            else:
                J1.flaglist[k] += 4e6; K1.flaglist[k] += 3e6
                L1.flaglist[k] += 4e6; M1.flaglist[k] += 4e6

# Consistency test for B1 and C1
def consistency_A1B1C1(i):
    # Initialize the variables
    A1 = elementlist[i][0]
    B1 = elementlist[i][1]
    C1 = elementlist[i][2]
    
    for k in range(len(B1.datelist)):
        # Check consistency between A1 and B1
        if (np.isnan(A1.valuelist[k]) and np.isnan(B1.valuelist[k])):
            A1.flaglist[k] += 0; B1.flaglist[k] += 0
        elif ((A1.valuelist[k] == 0 and B1.valuelist[k] != 0) or (A1.valuelist[k] != 0 and B1.valuelist[k] == 0)):
            A1.flaglist[k] += 4e6; B1.flaglist[k] += 4e6
            
        # Check consistency between B1 and C1
        if (np.isnan(B1.valuelist[k]) and np.isnan(C1.valuelist[k])):
            B1.flaglist[k] += 0
            C1.flaglist[k] += 0
        elif(B1.valuelist[k] == 0 and C1.valuelist[k] == 0):
            B1.flaglist[k] += 1e6
            C1.flaglist[k] += 1e6
        else:
            if (B1.valuelist[k] != 0 and C1.valuelist[k] != 0):
                if(C1.valuelist[k] >= B1.valuelist[k]):
                    if (C1.valuelist[k] == B1.valuelist[k]):
                        if (C1.valuelist[k] > 85):
                            B1.flaglist[k] += 2e6; C1.flaglist[k] += 2e6
                        else:
                            B1.flaglist[k] += 1e6; C1.flaglist[k] += 1e6
                    else:
                        B1.flaglist[k] += 1e6; C1.flaglist[k] += 1e6
                else:
                    B1.flaglist[k] += 4e6; C1.flaglist[k] += 4e6
            else:
                if (B1.valuelist[k] == 0):
                    if (C1.valuelist[k] > 3):
                        B1.flaglist[k] += 4e6; C1.flaglist[k] += 4e6
                    else:
                        B1.flaglist[k] += 1e6; C1.flaglist[k] += 1e6
                else:
                    B1.flaglist[k] += 4e6; C1.flaglist[k] += 4e6

# Consistency test for C1, E1, and H1
def consistency_C1E1H1(i):
    # Initialize the variables
    B1 = elementlist[i][1]
    C1 = elementlist[i][2]
    E1 = elementlist[i][4]
    F1 = elementlist[i][5]
    H1 = elementlist[i][7]
    I1 = elementlist[i][8]
    
    for k in range(len(C1.datelist)):
        if (~np.isnan(C1.valuelist[k]) and (C1.flaglist[k] // 1e7 % 1e1 == 1) and
            (C1.valuelist[k] > 650) and 
            ((abs(C1.valuelist[k] - E1.valuelist[k]) > 400) or (abs(C1.valuelist[k] - H1.valuelist[k]) > 400))):
            #Overwrite Q4
            B1.flaglist[k] += (6 - B1.flaglist[k] // 1e6 % 10) * 1e6
            C1.flaglist[k] += (6 - C1.flaglist[k] // 1e6 % 10) * 1e6
            E1.flaglist[k] += 6e6; F1.flaglist[k] += 6e6
            H1.flaglist[k] += 6e6; I1.flaglist[k] += 6e6
            #Overwrite Q7
            B1.flaglist[k] += (6 - B1.flaglist[k] // 1e3 % 10) * 1e3
            C1.flaglist[k] += (6 - C1.flaglist[k] // 1e3 % 10) * 1e3
            E1.flaglist[k] += 6e3; F1.flaglist[k] += 6e3
            H1.flaglist[k] += 6e3; I1.flaglist[k] += 6e3
            
# Consistency test for J1, K1, L1, and M1 (to be performed after Q7 generation)
def consistency_J1K1L1M1(i):
    # Initialize the variables
    J1 = elementlist[i][9]
    K1 = elementlist[i][10]
    L1 = elementlist[i][11]
    M1 = elementlist[i][12]
    
    for k in range(len(J1.datelist)):
        if (~np.isnan(J1.valuelist[k]) and ~np.isnan(K1.valuelist[k])):
            #Overwrite Q4 of L1 and M1 according to Q7 of J1 or K1
            if ((J1.flaglist[k] // 1e3 % 1e1 == 3) or (K1.flaglist[k] // 1e3 % 1e6 == 3)):
                L1.flaglist[k] -= (L1.flaglist[k] // 1e6 % 10) * 1e6 + 4e6
                M1.flaglist[k] -= (L1.flaglist[k] // 1e6 % 10) * 1e6 + 4e6
            elif ((J1.flaglist[k] // 1e3 % 1e1 == 4) or (K1.flaglist[k] // 1e3 % 1e6 == 4)):
                L1.flaglist[k] -= (L1.flaglist[k] // 1e6 % 10) * 1e6 + 3e6
                M1.flaglist[k] -= (L1.flaglist[k] // 1e6 % 10) * 1e6 + 3e6
            elif ((J1.flaglist[k] // 1e3 % 1e1 == 2) or (K1.flaglist[k] // 1e3 % 1e6 == 2)):
                L1.flaglist[k] -= (L1.flaglist[k] // 1e6 % 10) * 1e6 + 2e6
                M1.flaglist[k] -= (L1.flaglist[k] // 1e6 % 10) * 1e6 + 2e6
            else:
                #Overwrite Q4 of J1, K1, and L1 if Q7 of M1 is 3
                if (M1.flaglist[k] // 1e3 % 1e1 == 3):
                    J1.flaglist[k] -= (L1.flaglist[k] // 1e6 % 10) * 1e6 + 2e6
                    K1.flaglist[k] -= (L1.flaglist[k] // 1e6 % 10) * 1e6 + 4e6
                    L1.flaglist[k] -= (L1.flaglist[k] // 1e6 % 10) * 1e6 + 4e6
        else:
            if (np.isnan(K1.valuelist[k]) and ~np.isnan(M1.valuelist[k])):
                if (M1.flaglist[k] // 1e3 % 1e1 == 3):
                    L1.flaglist[k] -= (L1.flaglist[k] // 1e6 % 10) * 1e6 + 4e6

###############################################################################                 
# Main driver code for Q1-Q4          
# THIS PART composes of 5 modules:
    # (1): Find the longest length of csv files
    # (2): Normalize the length of input data, save to elementlist
    # (3): Perform threshold multiplications
    # (4): Perform intra-element tests (Q1 - Q3)

# Distance formula: evaluate the euclidean distance between two stations, assume that the difference in height is negligible.
def ComputeDistance(station1,station2):
    xy1 = [station1.lon,station1.lat]
    xy2 = [station2.lon,station2.lat]
    distance = spatial.distance.euclidean(xy1, xy2)
    return distance

# Return the nearest stations within a range of distance
def get_nearest_stations(target_station, min_distance, max_distance):
    distance_array=[]
    for i in range(len(stationlist)):
        if (stationlist[i] != target_station):
            D = ComputeDistance(stationlist[i],target_station)
            if (min_distance <= D <= max_distance):
               distance_array.append((stationlist[i], D))
    # Sort the distance array according to distance
    distance_array.sort(key=operator.itemgetter(1))
    return distance_array
def find_zero(array_in):
# Search for the starting and ending index of 0 elements in an array
     iszero = np.concatenate(([0], np.equal(array_in, 0), [0]))
     absdiff = np.abs(np.diff(iszero))
     ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
     return ranges

# Functions to evaluate QC performances:
# Evaluate root mean square difference
def RMSD(observed,estimated):
    counter = 0
    if(len(observed) == len(estimated)):
        for i in range(len(observed)):
            counter += (observed[i] - estimated[i]) ** 2
        return (counter / len(observed)) ** (1/2)
    else:
        print("Error: The length of observed data and estimated data should be the same!")

# Evaluate mean absolute difference
def MAD(observed,estimated):
    counter = 0
    if(len(observed) == len(estimated)):
        for i in range(len(observed)):
            counter += abs(observed[i] - estimated[i])
        return (counter / len(observed))
    else:
        print("Error: The length of observed data and estimated data should be the same!")

# (1): THIS PART evaluates the longest length of csv files

# Initialize a counter for longest length of list                 

longestdllen = 0; longestdl = np.array([])

# Initialize constants
mindate = 999999999999; maxdate = 0

# Loop through each element in each station
for i in range(len(stationlist)):
    for j in range(len(elementlist[i])):
        # For inputting CSV files
        if file_input == 1:
            csvfile = stationlist[i].sid + "_" + elementid[j] + ".csv"
            try:
                with open(data_dir + csvfile, newline='') as csvfile:
                    df = pd.read_csv(csvfile, header=None, skipinitialspace = True)
                    df.columns = ['Station', 'Element', 'Date', 'Value', 'QA Flag']
                    # Extract the length of the longest list
                    if (len(df['Date']) > longestdllen): 
                        longestdllen = len(df['Date'])
                        longestdl = np.array(df['Date'])
                    # Extract the minimum date
                    if (min(df['Date']) < mindate):
                        mindate = min(df['Date'])
                    # Extract the maximum date
                    if (max(df['Date']) > maxdate): 
                        maxdate = max(df['Date'])
            except IOError:
                continue

# (2): THIS PART puts elmeents into elementlist
for i in range(len(stationlist)):
    for j in range(len(elementlist[i])):
        # For inputting CSV files
        if file_input == 1:
            csvfile = stationlist[i].sid + "_" + elementid[j] + ".csv"
            if print_level <= 1:
                print(csvfile)
            try:
                with open(data_dir + csvfile, newline='') as csvfile:
                    df = pd.read_csv(csvfile, header=None, skipinitialspace = True)
                    df.columns = ['Station', 'Element', 'Date', 'Value', 'QA Flag']
            except IOError:
                if print_level <= 1:
                    print("File NOT exist!")
                continue
            
            # Initialize the data, use np.array for better indices manipulation
            dl = df['Date'] #initial date list
            vl = df['Value'] #initial value list
            datelist = np.array(dl); valuelist = np.array(vl)
            flaglist = np.array([0]*len(datelist)).astype('int64') #initial flag list
            
            # Initially, all threshold are unmultiplied
            threshold_multiplier = np.array([1]*len(datelist)).astype('int64')
            
            # Append NaN to incomplete lists
            diff = longestdllen - len(datelist)
            # The top is incomplete
            if(min(dl) != mindate and max(dl) == maxdate):
                datelist = np.append([np.nan] * diff, datelist) #extended date list
                valuelist = np.append([np.nan] * diff, valuelist) #extended value list
                flaglist = np.append([0] * diff, flaglist).astype('int64') #extended flag list
                threshold_multiplier = np.append([1] * diff, threshold_multiplier).astype('int64') #extended threshold multiplier list
            
            # The bottom is incomplete
            elif(min(dl) == mindate and max(dl) != maxdate):
                datelist = np.append(datelist, [np.nan] * diff) #extended date list
                valuelist = np.append(valuelist, [np.nan] * diff) #extended value list
                flaglist = np.append(flaglist, [0] * diff).astype('int64') #extended flag list
                threshold_multiplier = np.append(threshold_multiplier, [1] * diff).astype('int64') #extended threshold multiplier list

            # Both the top and bottom is incomplete
            elif(min(dl) != mindate and max(dl) != maxdate):
                topidx = np.where(longestdl == min(dl))[0] #Get starting index in longest data list
                bottomidx = np.where(longestdl == max(dl))[0] #Get ending index in longest data list
                
                datelist = np.append([np.nan] * (topidx + 1), datelist) #top extended date list
                valuelist = np.append([np.nan] * (topidx + 1), valuelist) #top extended value list
                flaglist = np.append([0] * (topidx + 1), flaglist).astype('int64') #top extended flag list
                threshold_multiplier = np.append([1] * (topidx + 1), threshold_multiplier).astype('int64') #top threshold multiplier list
                
                datelist = np.append(datelist, [np.nan] * (longestdllen - 1 - bottomidx)) #bottom extended date list
                valuelist = np.append(valuelist, [np.nan] * (longestdllen - 1 - bottomidx)) #bottom extended value list
                flaglist = np.append(flaglist, [0] * (longestdllen - 1 - bottomidx)).astype('int64') #bottom extended flag list
                threshold_multiplier = np.append(threshold_multiplier, [1] * (longestdllen - 1 - bottomidx)).astype('int64') #bottom threshold multiplier list

            # Put element into the element list
            elementlist[i][j] = element(elementname[j], elementid[j], datelist, valuelist, flaglist, threshold_multiplier)
        # For inputting real-time data
        elif file_input == 2:
            continue

elementlist = np.asarray(elementlist)

# (3): Thresholds flexibility
# In the following code, when certain condition is matched, threshold will be altered.
for i in range(len(stationlist)):
    # Get the near stations around station i within 0.05 degree;
    near_stations = np.asarray(get_nearest_stations(stationlist[i],0,0.05))
    
    # Skip when there is no near stations
    if len(near_stations) == 0:
        continue
    else:
        near_stations = near_stations[:,0]
    
    # Obtain the index of top near stations
    i_near = np.where(np.in1d(stationlist,near_stations))

    for k in range(longestdllen):
        # Cold fronts        
        elementlist_near = elementlist[i_near][:,10]
        elementlist_near = elementlist_near[np.where(elementlist_near != 0)]
        
        # Time step = 30 minutes
        ts = 30
    
        for m in range(len(elementlist_near)):
            # Condition: wet bulb temperature drops: change in 1-min wet bulb temperature in 30 mins > 1C
            if (10 < (np.mean(elementlist_near[m].valuelist[(k-5):k]) - np.mean(elementlist_near[m].valuelist[k-ts:(k-ts+5)]))/ts):
                # If any near station reaches condition, threshold of target station is multiplied by 1.2
                if elementlist[i][0] != 0:
                    elementlist[i][0].change_threshold(k,1.2)
                if elementlist[i][1] != 0:
                    elementlist[i][1].change_threshold(k,1.2)
                if elementlist[i][2] != 0:
                    elementlist[i][2].change_threshold(k,1.2)

        # Tropical Cyclones
        elementlist_near = elementlist[i_near][:,18]
        elementlist_near = elementlist_near[np.where(elementlist_near != 0)]
        for m in range(len(elementlist_near)):
            # Condition: cyclone approaches: sea level pressure < 1000 hpa
            if (9800 < elementlist_near[m].valuelist[k] <= 10000):
                # If any near station reaches condition, threshold of target station is multiplied by 1.2
                if elementlist[i][0] != 0:
                    elementlist[i][0].change_threshold(k,1.2)
                if elementlist[i][1] != 0:
                    elementlist[i][1].change_threshold(k,1.2)
                if elementlist[i][2] != 0:
                    elementlist[i][2].change_threshold(k,1.2)

            # Condition: very near intense cyclone: sea level pressure < 980 hpa
            elif (9000 < elementlist[i_near[m]][19].valuelist[k] <= 9800):
                # If any near station reaches condition, threshold of target station is multiplied by 1.4
                if elementlist[i][0] != 0:
                    elementlist[i][0].change_threshold(k,1.4)
                if elementlist[i][1] != 0:
                    elementlist[i][1].change_threshold(k,1.4)
                if elementlist[i][2] != 0:
                    elementlist[i][2].change_threshold(k,1.4)
                
        # Rainstorms
        elementlist_near = elementlist[i_near][:,19]
        elementlist_near = elementlist_near[np.where(elementlist_near != 0)]
        for m in range(len(elementlist_near)):
            # Condition: heavy rain: 1-min total rainfall > 0.5mm (30mm/hr)
            if (5 < elementlist_near[m].valuelist[k] <= 11.6):
                # If any near station reaches condition, threshold of target station is multiplied by 1.2
                if elementlist[i][0] != 0:
                    elementlist[i][0].change_threshold(k,1.2)
                if elementlist[i][1] != 0:
                    elementlist[i][1].change_threshold(k,1.2)
                if elementlist[i][2] != 0:
                    elementlist[i][2].change_threshold(k,1.2)

            # Condition: very heavy rain: 1-min total rainfall > 1.16mm (70mm/hr)
            elif (11.6 < elementlist[i_near[m]][19].valuelist[k] < 30):
                # If any near station reaches condition, threshold of target station is multiplied by 1.4
                if elementlist[i][0] != 0:
                    elementlist[i][0].change_threshold(k,1.4)
                if elementlist[i][1] != 0:
                    elementlist[i][1].change_threshold(k,1.4)
                if elementlist[i][2] != 0:
                    elementlist[i][2].change_threshold(k,1.4)
                
# (4): THIS PART performs INTRA-ELEMENT TESTS
for i in range(len(stationlist)):
    for j in range(len(elementlist[i])):
        if elementlist[i][j] != 0:
            # Define the threshold level by p value
            pvalue1 = 2.7e-3 # 3-sigma rule
            pvalue2 = 4.7e-4 # 3.5-sigma rule
            pvalue3 = 6.4e-5 # 4-sigma rule
            
            # Select elementlist
            v = np.asarray(elementlist[i][j].valuelist)
            v = v[~np.isnan(v)] 
            
            # Define range test threshold array
            rthreshold = [np.nan]*6
            
            # Define range test thresholds for different elements
            # Elements for simple range test
            if(elementlist[i][j].elementid in ('A1','D1','G1')):
                rthreshold[0] = 0; rthreshold[5] = 360
            # Elements for hybrid range-skewnorm fitting test
            elif(elementlist[i][j].elementid in ('B1','C1','E1','F1','H1','I1','T1','U1','V1','W1')):        
                # Define the acceptable range
                rthreshold[0] = rthreshold[1] = rthreshold[2] = 0
                rthreshold[3] = skewnorm.ppf(1-pvalue1,*skewnorm.fit(v))
                rthreshold[4] = skewnorm.ppf(1-pvalue2,*skewnorm.fit(v))
                rthreshold[5] = skewnorm.ppf(1-pvalue3,*skewnorm.fit(v))
            # Elements for doubled-sided skewnorm fitting test
            elif(elementlist[i][j].elementid in ('J1','K1','L1','R1','S1','X1','Y1','Z1')):        
                # Define the acceptable range
                rthreshold[0] = skewnorm.ppf(pvalue3,*skewnorm.fit(v))
                rthreshold[1] = skewnorm.ppf(pvalue2,*skewnorm.fit(v))
                rthreshold[2] = skewnorm.ppf(pvalue1,*skewnorm.fit(v))
                rthreshold[3] = skewnorm.ppf(1-pvalue1,*skewnorm.fit(v))
                rthreshold[4] = skewnorm.ppf(1-pvalue2,*skewnorm.fit(v))
                rthreshold[5] = skewnorm.ppf(1-pvalue3,*skewnorm.fit(v))     
            # Relative humidity
            elif(elementlist[i][j].elementid in ('M1')):    
                # Define the acceptable range
                rthreshold[0] = 0
                rthreshold[1] = max(skewnorm.ppf(pvalue2,*skewnorm.fit(v)),0)
                rthreshold[2] = max(skewnorm.ppf(pvalue1,*skewnorm.fit(v)),0)
                rthreshold[3] = min(skewnorm.ppf(1-pvalue1,*skewnorm.fit(v)),100)
                rthreshold[4] = min(skewnorm.ppf(1-pvalue2,*skewnorm.fit(v)),100)
                rthreshold[5] = 100    
            
            # Define step test threshold array 
            # Use climatological data (2018/01/25)
            sthreshold = [100,150,200]
            
            # PERFORM range test
            elementlist[i][j].range_test(rthreshold)
            
            # PERFORM step test
            elementlist[i][j].step_test(sthreshold)
            
            # PERFORM persistency test
            elementlist[i][j].persistency_test(pthreshold_nodata)
    
            # Obtaining temporary composite flag for use in Q4
            elementlist[i][j].get_composite_flag()
            
            # Print out results
            print('Station:', stationlist[i].sid, 'INTRA-ELEMENT TEST results for element', elementid[j], elementname[j])
            if print_level <= 2:
                print(elementlist[i][j].flaglist)

###############################################################################  
t1 = time.perf_counter() - t0
print("Time elapsed: ", round(t1 , 2) , "s")
print('NOW PROCEEDING TO INTER-ELEMENT TEST')

###############################################################################   
# Main driver code for Q3b, Q4, and Q6

for i in range(len(stationlist)):
    print(stationlist[i].sid)
     
    # Unflag for Q3
    # Reset Q3 for wind direction if: 
        # (1) flag exists
        # (2) wind speed = 0 (calm) for 15 minutes
    if(elementlist[i][1] != 0):
        range_of_zero = [(a,b) for (a,b) in find_zero(elementlist[i][1].valuelist) if (b - a) > 15]
        for (a,b) in range_of_zero:
            if elementlist[i][0] != 0:
                elementlist[i][0].unflag(3,a,b)
            if elementlist[i][3] != 0:
                elementlist[i][3].unflag(3,a,b)
            if elementlist[i][6] != 0:
                elementlist[i][6].unflag(3,a,b)
            
    # Reset Q3 for RH if:
        # (1) flag exists
        # (2) the correlation between two rolling variance series of J1 and M1 > 0.5
    if(elementlist[i][9] != 0 and elementlist[i][12] != 0):
        # str_flag = [str(elementlist[i][12].flaglist[k]).zfill(10)[2] for k in range(len(elementlist[i][12].datelist))]
        # flagged_index = np.where(np.asarray(str_flag) == '0')
        
        rolling_variance_J1 = pd.Series(elementlist[i][9].valuelist).rolling(window=60).std()
        rolling_variance_M1 = pd.Series(elementlist[i][12].valuelist).rolling(window=60).std()  
        
        mask = ~np.isnan(rolling_variance_J1) & ~np.isnan(rolling_variance_M1)
        
        model = sm.OLS(rolling_variance_M1[mask], sm.add_constant(rolling_variance_J1[mask]))
        results = model.fit()
        r_squared = results.rsquared
        
        if r_squared > 0.5:
            elementlist[i][12].flaglist -= (elementlist[i][12].flaglist // 1e7 % 10) * 1e7
        
    # PERFORM consistency test if element exists
    if(elementlist[i][0] != 0 and elementlist[i][1] != 0 and elementlist[i][2] != 0):
        consistency_A1B1C1(i)   

    if(elementlist[i][1] != 0 and elementlist[i][2] != 0 and elementlist[i][4] != 0 and
       elementlist[i][5] != 0 and elementlist[i][7] != 0 and elementlist[i][8] != 0):
        consistency_C1E1H1(i) 
    
    if(elementlist[i][9] != 0 and elementlist[i][10] != 0 and elementlist[i][11] != 0 and 
       elementlist[i][12] != 0 and elementlist[i][13] != 0 and elementlist[i][14] != 0):
        consistency_J1N1O1K1(i)
    
    if(elementlist[i][9] != 0 and elementlist[i][10] != 0 and elementlist[i][11] != 0 and 
       elementlist[i][12] != 0):
        consistency_J1K1L1M1(i) 

# THIS part performs the random forest algorithm
for i in range(len(stationlist)):
    # Initialize a data frame of size and frequency equals to the time series
    start = datetime.datetime.strptime(str(mindate), '%Y%m%d%H%M')
    end = datetime.datetime.strptime(str(maxdate), '%Y%m%d%H%M') + datetime.timedelta(minutes=1)
    timeseries = np.arange(start, end, datetime.timedelta(minutes=1)).astype(datetime.datetime)
    df = pd.DataFrame(timeseries.tolist(),columns=['Date'])
    
    # Put data into the data frame
    for j in range(len(elementlist[i])):
        if(elementlist[i][j] != 0):
            df[stationlist[i].sid + "_" + elementlist[i][j].elementid] = pd.DataFrame(elementlist[i][j].valuelist)

    # Dataframe organization
    df['Date'] = [df['Date'][i].strftime('%Y%m%d%H%M') for i in range(len(df['Date']))]

    if len(df.columns) <= 2:
        print('Error: RF bypassed due to insufficient features. Station:', stationlist[i])
        continue

    for j in range(len(elementlist[i])):
        if(elementlist[i][j] != 0):
            # Element we want to predict:
            ele = stationlist[i].sid + "_" + elementlist[i][j].elementid
            
            # Skip RF if NaN data > 20% 
            if(df[ele].count() / df['Date'].count() < 0.8):
                print('Error: RF bypassed due to NaN; Element:', ele)
                continue 
            
            # Remove the values we want to predict as labels
            labels = np.array(df[ele])
            df = df.drop(ele, axis = 1)
            df.fillna(df.mean(), inplace=True)
            
            # Saving feature names for later use
            df_list = list(df.columns)
            
            # Convert the data frame to a numpy array
            dm = np.array(df)
            
            # Split the data into training and testing sets
            train_features, test_features, train_labels, test_labels = train_test_split(dm, labels, test_size = 0.2, random_state = 42)
            
            # Instantiate model with 2000 decision trees
            rf = RandomForestRegressor(n_estimators = 2000, max_depth = 5, min_samples_leaf = 5)
            
            # Train the model on training data
            rf.fit(train_features, train_labels);
            
            # Make predictions based on RF model 
            predictions = rf.predict(test_features)
            
            # Calculate the absolute errors
            errors = abs(predictions - test_labels)
            
            # Accuracy of the random forest model:
            accuracy = 1 - errors / test_labels
            
            # Manipulating dates for true set data
            dates = df['Date']
            dates = [datetime.datetime.strptime(str(date), '%Y%m%d%H%M') for date in dates]
            true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})
            
            # Manipulating dates for prediction set data
            test_dates = test_features[:,0]
            test_dates = [datetime.datetime.strptime(str(date), '%Y%m%d%H%M') for date in test_dates]
            predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions})
            
            # THIS part plots the prediction and true data out
# =============================================================================
# plt.figure(figsize=(20,10))
# # Plot the actual values
# plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')
# # Plot the predicted values
# plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
# plt.xticks(rotation = '60'); 
# plt.legend()
# # Graph labels
# plt.xlabel('Date'); plt.ylabel('1-min scalar mean wind speed)'); plt.title('Actual and Predicted Values');
# =============================================================================
            
            # Resample and interpolate to yield a series with identical length of true data
            start = true_data['date'][0]
            end = true_data['date'][len(true_data)-1]
            
            # Set index to date
            true_data = true_data.set_index('date')
            sort_series = predictions_data.sort_values(by='date').set_index('date')
            
            # Evaluate new index
            new_index = pd.DatetimeIndex(start=start, end=end, freq='1min')
            predictions_data_r = sort_series.resample('1Min').interpolate(method='linear').reindex(new_index)
            
            # Evaluate differences and remove NaNs
            diff = abs(predictions_data_r['prediction'] - true_data['actual'])
            diff = np.asarray(diff)
            diffv = np.asarray(diff)[~np.isnan(diff)]

            # Evaluate the threshold based on skewnorm method
            rfthreshold = [0]*3           
            rfthreshold[0] = skewnorm.ppf(1-pvalue1,*skewnorm.fit(diffv))
            rfthreshold[1] = skewnorm.ppf(1-pvalue2,*skewnorm.fit(diffv))
            rfthreshold[2] = skewnorm.ppf(1-pvalue3,*skewnorm.fit(diffv))
            
            for k in range(len(elementlist[i][j].datelist)):            
                # Bypass QC for NaN data           
                if(np.isnan(diff[k])):
                    elementlist[i][j].flaglist[k] += 0
                
                # Check whether IDW exceeds threshold     
                elif (diff[k] <= rfthreshold[0]):
                    elementlist[i][j].flaglist[k] += 1e4
    
                elif (rfthreshold[0] <= diff[k] < rfthreshold[1]):
                    elementlist[i][j].flaglist[k] += 2e4
    
                elif (rfthreshold[1] <= diff[k] < rfthreshold[2]):
                    elementlist[i][j].flaglist[k] += 3e4
                     
                elif (rfthreshold[2] <= diff[k]):
                    elementlist[i][j].flaglist[k] += 4e4    

for i in range(len(stationlist)):
    # Print out results
    for j in range(len(elementlist[i])):
        if(elementlist[i][j] == 0):
            continue
        print('Station:', stationlist[i].sid, 'CONSISTENCY TEST RESULTS for element', elementid[j], elementname[j])
        if print_level <= 2:
            print(elementlist[i][j].flaglist)
        
###############################################################################
# Q5: INTER-STATION TESTS
###############################################################################  
t1 = time.perf_counter() - t0
print("Time elapsed: ", round(t1 , 2) , "s")
print('NOW PROCEEDING TO SPATIAL TEST')
###############################################################################
# Spatial test

# Return the inverse distance weighted average of a station estimated by other stations
def IDW(station, j, k):    
    # Initialize numerator (n) and denominator (d) of weighted average
    n = []; d = []
    for i in range(len(stationlist)-1):
        # Check for all stations except the selected one
        if stationlist[i] != station:
        # Check for availability of data
            if(elementlist[i][j] == 0):
                continue
        # Discard unsuitable stations flagged by previous tests
            flag = str(elementlist[i][j].flaglist[k])
            if("2" in flag or "3" in flag or "4" in flag):
                continue
            # Correct for station height (Use potential temperature for TT)
# =============================================================================
#             if(elementlist[i][j].elementid == 'J1' and elementlist[i][17].valuelist[k] is not None):
#                 Rd, Cp = 287, 1004
#                 pressure = elementlist[i][17].valuelist[k]
#                 value = elementlist[i][j].valuelist[k] * (pressure / 1000) ** (Rd / Cp)
# =============================================================================
            # Inverse distance weighting
            D = ComputeDistance(station,stationlist[i])
            # Ignore distance larger than 5km
            if(D > 0.1):
                continue
            n.append(elementlist[i][j].valuelist[k] / D ** 2)
            if math.isnan(n[-1]):
                d.append(0)
            else:
                d.append(1 / D ** 2)
        
    # Check that at least 3 other stations are used
    if len(n) > 0:
        idw_weight = np.nansum(n)/np.nansum(d)
        return idw_weight
    else:
        return(np.nan)

# Return the spatial regression estimate by other stations
# Weight determined by standard error between target station and each of the neighboring stations
def SRT(station, j):
    # Initialize values
    index = stationlist.index(station)
    targetvallist = elementlist[index][j].valuelist
    othervallist = []
    model = []
    regpara = [[0] * 3 for i in range(len(stationlist))] #regression parameters
    bar_x  = [[np.nan]* len(targetvallist) for i in range(len(stationlist))] #explained variable for each neighboring station
    weighted_x = [0] * len(targetvallist)
    
    for i in range(len(stationlist)):
        # Obtain other station data if element exists
        if stationlist[i] != station and elementlist[i][j] != 0:
            othervallist.append(elementlist[i][j].valuelist)
        else:
            continue
            
        # Fliter out NaN values
        mask = ~np.isnan(othervallist[-1])
                
        # Put regression parameters into array
        model = sm.OLS(targetvallist[mask], sm.add_constant(othervallist[-1][mask]))
        results = model.fit()
        regpara[i] = np.append(results.params,results.ssr)
        
        # Get estimated x for each time step by linear regression model of another station
        for k in range(len(targetvallist)):
            if othervallist[-1][k] == np.nan:
                continue
            else:
                bar_x[i][k] = regpara[i][1] * othervallist[-1][k] + regpara[i][0] #bar_x = slope*x + c
    
    #The following part generates the weighted x for each time step of target station
    for k in range(len(targetvallist)):
        for i in range(len(stationlist)):
            # Initialize variables for formula: sqrt(a * b) = sqrt(sum(bar_x^2 * s_error^-2)*sum(s_error^2))
            a = 0 ; b = 0
            
            # Obtain other station data
            if stationlist[i] == station:
                continue
            
            # Remove zeroes
            if sum(regpara[i]) == 0:
                continue
            
            a += (bar_x[i][k]) ** 2 / regpara[i][2]
            b += regpara[i][2]
        # Get weighted x for each time step
        weighted_x[k] = math.sqrt(a * b)
    return weighted_x

###############################################################################   
# Main driver code for Q5

# Select spatial-range test modes
    # (1): IDW
    # (2): SRT
spatial_method = 2

# Define the threshold for spatial-range test
srthreshold = [0]*4
# Looping through the stations
for i in range(len(stationlist)):
    if print_level <= 2:
        print(stationlist[i].sid)
    # Define the element list using the object "element"
    for j in range(len(elementlist[i])):
        if(elementlist[i][j] == 0):
            continue
        # Spatial-range test
        # (1): IDW (2019/01/29: tested to be slow and have higher MAD / RMSD)
        if (spatial_method == 1):
            diff = []; weighted = []; observed = []
            for k in range(len(elementlist[i][j].datelist)):
                weighted.append(IDW(stationlist[i],j,k))
                observed.append(elementlist[i][j].valuelist[k])
                
                # Define the difference
                if not(np.isnan(weighted[k]) or np.isnan(observed[k])):
                    diff.append(abs(weighted[k] - observed[k]))
                else:
                    diff.append(np.nan)
                    continue
        # (2): SRT
        elif (spatial_method == 2):
            weighted = SRT(stationlist[i],j)
            observed = elementlist[i][j].valuelist
            diff = abs(weighted - observed)
            
        # Define the threshold for spatial-range test using fitting to skewnorm curve method
        diffv = np.asarray(diff)[~np.isnan(diff)]
        # Skip spatial-range test when all differences are NaN
        if len(diffv) == 0:
            continue                      
        
        srthreshold[0] = skewnorm.ppf(1-pvalue1,*skewnorm.fit(diffv))
        srthreshold[1] = skewnorm.ppf(1-pvalue2,*skewnorm.fit(diffv))
        srthreshold[2] = skewnorm.ppf(1-pvalue3,*skewnorm.fit(diffv))

        for k in range(len(elementlist[i][j].datelist)):            
            # Bypass QC for NaN data           
            if(np.isnan(diff[k])):
                elementlist[i][j].flaglist[k] += 0
            
            # Check whether the difference between regressed and observed data exceeds srthreshold     
            elif (diff[k] <= srthreshold[0]):
                elementlist[i][j].flaglist[k] += 1e5

            elif (srthreshold[0] <= diff[k] < srthreshold[1]):
                elementlist[i][j].flaglist[k] += 2e5

            elif (srthreshold[1] <= diff[k] < srthreshold[2]):
                elementlist[i][j].flaglist[k] += 3e5
                 
            elif (srthreshold[2] <= diff[k]):
                elementlist[i][j].flaglist[k] += 4e5

        # Evaluate model performance:
        observedv = np.asarray(observed)[~np.isnan(observed) & ~np.isnan(weighted)]
        weightedv = np.asarray(weighted)[~np.isnan(observed) & ~np.isnan(weighted)]
        print("Model performance:")
        print("Root mean square difference:", RMSD(observedv,weightedv))
        print("Normalized root mean square difference:", RMSD(observedv,weightedv)/(max(observedv)-min(weightedv)))
        print("Mean absolute difference:", MAD(observedv,weightedv))
        print("Normalized mean absolute difference:", MAD(observedv,weightedv)/(max(observedv)-min(weightedv)))
        
        # Obtaining composite flag again
        elementlist[i][j].get_composite_flag()
                     
        # Print out results
        print('Station:', stationlist[i].sid, 'SPATIAL TEST RESULTS for element', elementid[j], elementname[j])
        if print_level <= 2:
            print(elementlist[i][j].flaglist)

###############################################################################
#-----------------------------------------------------------------------------#
# PART III: POST-QC ANALYSIS
#-----------------------------------------------------------------------------#
###############################################################################
# Create a 3D array to hold all flags
allflag = np.empty((len(stationlist), len(elementid), longestdllen), dtype=object)
allflag[:] = 'nan'
for i in range(len(stationlist)):
	for j in range(len(elementid)):
		if elementlist[i][j] != 0:
			for k in range(len(elementlist[i][j].datelist)):
				if ~np.isnan(elementlist[i][j].flaglist[k]):
					allflag[i][j][k] = str(elementlist[i][j].flaglist[k]).zfill(10)
		else:
			continue

#Print out all flags and count flags
counter_flag = [[0]*5 for i in range(10)]
for i in range(len(stationlist)):
    for j in range(len(elementlist[i])):
        if elementlist[i][j] != 0:
            for k in range(len(elementlist[i][j].datelist)):
                if print_level <= 3:
                    print(allflag[i][j][k],end=' ')
                if (allflag[i][j][k] != 'nan'):
                    for d in range(10):
                        if '0' in allflag[i][j][k][d] and d <= 7:
                            counter_flag[d][0] += 1
                        elif '1' in allflag[i][j][k][d]:
                            counter_flag[d][1] += 1
                        elif '2' in allflag[i][j][k][d]:
                            counter_flag[d][2] += 1
                        elif '3' in allflag[i][j][k][d]:
                            counter_flag[d][3] += 1
                        elif '4' in allflag[i][j][k][d]:
                            counter_flag[d][4] += 1

# Find the percentage of flags
percentage_flag = np.array(counter_flag) / max(np.array(counter_flag).ravel())

# Put statistics into data frame
df_pf = pd.DataFrame(percentage_flag)
df_pf.columns = ['Bypassed','Passed','Suspicious','High. Susp.','Error']
df_pf.index =['Range','Step','Persistency','Consistency','Spatial','RF Algorithm','Composite','Final','Ver','Ver']
                    
###############################################################################
# Evaluate time elapsed
t1 = time.perf_counter() - t0
print("\nTime elapsed: ", round(t1 , 2) , "s")
print("QC finished")
print("Statistics for QC")
print(df_pf)
###############################################################################
# Appendix
# i: dummy variable looping through stations
# j: dummy variable looping through elements
# k: dummy variable looping through step nodes
# d: dummy variable looping through digits of QA flags
###############################################################################