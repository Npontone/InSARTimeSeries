# -*- coding: utf-8 -*-
"""
Description: Automated classification of ISBAS time-series based on reading a
stack of raster data
Author: David Gee (david.gee@terramotion.co.uk)

"""

import numpy as np
from osgeo import gdal
import glob
import pandas as pd
from datetime import datetime
from scipy import stats
import scipy as sp

###############################################################################


#wdir = r'D:\Time_Series_Classification\Analysis_on_Raster_Data\SYork'
wdir = r'C:\Users\Nicho\Documents\GeoDynamics'
wdir = wdir + "/"

#Import linear velocities
lin_velo = gdal.Open(wdir + "ortho_velocity_utm_isbas.tif")
lin_velo = np.array(lin_velo.GetRasterBand(1).ReadAsArray())
lin_velo = lin_velo * 1000

#Import time-series rasters into 3D array
tiffs = glob.glob(wdir + "2*.tif")
data_cube = []
for a in tiffs:
    print("Importing image " + a[len(wdir):len(wdir) + 8] + " into 3D array")
    ds = gdal.Open(a)
    myarray = np.array(ds.GetRasterBand(1).ReadAsArray())
    data_cube.append(myarray)
data_cube = np.array(data_cube)
data_cube = data_cube * 1000

#Get dates from file names
dates = []
serial = []
print("Derriving image dates")
for j in tiffs:
    date_part = j[len(wdir): len(wdir) + 8] 
    date_part = datetime.strptime(date_part, '%Y%m%d')
    dates.append(date_part) #all_tiffs contains dates of all the images
    numbers = datetime.timestamp(date_part)
    numbers = numbers / 24 / 60 / 60
    numbers = int(numbers)
    serial.append(numbers)
    
print("Done")

#Get dimensions of data cube
no_images = data_cube.shape[0]
rows = data_cube.shape[1]
cols = data_cube.shape[2]

#Get mask of original coverage
cov = myarray != -999
cov = cov.astype(int) # 1s have coverage
cov = cov.astype('float')
cov[cov == 0 ] = np.nan

#Get Projection info
orig_ras = gdal.Open(a) #Original raster the geographical information is taken from
geot = orig_ras.GetGeoTransform() #Transformation info from orig raster
proj = orig_ras.GetProjection() #Projection info from orig raster
driver_tiff = gdal.GetDriverByName("GTiff")


###############################################################################

#Linear Regression

#Define P Value Calculation from Linear Regression 
def lin_reg_p(row, col):
    time_series = data_cube[:, row, col]
    slope, intercept, r_value, p_value, std_err = stats.linregress(serial, time_series)
    return p_value

#Calculate P Value for Array
p_val = np.zeros_like(data_cube[0])
for row in np.arange(rows):
    print("Calculating P value for array: Row " + str(row) + " out of " + str(rows))
    for col in np.arange(cols):
        p_val[row, col] = lin_reg_p(row, col)
p_val = p_val * cov #mask to original coverage

#Orthorectify P Values
new_ras = wdir + "P_Value.tif" #The new raster
new_tiff = driver_tiff.Create(new_ras, xsize=orig_ras.RasterXSize, ysize=orig_ras.RasterYSize, 
                              bands=1, eType=gdal.GDT_Float32)
new_tiff.GetRasterBand(1).WriteArray(p_val) #Put array you want to write here
new_tiff.GetRasterBand(1).SetNoDataValue(-999) #Set no data value
new_tiff.SetGeoTransform(geot) #Set Geotransfrom
new_tiff.SetProjection(proj) #Set Projection
new_tiff = None #close file

#------------------------------------------------------------------------------

#Define P Value Calculation from Linear Regression 
def lin_reg_r(row, col):
    time_series = data_cube[:, row, col]
    slope, intercept, r_value, p_value, std_err = stats.linregress(serial, time_series)
    return r_value

#Calculate R Value for Array
r_val = np.zeros_like(data_cube[0])
for row in np.arange(rows):
    print("Calculating R value for array: Row " + str(row) + " out of " + str(rows))
    for col in np.arange(cols):
        r_val[row, col] = lin_reg_r(row, col)
r_val = r_val * cov #mask to original coverage
r_sq = r_val * r_val

#Orthorectify R Values
new_ras = wdir + "R_Value.tif" #The new raster
new_tiff = driver_tiff.Create(new_ras, xsize=orig_ras.RasterXSize, ysize=orig_ras.RasterYSize, 
                              bands=1, eType=gdal.GDT_Float32)
new_tiff.GetRasterBand(1).WriteArray(r_val) #Put array you want to write here
new_tiff.GetRasterBand(1).SetNoDataValue(-999) #Set no data value
new_tiff.SetGeoTransform(geot) #Set Geotransfrom
new_tiff.SetProjection(proj) #Set Projection
new_tiff = None #close file

#Orthorectify R Squared Values
new_ras = wdir + "R_Squared.tif" #The new raster
new_tiff = driver_tiff.Create(new_ras, xsize=orig_ras.RasterXSize, ysize=orig_ras.RasterYSize, 
                              bands=1, eType=gdal.GDT_Float32)
new_tiff.GetRasterBand(1).WriteArray(r_sq) #Put array you want to write here
new_tiff.GetRasterBand(1).SetNoDataValue(-999) #Set no data value
new_tiff.SetGeoTransform(geot) #Set Geotransfrom
new_tiff.SetProjection(proj) #Set Projection
new_tiff = None #close file


###############################################################################


#Segment and perform linear regressions
def segmented(row,col):
    for row in np.arrange(rows):
        print("Yipee")


###############################################################################

#Anderson Darling Test (Non-linearity) - Value
def and_darl_and(row, col):
    time_series = data_cube[:, row, col]
    Anderson = sp.stats.anderson(time_series,dist = 'norm')
    Ander = Anderson[0]
    And_Thr = Anderson[1]
    And_Thr = And_Thr[2] #Selects at 5% but this can be changed
    return Ander

#Anderson Darling Test (Non-linearity) - Threshold
def and_darl_thr(row, col):
    time_series = data_cube[:, row, col]
    Anderson = sp.stats.anderson(time_series,dist = 'norm')
    Ander = Anderson[0]
    And_Thr = Anderson[1]
    And_Thr = And_Thr[2] #Selects at 5% but this can be changed
    return And_Thr


#Calculate Anderson Darling value for the array
and_val = np.zeros_like(data_cube[0])
for row in np.arange(rows):
    print("Calculating Anderson Darling value for array: Row " + str(row) + " out of " + str(rows))
    for col in np.arange(cols):
        and_val[row, col] = and_darl_and(row, col)
and_val = and_val * cov #mask to original coverage

#Calculate Anderson Darling threshold for the array
and_thr = np.zeros_like(data_cube[0])
for row in np.arange(rows):
    print("Calculating Anderson Darling threshold for array: Row " + str(row) + " out of " + str(rows))
    for col in np.arange(cols):
        and_thr[row, col] = and_darl_thr(row, col)
and_thr = and_thr * cov #mask to original coverage


#Orthorectify Anderson Darling values
new_ras = wdir + "And_Darl_Val.tif" #The new raster
new_tiff = driver_tiff.Create(new_ras, xsize=orig_ras.RasterXSize, ysize=orig_ras.RasterYSize, 
                              bands=1, eType=gdal.GDT_Float32)
new_tiff.GetRasterBand(1).WriteArray(and_val) #Put array you want to write here
new_tiff.GetRasterBand(1).SetNoDataValue(-999) #Set no data value
new_tiff.SetGeoTransform(geot) #Set Geotransfrom
new_tiff.SetProjection(proj) #Set Projection
new_tiff = None #close file

#Orthorectify Anderson Darling thresholds
new_ras = wdir + "And_Darl_Thres.tif" #The new raster
new_tiff = driver_tiff.Create(new_ras, xsize=orig_ras.RasterXSize, ysize=orig_ras.RasterYSize, 
                              bands=1, eType=gdal.GDT_Float32)
new_tiff.GetRasterBand(1).WriteArray(and_thr) #Put array you want to write here
new_tiff.GetRasterBand(1).SetNoDataValue(-999) #Set no data value
new_tiff.SetGeoTransform(geot) #Set Geotransfrom
new_tiff.SetProjection(proj) #Set Projection
new_tiff = None #close file


###############################################################################

#Annual Peroidicity Index (Berti et al., 2013)

def API(row, col):
    
    #Oversampe data at six days
    time_series = data_cube[:, row, col]
    samp = 6 #Oversample at 6 day intervals for sentinel
    df = pd.DataFrame()
    global dates
    dates_orig = dates
    df['Date'] = dates
    df['Time_Series'] = time_series
    #Change data in datetime
    df['Date'] = pd.to_datetime(df.Date)
    df['Date'] = pd.to_datetime((df['Date'].astype(np.int64)//10**9 * 10**9).astype('datetime64[ns]'))
    new_range = pd.date_range(df.Date[0], df.Date.values[-1], freq= str(samp)+'D')
    new_range = df.set_index('Date').reindex(new_range).interpolate(method='linear').reset_index() #Define type of interpolation
    time_series = new_range["Time_Series"] #Time Series now sampled to new intervals
    time_series2 = np.float32(time_series)
    
    #Fast Fourier Transform Analysis
    temp_fft = sp.fftpack.fft(time_series2)   
    temp_psd = np.abs(temp_fft) ** 2
    fftfreq = sp.fftpack.fftfreq(len(temp_psd), 1./(365/samp)) #Yearly divided by samp interval
    
    #Find Maximum PSD values at <0.5 1/year and between 0.8-1.2 1/year
    AP_Expo = pd.DataFrame()
    AP_Expo['fftfreq'] = fftfreq
    AP_Expo['temp_psd'] = temp_psd
    uncor = AP_Expo.loc[(AP_Expo['fftfreq'] <= 0.5) & (AP_Expo['fftfreq'] > 0.0)]
    P0 = uncor['temp_psd'].mean() #Take mean to reduce large initial effects (or min?)
    cor = AP_Expo.loc[(AP_Expo['fftfreq'] <= 1.2) & (AP_Expo['fftfreq'] >= 0.8)]
    P1 = cor['temp_psd'].max()
    if P0 >= P1:
        ap = 0.5*(P1/P0)
    else:
        ap = 1 - (0.5*(P0/P1))    
    
    #Restore original dates (not the oversampled dates at six days)
    dates = new_range["index"] 
    dates = dates_orig

    return ap


#Calculate API for the array

api = np.zeros_like(data_cube[0])
for row in np.arange(rows):
    print("Calculating Annual Periodicity Index for array: Row " + str(row) + " out of " + str(rows))
    for col in np.arange(cols):
        api[row, col] = API(row, col)
api = api * cov #mask to original coverage


#Orthorectify API values
new_ras = wdir + "API.tif" #The new raster
new_tiff = driver_tiff.Create(new_ras, xsize=orig_ras.RasterXSize, ysize=orig_ras.RasterYSize, 
                              bands=1, eType=gdal.GDT_Float32)
new_tiff.GetRasterBand(1).WriteArray(api) #Array to write here
new_tiff.GetRasterBand(1).SetNoDataValue(-999) #Set no data value
new_tiff.SetGeoTransform(geot) #Set Geotransfrom
new_tiff.SetProjection(proj) #Set Projection
new_tiff = None #close file


###############################################################################

#Classification
 
#Thres_Lin = np.zeros_like(data_cube[0])
Thres_Lin = np.zeros_like(lin_velo)
Thres_Lin = Thres_Lin + 0.05 #If P val over 95% then linear
Thres_Cyc = np.zeros_like(lin_velo)
Thres_Cyc = Thres_Cyc + 0.4 #If API > 0.4 is has annual cyclicity
p_value = gdal.Open(wdir + "P_Value.tif")
p_value = np.array(p_value.GetRasterBand(1).ReadAsArray())
ap = gdal.Open(wdir + "API.tif")
ap = np.array(ap.GetRasterBand(1).ReadAsArray())
And = gdal.Open(wdir + "And_Darl_Val.tif")
And = np.array(And.GetRasterBand(1).ReadAsArray())
And_Thr = gdal.Open(wdir + "And_Darl_Thres.tif")
And_Thr = np.array(And_Thr.GetRasterBand(1).ReadAsArray())

print("Classifying rasters")

#------------------------------------------------------------------------------

#Stable Pixels
#From linear velocities find those which are +-2mm/yr from the linear analysis and hence stable
stab_pos = lin_velo <= 2
stab_pos = stab_pos.astype(int) #Velocities less than 2mm/yr as 1s
stab_neg = lin_velo >= -2
stab_neg = stab_neg.astype(int) #Velocities greater than 2mm/yr as 1s
stab_2mm = stab_pos + stab_neg #Values of 2 are those +-2mm/yr
stab_2mm = stab_2mm == 2
stab_2mm = stab_2mm.astype(int) #Values of 1 are those within +-2mm/yr

#Stats tests
s1 = p_value > Thres_Lin #Linear Test: True = not linear
s1 = s1.astype(int) # 1s are not linear
s1 = s1 + stab_2mm
s1 = s1 > 0 #(or statement essentially)
s1 = s1.astype(int) # 1s are not linear or between +-2mm/yr
s2 = ap < Thres_Cyc #Periodicity Test: True = not periodic
s2 = s2.astype(int) # 1s are not periodic
stable = s1 + s2 #2's are stable (not lin and not periodic)
stable = stable == 2
stable = stable.astype(int)
stable_1 = stable
stable = stable * cov #Mask orignal coverage
stable[stable == 0 ] = np.nan #Set all other pixels to nan


#Orthorectify Stable Raster
new_ras = wdir + "stable.tif" #The new raster
new_tiff = driver_tiff.Create(new_ras, xsize=orig_ras.RasterXSize, ysize=orig_ras.RasterYSize, 
                              bands=1, eType=gdal.GDT_Float32)
new_tiff.GetRasterBand(1).WriteArray(stable) #Put array you want to write here
new_tiff.GetRasterBand(1).SetNoDataValue(-999) #Set no data value
new_tiff.SetGeoTransform(geot) #Set Geotransfrom
new_tiff.SetProjection(proj) #Set Projection
new_tiff = None #close file

#------------------------------------------------------------------------------

#Cyclical No Trend
c1 = p_value > Thres_Lin #Linear Test: True = not linear
c1 = c1.astype(int) # 1s are not linear
c1 = c1 + stab_2mm
c1 = c1 > 0 #(or statement essentially)
c1 = c1.astype(int) # 1s are not linear or between +-2mm/yr
c2 = ap > Thres_Cyc #Periodicity Test: True = are periodic (note sign change from above)
c2 = c2.astype(int) # 1s are periodic
cyc_no_trend = c1 + c2 #2's are not linear but periodic (cyclical without trend)
cyc_no_trend = cyc_no_trend == 2
cyc_no_trend = cyc_no_trend.astype(int)
cyc_no_trend_2 = cyc_no_trend * 2
cyc_no_trend = cyc_no_trend * cov
cyc_no_trend[cyc_no_trend == 0 ] = np.nan #Set all other pixels to nan


#Orthorectify Cyclical No Trend Raster
new_ras = wdir + "cyclical_no_trend.tif" #The new raster
new_tiff = driver_tiff.Create(new_ras, xsize=orig_ras.RasterXSize, ysize=orig_ras.RasterYSize, 
                              bands=1, eType=gdal.GDT_Float32)
new_tiff.GetRasterBand(1).WriteArray(cyc_no_trend) #Put array you want to write here
new_tiff.GetRasterBand(1).SetNoDataValue(-999) #Set no data value
new_tiff.SetGeoTransform(geot) #Set Geotransfrom
new_tiff.SetProjection(proj) #Set Projection
new_tiff = None #close file

#------------------------------------------------------------------------------

#Linear
#Linear Stats Test
l1 = p_value < Thres_Lin #Linear Test: True = linear
l1 = l1.astype(int) # 1s are linear
move_2mm = stab_2mm == 0
move_2mm = move_2mm.astype(int)
l1 = l1 + move_2mm
l1 = l1 > 1 #(or statement essentially)
l1 = l1.astype(int) # 1s are linear and greater than +-2mm/yr
#API test
l2 = ap < Thres_Cyc #Periodicity Test: True = not periodic
l2 = l2.astype(int) # 1s are not periodic 
#Anderson Darling
l3 = And > And_Thr # AD: True = linear (Thres at 5% - set in function above)
l3 = l3.astype(int) # 1s are linear
linear = l1 + l2 + l3 #3's are linear
linear = linear == 3
linear = linear.astype(int)
linear_3 = linear * 3
linear = linear * cov
linear[linear == 0 ] = np.nan #Set all other pixels to nan


#Orthorectify Linear Raster
new_ras = wdir + "linear.tif" #The new raster
new_tiff = driver_tiff.Create(new_ras, xsize=orig_ras.RasterXSize, ysize=orig_ras.RasterYSize, 
                              bands=1, eType=gdal.GDT_Float32)
new_tiff.GetRasterBand(1).WriteArray(linear) #Put array you want to write here
new_tiff.GetRasterBand(1).SetNoDataValue(-999) #Set no data value
new_tiff.SetGeoTransform(geot) #Set Geotransfrom
new_tiff.SetProjection(proj) #Set Projection
new_tiff = None #close file

#------------------------------------------------------------------------------

#Non-linear
nl1 = p_value < Thres_Lin #Linear Test: True = linear
nl1 = nl1.astype(int) # 1s are linear
nl1 = nl1 + move_2mm
nl1 = nl1 > 1 #(or statement essentially)
nl1 = nl1.astype(int) # 1s are linear and greater than +-2mm/yr
#API test
nl2 = ap < Thres_Cyc #Periodicity Test: True = not periodic
nl2 = nl2.astype(int) # 1s are not periodic 
#Anderson Darling
nl3 = And < And_Thr # AD: True = non - linear term significant (Thres. at 5%) (NB. Sign change from above)
nl3 = nl3.astype(int) # 1s have sig. non linear term
non_linear = nl1 + nl2 + nl3
non_linear = non_linear == 3
non_linear = non_linear.astype(int)
non_linear_4 = non_linear * 4
non_linear = non_linear * cov
non_linear[non_linear == 0 ] = np.nan #Set all other pixels to nan


#Orthorectify Non-Linear Raster
new_ras = wdir + "non_linear.tif" #The new raster
new_tiff = driver_tiff.Create(new_ras, xsize=orig_ras.RasterXSize, ysize=orig_ras.RasterYSize, 
                              bands=1, eType=gdal.GDT_Float32)
new_tiff.GetRasterBand(1).WriteArray(non_linear) #Put array you want to write here
new_tiff.GetRasterBand(1).SetNoDataValue(-999) #Set no data value
new_tiff.SetGeoTransform(geot) #Set Geotransfrom
new_tiff.SetProjection(proj) #Set Projection
new_tiff = None #close file

#------------------------------------------------------------------------------

#Cyclical with Trend
ct1 = p_value < Thres_Lin #Linear Test: True = linear
ct1 = ct1.astype(int) # 1s are linear
ct1 = ct1 + move_2mm
ct1 = ct1 > 1 #(or statement essentially)
ct1 = ct1.astype(int) # 1s are linear and greater than +-2mm/yr
#API test
ct2 = ap > Thres_Cyc #Periodicity Test: True = periodic
ct2 = ct2.astype(int) # 1s are periodic
cyc_trend = ct1 + ct2
cyc_trend = cyc_trend == 2
cyc_trend = cyc_trend.astype(int)
cyc_trend_5 = cyc_trend * 5
cyc_trend = cyc_trend * cov
cyc_trend[cyc_trend == 0] = np.nan 


#Orthorectify Cyclical with Trend Raster
new_ras = wdir + "cyclical_with_trend.tif" #The new raster
new_tiff = driver_tiff.Create(new_ras, xsize=orig_ras.RasterXSize, ysize=orig_ras.RasterYSize, 
                              bands=1, eType=gdal.GDT_Float32)
new_tiff.GetRasterBand(1).WriteArray(cyc_trend) #Put array you want to write here
new_tiff.GetRasterBand(1).SetNoDataValue(-999) #Set no data value
new_tiff.SetGeoTransform(geot) #Set Geotransfrom
new_tiff.SetProjection(proj) #Set Projection
new_tiff = None #close file

#------------------------------------------------------------------------------

#Classified raster
classified = stable_1 + cyc_no_trend_2 + linear_3 + non_linear_4 + cyc_trend_5
classified = classified * cov
classified[classified == 0] = np.nan 


#Ortho rectify classified raster
new_ras = wdir + "classified.tif" #The new raster
new_tiff = driver_tiff.Create(new_ras, xsize=orig_ras.RasterXSize, ysize=orig_ras.RasterYSize, 
                              bands=1, eType=gdal.GDT_Float32)
new_tiff.GetRasterBand(1).WriteArray(classified) #Put array you want to write here
new_tiff.GetRasterBand(1).SetNoDataValue(-999) #Set no data value
new_tiff.SetGeoTransform(geot) #Set Geotransfrom
new_tiff.SetProjection(proj) #Set Projection
new_tiff = None #close file



"""
For Testing

import matplotlib.pyplot as plt

#Select Row and Column
r, c = 3, 3

time_series = data_cube[:, r , c]
plt.plot(time_series)
plt.show()

and_darl_and(r,c)
and_darl_thr(r,c)
API(r,c)
lin_reg_r(r,c)

"""

   
