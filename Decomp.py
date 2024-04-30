# -*- coding: utf-8 -*-
"""
Description: Automated classification of ISBAS time-series based on reading a
stack of raster data
Author: David Gee (david.gee@terramotion.co.uk)

"""

import numpy as np
from osgeo import gdal
import glob
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose



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

#Decomposition


def contains_nan(arr):
    return np.isnan(arr).any()

def decomp(row, col):

    time_series = data_cube[:, row, col]

    if contains_nan(time_series) == True:
        return(["NaN"])
    
    else:

        result = seasonal_decompose(time_series, model='multiplicative')

        trend = result.trend
        seasonal = result.seasonal
        residual = result.resid

        list = [trend,seasonal,residual]

        return trend,seasonal,residual


#Calculate P Value for Array
p_val = np.zeros_like(data_cube[0])
for row in np.arange(rows):
    for col in np.arange(cols):
        p_val[row, col] = decomp(row, col)




