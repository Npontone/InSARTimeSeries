import numpy as np
from osgeo import gdal
import glob
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import scipy as sp
from scipy.fftpack import fft


import Functions

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



#This takes a long time to run, I think there are many optimizations to be done, probably with the
#for loops.

New_Raster = np.zeros_like(data_cube[0])
for row in np.arange(rows):
    for col in np.arange(cols):
        time_series = data_cube[:, row, col]        
        if not np.isnan(np.sum(time_series)):
            New_Raster[row, col] = Functions.CompareBIC(np.array(serial),time_series)
        else:
            New_Raster[row,col] = np.nan
        #print(New_Raster[row,col])

new_ras = wdir + "classified.tif" #The new raster
new_tiff = driver_tiff.Create(New_Raster, xsize=orig_ras.RasterXSize, ysize=orig_ras.RasterYSize, 
                              bands=1, eType=gdal.GDT_Float32)
new_tiff.GetRasterBand(1).WriteArray(New_Raster) #Put array you want to write here
new_tiff.GetRasterBand(1).SetNoDataValue(-999) #Set no data value
new_tiff.SetGeoTransform(geot) #Set Geotransfrom
new_tiff.SetProjection(proj) #Set Projection
new_tiff = None #close file





