#!/nobackup/users/patel/miniconda2/bin/python
#
# Reading gridded population in 2015 from Popmap (100m x 100m resolution) for a given lat,lon window
# Write to NetCDF, and preview on screen.
#
# Population data is read from DBF format.
#  -99999 : No data
#  -99998 : Insignificant
#  -99997 : Secret
# Grid cell ID ExxxxNyyyy: Coordinates SW-corner grid box in Rijksdriehoeksmeting: (xxxx*100,yyyy*1000)
#
# Algorithm:
#    1. Load file using gDAL library.
#    2. Retrieve corner coordinates using geotransform.
#    3. Convert projection of corner coordinates to Delhi.
#    4. Build a new grid (in Delhi projection).
#    5. Store population density from raster file.
#
# Bas Mijling, June 2016
# Edited: Karsh Patel, November 2017


import pyproj
import numpy as np
import matplotlib.pyplot as plt
import sys
import struct
import gdal
import math
import numpy
from matplotlib import cm

domain = "delhi"
dataFile  = "/usr/people/patel/retina_delhi/test/source/population/smallgtiff.tif"
#dataFile  = "C:\Users\Karsh\Documents\Academics\TU Delft\Internship\Popmap\smallgtiff.tif"

wgs84 = pyproj.Proj("+init=EPSG:4326") 	# Lat/Lon projection
lat0, lon0 = (28.68, 77.10)		# Delhi centre
myProj = pyproj.Proj("+proj=sterea +lat_0=%f +lon_0=%f +ellps=bessel +units=m" % (lat0,lon0)) # Projection for Delhi

# 1. Read raster layer
print("\n Read population data from geotiff")
raster = gdal.Open(dataFile, gdal.GA_ReadOnly)

# 2. Get all coordinates from geotransform
geotransform = raster.GetGeoTransform()     # Stores top left coordinates and pixel size
cols = raster.RasterXSize       # Size of image file
rows = raster.RasterYSize
xarr=[0,cols]
yarr=[0,rows]
ext = []                        # Initialize to store extent of image
# Routine to calculate all coordinates from top left coordinate and pixel width
for px in xarr:
    for py in yarr:
        x=geotransform[0]+(px*geotransform[1])+(py*geotransform[2])
        y=geotransform[3]+(px*geotransform[4])+(py*geotransform[5])
        ext.append([x,y])
centerx = geotransform[0]+((cols/2)*geotransform[1])
centery = geotransform[3]+((rows/2)*geotransform[5])

# Assign latitudes and longitudes seperatly for clarity
ULlon, ULlat=ext[0][0], ext[0][1]
LLlon, LLlat=ext[1][0], ext[1][1]
URlon, URlat=ext[2][0], ext[2][1]
LRlon, LRlat=ext[3][0], ext[3][1]
CTlon, CTlat= centerx, centery

print("\n\t Coordinates of the image file:")
print("\t\t Upper left:\t",   ULlon, ULlat)
print("\t\t Lower left:\t",   LLlon, LLlat)
print("\t\t Upper Right:\t",  URlon, URlat)
print("\t\t Lower Right:\t",  LRlon, LRlat)
print("\t\t Center     :\t",  CTlon, CTlat)

# Values from cmd line: gdalinfo "filename.tiff" (for verification of above values)
# ULlon, ULlat=76.8370000, 28.8832000
# LLlon, LLlat=76.8370000, 28.4059000
# URlon, URlat=77.3451000, 28.8832000
# LRlon, LRlat=77.3451000, 28.4059000
# CTlon, CTlat=77.0910500, 28.6445500

# 3. Convert coordinates to Delhi projection
print("\n\t Converting coordinate projection...")
ULx,ULy = pyproj.transform(wgs84, myProj, ULlon, ULlat)
LLx,LLy = pyproj.transform(wgs84, myProj, LLlon, LLlat)
LRx,LRy = pyproj.transform(wgs84, myProj, LRlon, LRlat)
URx,URy = pyproj.transform(wgs84, myProj, URlon, URlat)
CTx,CTy = pyproj.transform(wgs84, myProj, CTlon, CTlat)

# Note: After transforming projection, corners are no longer represent square
# or rectangle, they need to be rouded off to make them so.

# Rounding off coordinates
print("\n\t Rounding off coordinates...")
ULx, ULy = (int(ULx/1000)*1000, int(ULy/1000)*1000)
LLx, LLy = (int(LLx/1000)*1000, int(LLy/1000)*1000)
LRx, LRy = (int(LRx/1000)*1000, int(LRy/1000)*1000)
URx, URy = (int(math.ceil(float(URx)/1000)*1000), int(URy/1000)*1000)
CTx, CTy = (int(CTx/100)*100, int(CTy/1000)*1000)

print("\n\t Transformed coordinates of the image file:")
print("\t\t Upper left:\t",   ULx, ULy)
print("\t\t Lower left:\t",   LLx, LLy)
print("\t\t Upper Right:\t",  URx, URy)
print("\t\t Lower Right:\t",  LRx, LRy)
print("\t\t Center     :\t",  CTx, CTy)

# 4. Prepare grid
print("\n\t Preparing the resulting grid...")
nx = 100                        # Resoltuion of 100m
ny = 100
rdx = np.zeros(nx, dtype=np.int)     # Initializing arrays
rdy = np.zeros(ny, dtype=np.int)

# Fill RD grid definition
rdx = np.linspace(LLx, LRx, nx)
rdy = np.linspace(LLy, ULy, ny)
#print "RD x y", rdx, rdy
print("\t\t Grid resolution: \t", len(rdx), "by", len(rdy), "meters")

# Make grid
print("\t\t Calculating lat,lon coordinates of grid...")
xx,yy = np.meshgrid(rdx,rdy)

# 5. Read band(population density numbers) from raster
print("\n\t Reading population density from image file...")
band = raster.GetRasterBand(1)
print("\t\t Original size of the image:\t", band.XSize, band.YSize)

# Read image in original size
print("\t\t Storing image in original size as array called dost...")
dost = band.ReadAsArray(0,0,band.XSize,band.YSize) # No. of cols and no. of rows

print("\t\t Size of dost:\t", dost.shape)
# Plot on PC (linux doesn't support jpeg and gdal simultaneously)
#plt.imshow(dost, cmap='autumn')
#plt.show()

# Store values as a text file
np.savetxt('dost.txt',dost)

# Density numbers stored in scanline as tuples of binary numbers
scanline = band.ReadRaster(xoff=0, yoff=0,          # Read image from top left (0,0) to bottom right (xsize = 610, ysize = 573)
                           xsize=band.XSize, ysize=band.YSize,
                           buf_xsize=100, buf_ysize=100, # Downsample the image from original size (610 by 510 to 100 by 100)
                           buf_type=gdal.GDT_Float32)
print("\t\t Population density stored as binary in tuples of length:", len(scanline))

# Converting binary numbers to proper human numbers
value = struct.unpack('f' * 10000, scanline)
print("\t\t Population density stored as float in tuples of length:", len(value))

# Convert tuples to 2D array
value_array = np.asarray(value)
re_value_array = np.reshape(value_array,(100,100))
print("\t\t Population density stored as float in array of size:", re_value_array.shape)

# Plot on PC (linux doesn't support jpeg and gdal simultaneously)
# plt.imshow(re_value_array)
# plt.show()

# Store values as a text file
np.savetxt('value.txt',value)

# Plot population arrays
plt.figure(figsize=(10,10))
plt.imshow(dost, cmap=cm.YlOrRd)
plt.title('Population density array',fontsize=18)
plt.xlabel('Rows',fontsize=15)
plt.ylabel('Columns',fontsize=15)
plt.tick_params(labelsize=20)
c = plt.colorbar()
c.set_label('Estimated persons per grid square',fontsize=15)
