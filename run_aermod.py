#!/nobackup/users/patel/miniconda2/bin/python
##!/Users/bas/anaconda/bin/python
#
# Wrapper script to run AERMOD with start hour and end hour <yyyymmddhh> <yyyymmddhh>
# Output files in work directory (total concentration, and by source group):
#   conc_<startHour>-<endHour>_all
#   conc_<startHour>-<endHour>_A
#   conc_<startHour>-<endHour>_B
#   conc_<startHour>-<endHour>_C
#
# Prepare and run AERMOD from work directory:
# (1) Extract meteo
# (2) Prepare emissions for group A,B,C
# (3) Extract background concentrations, matching run period
# (4) Extract background ozone for OLM, matching run period
# (5) Copy receptor locations to work directory
# (6) Create aermod.inp: Read template file and subsitute fields
# (7) Run AERMOD
# (8) Check if AERMOD run without errors
#
# Bas Mijling, April-July 2017

import numpy as np
import pandas as pd
#from netCDF4 import Dataset
import datetime
import pytz
import shutil
import os
import sys
import pyproj
#import matplotlib.pyplot as plt
import sys
import struct
import gdal
import math



### SIMULATION PARAMETERS ######################################################################

do_background = True		# Add temporally varying background concentrations from file
do_OLM = True				# Apply NO2/NOx ratio, using ozone limited method (OLM)

# Width of road line emissions (line emissions are internally treated as areas sources)
roadWidth = 10.0

# Emission factors (optionally adjusted with correction factors from correction_factor.py)
# EF_traffic_highway = 0.15 * 1.303	# Typical highway emission factor [g NOx/km]
# EF_traffic_urban = 0.15 * 0.455		# Typical urban emission factor [g NOx/km]
# EF_population = 5e-9 * 0.170		# Emission factor relating population per hectare to emission rate [g NOx/s.m2]

EF_traffic_highway = 2.12	# Typical highway emission factor [g NOx/km]
EF_traffic_urban = 2.12		# Typical urban emission factor [g NOx/km]
#EF_population = 0.000000158		# Emission factor relating population per hectare to emission rate [g NOx/s.m2]
EF_population = 0.00000000158		# Emission factor relating population per hectare to emission rate [g NOx/s.m2]
EF_powerplants = 1

### INPUT FILES ################################################################################

# Work directory (containing the AERMOD executable)
#workDirectory = "/nobackup/users/mijling/aermod"
workDirectory = "/usr/people/patel/retina_delhi/test"

# Template for aermod.inp
templateFile = "./aermod.template.inp"

# Meteo files, produced with AERMET
#meteoSurfaceInput = "/usr/people/mijling/retina/github/data/aermod/aermet/EINDHOVEN_2016.SFC"
#meteoProfileInput = "/usr/people/mijling/retina/github/data/aermod/aermet/EINDHOVEN_2016.PFL"

meteoSurfaceInput = "/usr/people/patel/retina_delhi/test/source/surface/DELHI_2015_2017.SFC"
meteoProfileInput = "/usr/people/patel/retina_delhi/test/source/surface/DELHI_2015_2017.PFL"


# Definition of receptor locations
## LML and AiREAS station locations (by write_station_locations.py)):
#receptorFile = "/usr/people/mijling/retina/github/data/aermod/RE_stations.inc"
# Road following coordinates (by make_roadfollowing_grid.py):
#receptorFile = "/usr/people/mijling/retina/github/data/aermod/RE_roadfollowing_grid.inc"
receptorFile = "/usr/people/patel/retina_delhi/test/source/rgrid/RE_roadfollowing_grid.inc"

# If do_background=True: Specify hourly background concentration file
#backgroundFile = "/usr/people/mijling/retina/github/data/aermod/SO_no2_background_cams.inc"

# If do_OLM=True: OLM settings
ozoneDefaultValue = 48.8
#ozoneFile = "/usr/people/mijling/retina/github/data/aermod/CO_ozone_cams.inc"

### EMISSION FILES #############################################################################

# road segments from OpenStreetMaps, preprocessed with write_street_segments.py
#streetFile_segment = "/usr/people/mijling/retina/github/data/osm/eindhoven_street_segments.csv"

streetFile_segment = "/usr/people/patel/retina_delhi/test/source/streetfile/delhi_street_segments.csv"


# Gridded population from CBS [100m], preprocessed with write_popdensity.py
#populationFile = "/usr/people/mijling/retina/github/data/cbs/eindhoven_population_density_100m.nc"
populationFile = "/usr/people/patel/retina_delhi/test/source/population/smallgtiff.tif"

# Traffic emissions from flow and speed by NDW [vehicle/km] for motorways and primary roads,
# preprocessed with traffic_motorway.py and traffic_primary.py
#trafficMotorwayFile = "/usr/people/mijling/retina/github/data/ndw/traffic_motorway.csv"
#trafficPrimaryFile = "/usr/people/mijling/retina/github/data/ndw/traffic_primary.csv"

trafficMotorwayFile = "/usr/people/patel/retina_delhi/test/source/traffic/traffic_motorway.csv"
trafficPrimaryFile = "/usr/people/patel/retina_delhi/test/source/traffic/traffic_primary.csv"

# Powerplant emissions from DPCC website data + Delhi(2010) report + litreture (Arun Kansal et al.)
powerplantFile = '/usr/people/patel/retina_delhi/test/source/powerplants/location_powerplant.csv'

# Emission files (created in work directory)
emisDefFile = "SO_sources.inc"
emisHourlyFile  = "SO_houremis.inc"

################################################################################################


def aermodDate(dateTime, yyyy=False):
	"""
	Write date and hours in AERMOD convention (hours from 1 to 24)
	Dates are returned as string yymmmddhh if yyyy=False, or yyyymmddhh if yyyy=True
	"""
	yyyymmddhh = datetime.datetime.strftime(dateTime, '%Y%m%d%H')
	hh = yyyymmddhh[8:10]
	if hh=='00':
		yyyymmdd = datetime.datetime.strftime(dateTime - datetime.timedelta(days=1), '%Y%m%d')
		yyyymmddhh = yyyymmdd + '24'
	if yyyy:
		return yyyymmddhh
	else:
		return yyyymmddhh[2:10]



def writeSurfaceMeteo(meteoInput, meteoOutput, timeWindow):
	"""
	Extract from large AERMET surface file only the hours used in this AERMOD run
	"""
	aermodStart = aermodDate(timeWindow[0])
	aermodEnd = aermodDate(timeWindow[-1])

	g = open(meteoOutput, 'w')
	with open(meteoInput) as f:
		n = 0	# Count lines in file
		m = 0	# Count hours written to output
		for line in f:
			if n>0:
				field = line.split()
				metDate = "%02d%02d%02d%02d" % (int(field[0]), int(field[1]), int(field[2]), int(field[4]))
				if (metDate >= aermodStart) and (metDate <= aermodEnd):
					g.write(line)
					m += 1
			else:
				g.write(line)
			n += 1
	if m <> len(timeWindow):
		print "%s : ERROR: not enough time slots available in surface meteo file %s" % (me, meteoInput)
		sys.exit()
	g.close()



def writeProfileMeteo(meteoInput, meteoOutput, timeWindow):
	"""
	Extract from large AERMET profile file only the hours used in this AERMOD run
	"""
	aermodStart = aermodDate(timeWindow[0])
	aermodEnd = aermodDate(timeWindow[-1])

	g = open(meteoOutput, 'w')
	with open(meteoInput) as f:
		m = 0	# Count hours written to output
		for line in f:
			field = line.split()
			metDate = "%02d%02d%02d%02d" % (int(field[0]), int(field[1]), int(field[2]), int(field[3]))
			if (metDate >= aermodStart) and (metDate <= aermodEnd):
				g.write(line)
				m += 1
	if m <> len(timeWindow):
		print "%s : ERROR: not enough time slots available in profile meteo file %s" % (me,meteoInput)
		sys.exit()
	g.close()



def writeHourlyData(fileInput, fileOutput, timeWindow):
	"""
	Extract from hourly data file only the hours used in this AERMOD run
	"""
	aermodStart = aermodDate(timeWindow[0], yyyy=True)
	aermodEnd = aermodDate(timeWindow[-1], yyyy=True)

	g = open(fileOutput, 'w')
	with open(fileInput) as f:
		m = 0	# Count hours written to output
		for line in f:
			if line == '': continue
			field = line.split()
			if field[0] == '**': continue
			metDate = "%02d%02d%02d%02d" % (int(field[0]), int(field[1]), int(field[2]), int(field[3]))
			if (metDate >= aermodStart) and (metDate <= aermodEnd):
				g.write(line)
				m += 1
	if m <> len(timeWindow):
		print "%s : ERROR: not enough time slots available in data file %s" % (me,fileInput)
		sys.exit()
	g.close()



def writeSourceDef_area(f, xGrid, yGrid, valGrid, leadingID):
	"""
	Write definition of AREA sources (only for emissions greater than 0)
	The emission value set in SO SRCPARAM is a dummy value, as AERMOD reads it later from HOUREMIS
	The function returns a dictionairy labelling the non-zero emission cells
	"""
	rotation = 0.0		# No rotation of square emission area
	sigmaZ0 = 2.0		# Initial vertical extension of concentration layer: 2m
	emisHeight = 0.5	# Emission height (tailpipe heigth): 50 cm

	# Built dictionary with grid definition
	grid = {}

	# Grid resolution (25m)
	dx = xGrid[1]-xGrid[0]
	dy = yGrid[1]-yGrid[0]

	nx = len(xGrid)
	ny = len(yGrid)

	# Loop over all cells in grid
	cnt = 0
	for i in range(nx):
		for j in range(ny):

			# Only add source if grid value > 0
			if valGrid[j,i]>0:
				cnt += 1
				x = xGrid[i]
				y = yGrid[j]
				x_sw = x-0.5*dx
				y_sw = y-0.5*dx
				srcID = "%1s%07d" % (leadingID, cnt)
				emis = 1.0

				f.write("SO location %8s area %d %d\n" % (srcID, x_sw, y_sw))
				f.write("SO srcparam %8s %0.3e %0.2f %d %d %3d %0.2f\n" % (srcID, emis, emisHeight, dx, dy, rotation, sigmaZ0))

				# Add new srcID to dictionary, containing grid index pair
				grid[srcID] = (i,j)

	print "%s : %7d area sources found for %s group" % (me, cnt, leadingID)
	return grid



def writeSourceDef_line(f, df, leadingID):
	"""
	Write definition of LINE sources defined in dataframe df
	The emission value set in SO SRCPARAM is a dummy value, as AERMOD reads it later from HOUREMIS
	"""
	sigmaZ0 = 2.0		# Initial vertical extension of concentration layer: 2m
	emisHeight = 0.5	# Emission height (tailpipe heigth): 50 cm

	for srcID, row in df.iterrows():
		emis = 1.0
		f.write("SO location %8s line %d %d %d %d\n" % (srcID, row['x0'], row['y0'], row['x1'], row['y1']))
		f.write("SO srcparam %8s %0.3e %0.2f %0.2f %0.2f\n" % (srcID, emis, emisHeight, roadWidth, sigmaZ0))
	print "%s : %7d line sources found for %s group" % (me, len(df), leadingID)

def writeSourceDef_point(f, df, leadingID):
	"""
	Write definition of POINT sources defined in dataframe df
	The emission value set in SO SRCPARAM is from Delhi 2010 report (2005-2006) for now
	"""
	for idx, row in df.iterrows():
		f.write("SO location %s point %d %d \n" % (row['srcID'], row['x'], row['y']))
		f.write("SO srcparam %s %0.3e %0.2f %0.2f %0.2f %0.2f\n" % (row['srcID'], \
		row['emission_rate'], row['stack_height'], row['exit_temperature'], row['exit_velocity'], row['stack_diameter'] ))
	print "%s : %7d point sources found for %s group" % (me, len(df), leadingID)



def writeHourlyEmis_area(f, dateTime, grid, emis):
	"""
	Write AREA emissions for a given hour and given source group [g NOx/s.m2]
	"""
	yymmddhh = aermodDate(dateTime)
	y = int(yymmddhh[0:2])
	m = int(yymmddhh[2:4])
	d = int(yymmddhh[4:6])
	h = int(yymmddhh[6:8])

	grid_keys = np.sort(grid.keys())
	for srcID in grid_keys:
		i = grid[srcID][0]
		j = grid[srcID][1]
		f.write("SO houremis  %2d %2d %2d %2d %s %0.3e\n" % (y,m,d,h, srcID, emis[j,i]))



def writeHourlyEmis_line(f, dateTime, df, emis):
	"""
	Write LINE emissions for a given hour
	All line sources will be attributed the same area emission rate [g NOx/s.m2]
	"""
	yymmddhh = aermodDate(dateTime)
	y = int(yymmddhh[0:2])
	m = int(yymmddhh[2:4])
	d = int(yymmddhh[4:6])
	h = int(yymmddhh[6:8])

	for srcID in df.index:
		f.write("SO houremis  %2d %2d %2d %2d %s %0.3e\n" % (y,m,d,h, srcID, emis))

def writeHourlyEmis_point(f, dateTime, df, emis):
	"""
	Write POINT emissions for a given hour
	"""
	yymmddhh = aermodDate(dateTime)
	y = int(yymmddhh[0:2])
	m = int(yymmddhh[2:4])
	d = int(yymmddhh[4:6])
	h = int(yymmddhh[6:8])

	for idx,row in df.iterrows():
		f.write("SO houremis  %2d %2d %2d %2d %s %0.3e %0.2f %0.2f\n" % (y,m,d,h, row['srcID'], emis[idx], row['exit_temperature'], row['exit_velocity']))


def readpopulation():
	"""
	Reads GeoTIFF file from the path provided and returns population density, and
	written to grid of size 100 by 100
	"""
	domain = "delhi"
	dataFile  = "/usr/people/patel/retina_delhi/test/source/population/smallgtiff.tif"
	#dataFile  = "C:\Users\Karsh\Documents\Academics\TU Delft\Internship\Popmap\smallgtiff.tif"

	wgs84 = pyproj.Proj("+init=EPSG:4326") 	# Lat/Lon projection
	lat0, lon0 = (28.68, 77.10)		# Delhi centre
	myProj = pyproj.Proj("+proj=sterea +lat_0=%f +lon_0=%f +ellps=bessel +units=m" % (lat0,lon0)) # Projection for Delhi

	# Read raster layer
	print "\n Read population data from geotiff"
	raster = gdal.Open(dataFile, gdal.GA_ReadOnly)

	# Get all coordinates from geotransform
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

	print "\n\t Coordinates of the image file:"
	print "\t\t Upper left:\t",   ULlon, ULlat
	print "\t\t Lower left:\t",   LLlon, LLlat
	print "\t\t Upper Right:\t",  URlon, URlat
	print "\t\t Lower Right:\t",  LRlon, LRlat
	print "\t\t Center     :\t",  CTlon, CTlat

	# Values from cmd line: gdalinfo "filename.tiff" (for verification of above values)
	# ULlon, ULlat=76.8370000, 28.8832000
	# LLlon, LLlat=76.8370000, 28.4059000
	# URlon, URlat=77.3451000, 28.8832000
	# LRlon, LRlat=77.3451000, 28.4059000
	# CTlon, CTlat=77.0910500, 28.6445500

	# Convert coordinates to Delhi projection
	print "\n\t Converting coordinate projection..."
	ULx,ULy = pyproj.transform(wgs84, myProj, ULlon, ULlat)
	LLx,LLy = pyproj.transform(wgs84, myProj, LLlon, LLlat)
	LRx,LRy = pyproj.transform(wgs84, myProj, LRlon, LRlat)
	URx,URy = pyproj.transform(wgs84, myProj, URlon, URlat)
	CTx,CTy = pyproj.transform(wgs84, myProj, CTlon, CTlat)

	# Note: After transforming projection, corners are no longer represent square
	# or rectangle, they need to be rouded off to make them so.

	# Rounding off coordinates
	print "\n\t Rounding off coordinates..."
	ULx, ULy = (int(ULx/1000)*1000, int(ULy/1000)*1000)
	LLx, LLy = (int(LLx/1000)*1000, int(LLy/1000)*1000)
	LRx, LRy = (int(LRx/1000)*1000, int(LRy/1000)*1000)
	URx, URy = (int(math.ceil(float(URx)/1000)*1000), int(URy/1000)*1000)
	CTx, CTy = (int(CTx/100)*100, int(CTy/1000)*1000)

	print "\n\t Transformed coordinates of the image file:"
	print "\t\t Upper left:\t",   ULx, ULy
	print "\t\t Lower left:\t",   LLx, LLy
	print "\t\t Upper Right:\t",  URx, URy
	print "\t\t Lower Right:\t",  LRx, LRy
	print "\t\t Center     :\t",  CTx, CTy

	# Prepare grid
	print "\n\t Preparing the resulting grid..."
	nx = 100                        # Resoltuion of 100m
	ny = 100
	rdx = np.zeros(nx, dtype=np.int)     # Initializing arrays
	rdy = np.zeros(ny, dtype=np.int)

	# Fill RD grid definition
	rdx = np.linspace(LLx, LRx, nx)
	rdy = np.linspace(LLy, ULy, ny)
	#print "RD x y", rdx, rdy
	print "\t\t Grid resolution: \t", len(rdx), "by", len(rdy), "meters"

	# Make grid
	print "\t\t Calculating lat,lon coordinates of grid..."
	xx,yy = np.meshgrid(rdx,rdy)

	# Read band(population density numbers) from raster
	print "\n\t Reading population density from image file..."
	band = raster.GetRasterBand(1)
	print "\t\t Original size of the image:\t", band.XSize, band.YSize

	# Read image in original size
	print "\t\t Storing image in original size as array called dost..."
	dost = band.ReadAsArray(0,0,band.XSize,band.YSize) # No. of cols and no. of rows

	print "\t\t Size of dost:\t", dost.shape
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
	print "\t\t Population density stored as binary in tuples of length:", len(scanline)

	# Converting binary numbers to proper human numbers
	value = struct.unpack('f' * 10000, scanline)
	print "\t\t Population density stored as float in tuples of length:", len(value)

	# Convert tuples to 2D array
	value_array = np.asarray(value)
	re_value_array = np.reshape(value_array,(100,100))
	print "\t\t Population density stored as float in array of size:", re_value_array.shape

	# Plot on PC (linux doesn't support jpeg and gdal simultaneously)
	# plt.imshow(re_value_array)
	# plt.show()

	# Store values as a text file
	np.savetxt('value.txt',value)

	print "\n End of function \n"
	return re_value_array, rdx, rdy


def writeEmissions(sourceDefFile, hourlyEmisFile, timeWindow):
	"""
	Write emission definition file [SO_sources] for 3 categories:
	A: Motorway + Trunk roads (Line sources, 10m width)
	B: Primary + Secondary + Tertiary roads (Line sources, 10m width)
	C: Populaton (Area sources 100 x 100m)
    D: Powerplants (Point sources)
	Afterwards, the emission strength for each source is written for all hours in time window
	"""

	# Read coordinates of road segments for categories A and B
	df_roads = pd.read_csv(streetFile_segment, header=17, index_col='roadType')
	df_roads_A = df_roads[(df_roads.index>=1) & (df_roads.index<=3)].copy()
	df_roads_B = df_roads[(df_roads.index>=4) & (df_roads.index<=5)].copy()

	# Assign unique IDs to road segments
	lineID = ["A%07d" % (i+1) for i in range(len(df_roads_A))]
	df_roads_A['lineID'] = lineID
	df_roads_A.set_index('lineID', inplace=True)
	lineID = ["B%07d" % (i+1) for i in range(len(df_roads_B))]
	df_roads_B['lineID'] = lineID
	df_roads_B.set_index('lineID', inplace=True)

	# Read gridded population
	(populationGrid, xPopulation, yPopulation) = readpopulation()

	# Read powerplant file
	df_pp = pd.read_csv(powerplantFile, skiprows=1)

	# Read traffic flow and speed
	df_traffic_motorway = pd.read_csv(trafficMotorwayFile, index_col='UTC', parse_dates=True, header=0, na_values=[''])
	df_traffic_primary = pd.read_csv(trafficPrimaryFile, index_col='UTC', parse_dates=True, header=0, na_values=[''])
	df_traffic_motorway.index = df_traffic_motorway.index.tz_localize('utc')	# Set index explicitly to UTC
	df_traffic_primary.index = df_traffic_primary.index.tz_localize('utc')
	df_traffic_motorway = df_traffic_motorway.reindex(timeWindow)				# Reindex, filling gaps with NaNs
	df_traffic_primary = df_traffic_primary.reindex(timeWindow)

	# Write definition of area sources (SO LOCATION and SO SRCPARAM)
	f = open(sourceDefFile,'w')
	f.write("** Emission sources for Delhi area on 25m resolution\n")
	f.write("** A: Highways (line)\n")
	f.write("** B: Primary + Secondary + Tertiary roads (line)\n")
	f.write("** C: Population (area)\n")
	f.write("** D: Powerplants (point)\n")
	f.write("** written by %s\n" % sys.argv[0].split('/')[-1])
	writeSourceDef_line(f, df_roads_A, 'A')
	writeSourceDef_line(f, df_roads_B, 'B')
	gridC = writeSourceDef_area(f, xPopulation, yPopulation, populationGrid, 'C')
	writeSourceDef_point(f, df_pp, 'D')
	f.close()
	print "%s : Emission source definition written to %s" % (me, sourceDefFile)


	# Write hourly emission for each area [SO_houremis]
	f = open(hourlyEmisFile,'w')
	print "%s : Writing emissions to %s\n" % (me, hourlyEmisFile),
	for dateTime in timeWindow:

		flowA = df_traffic_motorway.loc[dateTime]['flow']		# Nr of vehicles in one hour [vehicle/h]
		emisA = 0.001 * EF_traffic_highway * flowA / roadWidth	# Road segment emission [g NOx/h.m2]
		emisA /= 3600.											# Road segment emission [g NOx/s.m2], scalar

		flowB = df_traffic_primary.loc[dateTime]['flow']		# Nr of vehicles in one hour [vehicle/h]
		emisB = 0.001 * EF_traffic_urban * flowB / roadWidth	# Road segment emission [g NOx/h.m2]
		emisB /= 3600.											# Road segment emission [g NOx/s.m2], scalar

		emisC = EF_population*populationGrid					# Residential emission  [g NOx/s.m2], 2D array
		emisD = EF_powerplants*df_pp['emission_rate']
		
		 #When flowA of flowB is NaN this indicates missing NDW data
		if (not np.isfinite(flowA)) or (not np.isfinite(flowB)):
			print "%s : WARNING: no valid NDW data for %s. Put all emission to zero!" % (me, dateTime)
			emisA = 0.0
			emisB = 0.0
			emisC = np.zeros(emisC.shape)
			emisD = np.zeros(emisD.shape)

		writeHourlyEmis_line(f, dateTime, df_roads_A, emisA)
		writeHourlyEmis_line(f, dateTime, df_roads_B, emisB)
		writeHourlyEmis_area(f, dateTime, gridC, emisC)
		writeHourlyEmis_point(f, dateTime, df_pp, emisD)
	f.close()


### MAIN ##########################################################################################

me = os.path.basename(__file__)

# Get time window [UTC] to be considered from arguments
if len(sys.argv)<>3:
	print "%s : Run script with start hour and end hour <yyyymmddhh> <yyyymmddhh>" % me
	sys.exit()
arg1 = sys.argv[1]
arg2 = sys.argv[2]
startDate = arg1[0:4]+'-'+arg1[4:6]+'-'+arg1[6:8]+' '+arg1[8:10]+':00'
endDate = arg2[0:4]+'-'+arg2[4:6]+'-'+arg2[6:8]+' '+arg2[8:10]+':00'

timeWindow = pd.date_range(startDate, endDate, freq='H', tz=pytz.UTC)

runStart = aermodDate(timeWindow[0], yyyy=True)
runEnd = aermodDate(timeWindow[-1], yyyy=True)
print "%s : Preparing AERMOD run for period %s - %s" % (me, runStart, runEnd)
print "%s : Start at %s" % (me, datetime.datetime.utcnow().strftime("%d-%m-%Y, %H:%M:%S UTC"))


print "%s : Work directory is %s" % (me, workDirectory)

if not do_background:
	print "%s : WARNING: No background concentrations for NO2 are added" % me

if not do_OLM:
	print "%s : WARNING: OLM is not used for NO2/NOx ratio calculation" % me


# (1) Extract meteo
print "%s : Extract surface meteo" % me
meteoSurface = workDirectory + '/' + meteoSurfaceInput.split('/')[-1]
writeSurfaceMeteo(meteoSurfaceInput, meteoSurface, timeWindow)

print "%s : Extract profile meteo" % me
meteoProfile = workDirectory + '/' + meteoProfileInput.split('/')[-1]
writeProfileMeteo(meteoProfileInput, meteoProfile, timeWindow)


# (2) Prepare emissions
print "%s : Prepare hourly emissions" % me
writeEmissions(workDirectory + '/' + emisDefFile, workDirectory + '/' + emisHourlyFile, timeWindow)


# (3) Extract background concentration, matching run period
if do_background:
	print "%s : Extract background concentration" % me
#	backgroundOutput = workDirectory + '/' + backgroundFile.split('/')[-1]
#	writeHourlyData(backgroundFile, backgroundOutput, timeWindow)


# (4) Extract background ozone for OLM, matching run period
if do_OLM:
	print "%s : Extract background ozone" % me
#	ozoneOutput = workDirectory + '/' + ozoneFile.split('/')[-1]
#	writeHourlyData(ozoneFile, ozoneOutput, timeWindow)


# (5) Copy receptor locations to work directory
shutil.copy(receptorFile, workDirectory+'/'+receptorFile.split('/')[-1])


# (6) Create armod.inp: Read template and subsitute fields.
#     Remove directory paths from include files, as all files are written in the work directory
print "%s : Create aermod.inp from template %s" % (me,templateFile)
g = open(workDirectory + '/aermod.inp', 'w')
with open(templateFile) as f:
	for line in f:

		if "[ME_surface]" in line:
			line = line.replace('[ME_surface]', meteoSurface.split('/')[-1])

		if "[ME_profile]" in line:
			line = line.replace('[ME_profile]', meteoProfile.split('/')[-1])

		if "[RUN_start]" in line:
			line = line.replace('[RUN_start]', runStart[0:4]+' '+runStart[4:6]+' '+runStart[6:8]+' '+runStart[8:10])

		if "[RUN_end]" in line:
			line = line.replace('[RUN_end]', runEnd[0:4]+' '+runEnd[4:6]+' '+runEnd[6:8]+' '+runEnd[8:10])

#		if do_background:
#			if "[SO_backgrnd]" in line:
#				line = line.replace('[SO_backgrnd]', backgroundOutput.split('/')[-1])
		else:
			if "[SO_backgrnd]" in line:
				line = ''
			if "SO backunit" in line:
				line = ''
			if "background" in line:					# Remove 'background' keyword in 'SO srcgroup  all'
				line = line.replace('background', '')

		if do_OLM:
			if "[CO_olm]" in line:
				line = line.replace('[CO_olm]', 'olm')
#			if "[CO_ozonefil]" in line:
#				line = line.replace('[CO_ozonefil]', ozoneOutput.split('/')[-1])
			if "[CO_ozoneval]" in line:
				line = line.replace('[CO_ozoneval]', str(ozoneDefaultValue))
		else:
			if "[CO_olm]" in line:
				line = line.replace('[CO_olm]', '')
			if "CO ozonefil" in line:
				line = ''
			if "CO ozoneval" in line:
				line = ''
			if "SO no2ratio" in line:
				line = ''
			if "SO olmgroup" in line:
				line = ''

		if "[SO_sources]" in line:
			line = line.replace('[SO_sources]', emisDefFile)

		if "[SO_houremis]" in line:
			line = line.replace('[SO_houremis]', emisHourlyFile)

		if "[RE_stations]" in line:
			line = line.replace('[RE_stations]', receptorFile.split('/')[-1])

		if "[OU_file]" in line:
			line = line.replace('[OU_file]', 'conc_%s-%s' % (runStart,runEnd))


		if line<>'':
			g.write(line)
g.close()


# (7) Run AERMOD
print "%s : Run AERMOD" % me
command = "cd %s; ./aermod" % workDirectory
os.system(command)


# (8) Check if AERMOD crashed by looking for keyword 'UN-successfully' in tail of aermod.out
f = open("%s/aermod.out" % workDirectory, "r" )
lineList = f.readlines()
f.close()
lastLines = lineList[-4:]
if 'UN-successfully' in ''.join(lastLines):
	print '%s : ERROR: AERMOD crashed. Keep aermod.out file' % (me)
	command = "mv %s/aermod.out %s/aermod.out.%s-%s" % (workDirectory, workDirectory, arg1, arg2)
	os.system(command)

print "%s : End at %s" % (me, datetime.datetime.utcnow().strftime("%d-%m-%Y, %H:%M:%S UTC"))
