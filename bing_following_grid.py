#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 15:20:46 2018
This file makes road following grid along roads from bing maps.
Code is modified from OSM's road following grid.
@author: patel
"""

#!/nobackup/users/mijling/miniconda2/bin/python
#
# Produce road following grid:
# Read roads from OpenStreetMap data file
# Determine location of receptor points with equal distances to road, using Shapely
# Add regular background grid
# Write AERMOD receptor definition file
# Plot results
#
# Bas Mijling, May 2017

import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import cascaded_union, transform
#from descartes import PolygonPatch
import matplotlib.pyplot as plt
import sys
import ast
import time
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Time the execution of code
start_time = time.time()


# AERMOD receptor definition file
outputFile = 'RE_roadfollowing_grid.inc'

# Select bing input file with shape coordinates information
bingFile = '/nobackup/users/patel/Segments//bing_shape.csv'

# Highway classes http://wiki.openstreetmap.org/wiki/Key:highway
# highway = motorway		fast, restricted access road
# highway = trunk			most important in standard road network
# highway = primary			...down to...
# highway = secondary		...
# highway = tertiary		...
# highway = unclassified	least important in standard road network
# highway = residential		smaller road for access mostly to residential properties
# highway = service			smaller road for access, often but not exclusively to non-residential properties

# Select all streets
#highwayClasses = set(["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential"])

# Select main arteries
#highwayClasses = set(["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link"])

# Select super-main arteries
#highwayClasses = set(["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",  "secondary", "secondary_link"])

# Select only primary
#highwayClasses = set(["motorway", "motorway_link",  "trunk", "trunk_link", "primary", "primary_link",  "secondary", "secondary_link"])


# Grid distance settings by dictionary:
# Key indicates distance perpendicular to road, Value indicates distance parallel to road
#distances = {25:70, 50:100, 100:200}
#distances = {25:70, 50:100}
distances = {250:2000}

# Resolution of regular background grid
bg_resolution = 3000


def equidistantCoords(coords, dist):
    """
    Divide line strings (Shapely object) defined by coords in equidistant steps dist
    """

    # Get coordinates
    n = len(coords)
    x,y = coords.xy

    # Write result coordinates in lists
    rx = []
    ry = []

    # Set offset to dist to include starting coordinate of first line segment
    offset = dist

    # Loop over all line segments
    for i in range(n-1):

        # Get coordinates of line segment
        x0,y0 = (x[i], y[i])
        x1,y1 = (x[i+1], y[i+1])

        # Length of segment
        length = np.sqrt((x1-x0)**2 + (y1-y0)**2)
#        print ('\n Length', length)
#        print ('offset', offset)
#        print ('length + offset', length+offset)
#        print ('dist', dist)
        # Skip if segment is too short to contain a coordinate
        if (length+offset) < dist:
            offset += length
            continue

        # Loop over segment with steps of dist, parametrized by l
        # l=0 is start of segment, l=length is end of segment
        l = dist-offset
        
        while (l<=length):
            rx.append(x0 + l/length * (x1-x0))
            ry.append(y0 + l/length * (y1-y0))
            l += dist
            
            # Calculate new offset for next segment
            offset = length-(l-dist)

    
    # Return results
    return rx,ry



print ("Read bing data from %s", bingFile)

# Droped unamed column and duplicates to resduce size
df_bing = pd.read_csv(bingFile)
df_bing = df_bing.loc[:, ~df_bing.columns.str.contains('^Unnamed')]
df_bing.set_index('coordinates', inplace=True)
df_bing.drop_duplicates(inplace=True)
df_bing['coordinates'] = df_bing.index


# Access the coordinates and store in the list
k2 = []
k4 = []
for i in range(len(df_bing['coordinates'])):
    # Extract coordinates from string
    big_box = ast.literal_eval(df_bing.coordinates[i])
    k2 = []
    for j in range(len(big_box)):
        box = big_box[j]['value']
        box = box[0].strip()
        bs = box.split(' ')
        bss = [b.split(',') for b in bs]
        # Convert coordinates to list of tuples
        ln = [tuple(float(n) for n in bss[m]) for m in range(len(bss))]           
        k2.extend(ln)
    k4.append(k2)
del k4[10340:]
###############################################################################
                        #****************************#
###############################################################################

# Plot coordinates to check if they represent the road segments as a check (optional)
#from matplotlib.figure import Figure                       
#from matplotlib.axes import Axes                           
#from matplotlib.lines import Line2D                        
#from matplotlib.backends.backend_agg import FigureCanvasAgg
#fig = Figure(figsize=[14,14])                                
#ax = Axes(fig, [.1,.1,.8,.8])        
#for i in [10341,10342, 10343]:#range(10341):#(len(k)):
##for i in range(10):
#    (line1_xs, line1_ys) = zip(*k4[i])
#    
#    fig.add_axes(ax)     
#    ax.add_line(Line2D(line1_xs, line1_ys, linewidth=2, color='b'))
#    ax.set_xlim(28.35,28.85)
#    ax.set_ylim(76.8,77.6)
#    #ax.annotate(str(df['jam_factor'][i]), (line1_xs, line1_ys))
#    canvas = FigureCanvasAgg(fig)                              
#canvas.print_figure("/nobackup/users/patel/Segments/line_ex.png")  
#sys.exit()

###############################################################################
                        #****************************#
###############################################################################



# Code for projection change
wgs84 = pyproj.Proj("+init=EPSG:4326") 	# Lat/Lon projection
lat0, lon0 = (28.68, 77.10)		# Delhi centre
myProj = pyproj.Proj("+proj=sterea +lat_0=%f +lon_0=%f +ellps=bessel +units=m" % (lat0,lon0)) # Projection for Delhi

# Change projection of list of saved coordinates (club all xs and ys toghether)
knew = []
for i in range(len(k4)):
    knew.append(pyproj.transform(wgs84,myProj,list(zip(*k4[i]))[1], list(zip(*k4[i]))[0]))


###############################################################################
                        #****************************#
###############################################################################



# Bounding box of grid
temp_xmin, temp_ymin = pyproj.transform(wgs84,myProj,76.8,28.35)
temp_xmax, temp_ymax = pyproj.transform(wgs84,myProj,77.6,28.9)

# Plot coordinates to see how the segments appear after changing projection
#from matplotlib.figure import Figure                       
#from matplotlib.axes import Axes                           
#from matplotlib.lines import Line2D                        
#from matplotlib.backends.backend_agg import FigureCanvasAgg
#fig = Figure(figsize=[14,14])                                
#ax = Axes(fig, [.1,.1,.8,.8])        
#for i in range(100000):#(len(knew)):
##for i in range(10):
#    #(line1_xs, line1_ys) = zip(*knew[i])
#    (line1_xs, line1_ys) = knew[i]
#    
#    fig.add_axes(ax)     
#    ax.add_line(Line2D(line1_xs, line1_ys, linewidth=2, color='b'))
#    ax.set_xlim(temp_xmin,temp_xmax)
#    ax.set_ylim(temp_ymin,temp_ymax)
#    #ax.annotate(str(df['jam_factor'][i]), (line1_xs, line1_ys))
#    canvas = FigureCanvasAgg(fig)                              
#canvas.print_figure("/nobackup/users/patel/Segments/line_knew.png")  
#sys.exit()
#
#


###############################################################################
                        #****************************#
###############################################################################

# Convert formate of coordinates from clubbed (x1,x2,...)(y1,y2,...) to shp style
# (x1,y1)(x2,y2)...
                        
for i in range(len(knew)):
    knew[i] = list(zip(*knew[i]))

# Convert list of projection trnasformed coordinates to shapely line segments
segment = MultiLineString(knew)
lines = segment#len(segment)/1000]

   
# Loop over configured cross distances. Get polygons and collect coordinate pairs at equidistant steps on outlines
all_x = []
all_y = []
grey_lines = []

count = 0
for cross_distance in distances:
    print("Calculate grid points at %dm distance" % cross_distance)

    # Dilate all segments for this distance
    polygons = [line.buffer(cross_distance, cap_style=1, resolution=4) for line in lines]
    
    # Merge to polygons. This returns a MultiPolygon or a Polygon object
    polygon = cascaded_union(polygons)

    # Convert Polygon to MultiPolygon object with one member (avoids problems in for ... in ... statement)
    if polygon.geom_type=='Polygon':
        polygon = [polygon]

    # Loop over polygons in MultiPolygon object
    for poly in polygon:
        
        # Get coordinates of exterior and plot
        coords = poly.exterior.coords
        grey_lines.append(coords)

        x,y = equidistantCoords(coords, distances[cross_distance])
        
        all_x.extend(x)
        all_y.extend(y)

        # Get coordinates of interiors and plot
        for i in range(len(poly.interiors)):
            coords = poly.interiors[i].coords
            grey_lines.append(coords)

            x,y = equidistantCoords(coords, distances[cross_distance])
            all_x.extend(x)
            all_y.extend(y)

road_x = np.array(all_x)
road_y = np.array(all_y)
print("%d road following receptor points found" % len(road_x))
print("Adding background grid at %dm resolution" % bg_resolution)

# Get domain boundaries, based on boundary box of roads
# Shift grid such that background grid boxes are on multiples of background resolution
# Background receptor points are at centre of grid boxes


xMin = np.ceil(temp_xmin / bg_resolution) * bg_resolution
xMax = np.floor(temp_xmax / bg_resolution) * bg_resolution
yMin = np.ceil(temp_ymin / bg_resolution) * bg_resolution
yMax = np.floor(temp_ymax / bg_resolution) * bg_resolution


# Loop over background grid points. Only add point if no road receptor points are close to its grid box centre
bg_x = np.array([])
bg_y = np.array([])
for y in np.arange(yMin+0.5*bg_resolution, yMax, bg_resolution):
	for x in np.arange(xMin+0.5*bg_resolution, xMax, bg_resolution):
		d2 = (road_x-x)**2 + (road_y-y)**2
		if np.min(d2) > (0.5*bg_resolution)**2:
			bg_x = np.append(bg_x,x)
			bg_y = np.append(bg_y,y)
print("%d background receptor points added" % len(bg_x))

grid_x = np.append(road_x, bg_x)
grid_y = np.append(road_y, bg_y)
print("Total receptor points: %d" % len(grid_x))

# Write result to AERMOD receptor file
f = open(outputFile, 'w')
f.write("** Road following receptor grid based on bing file %s\n" % bingFile.split('/')[-1])
f.write("** Produced by %s\n" % sys.argv[0].split('/')[-1])
f.write("** Perpendicular distances [m]: %s\n" % ', '.join(str(k) for k in list(distances.keys())))
f.write("** Parellel distances [m]: %s\n" % ', '.join(str(k) for k in list(distances.values())))
f.write("** Resolution regular background grid: %dm\n" % bg_resolution)
for coord in zip(grid_x,grid_y):
	f.write("RE disccart %0.1f %0.1f\n" % (coord[0], coord[1]))
f.close()
print("AERMOD receptor definition written to %s" % outputFile)


print("Plot roads and grid points")



fig = Figure(figsize=[14,14])
ax = Axes(fig, [.1,.1,.8,.8])

# Add coordinates as lines to the plot
for i in range(len(knew)):
#for i in range(10):
   (line1_xs, line1_ys) = zip(*knew[i])
   #(line1_xs, line1_ys) = knew[i]

   fig.add_axes(ax)
   ax.add_line(Line2D(line1_xs, line1_ys, linewidth=2, color='k'))
   ax.set_xlim(temp_xmin,temp_xmax)
   ax.set_ylim(temp_ymin,temp_ymax)

# Plot road following points in red
ax.scatter(road_x,road_y, color='red', s=3)

# Plot background points in green
ax.scatter(bg_x,bg_y, color='green', s=3)

ax.set_xlim(temp_xmin,temp_xmax)
ax.set_ylim(temp_ymin,temp_ymax)
ax.set_aspect('equal')
canvas = FigureCanvasAgg(fig)
canvas.print_figure("/nobackup/users/patel/Segments/final_grid.png")

plt.show()
plt.savefig('/nobackup/users/patel/Segments/finalgrid.png')

print("--- %s seconds ---" % (time.time() - start_time))
