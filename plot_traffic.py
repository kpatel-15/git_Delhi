#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 12:26:49 2018
Script to plot relation between traffic parameters obtained from bing maps.
@author: patel
"""
import pandas as pd
import matplotlib.pyplot as plt

# Path 
sourcepath = '/nobackup/users/patel/Traffic'

# Read hourly average pulse
df = pd.read_csv(sourcepath + '/bing_traffic.csv', skiprows=2)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Plot Speed vs Free flow speed
fig, ax = plt.subplots()
plt.scatter(df.speed, df.free_flow_speed, c=df.jam_factor)
plt.title('Speed vs Free flow speed')
plt.xlabel('Speed (km/hr)')
plt.ylabel('Free flow speed (km/hr)')
plt.colorbar()
#plt.show()
plt.savefig(sourcepath + '/Speed vs Free flow speed.png')

# Plot Jam factor vs Speed ratio
fig, ax = plt.subplots()
plt.scatter(df.jam_factor, df.free_flow_speed/df.speed, c=df.jam_factor)
plt.title('Jam factor vs Speed ratio')
plt.xlabel('Jam factor')
plt.ylabel('Speed ratio (Free flow speed/speed)')
#plt.show()
plt.savefig(sourcepath + '/Jam factor vs Speed ratio.png')

# Plot Jam factor vs Speed
fig, ax = plt.subplots()
plt.scatter(df.jam_factor, df.speed, marker='o', facecolors='none', edgecolors='r')
plt.title('Jam factor vs Speed')
plt.xlabel('Jam factor')
plt.ylabel('Speed (km/hr)')
#plt.show()
plt.savefig(sourcepath + '/Jam factor vs Speed.png')

# Plot Jam factor vs Free flow speed
fig, ax = plt.subplots()
plt.scatter(df.jam_factor, df.free_flow_speed, marker='o', facecolors='none', edgecolors='r')
plt.title('Jam factor vs Free Flow speed')
plt.xlabel('Jam factor')
plt.ylabel('Free Flow speed (km/hr)')
#plt.show()
plt.savefig(sourcepath + '/Jam factor vs Free Flow speed.png')
