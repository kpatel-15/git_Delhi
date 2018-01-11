import glob
import time
import json
import datetime
#from datetime import datetime
import pandas as pd
import sys
import math
import numpy as np
import ntpath
import ast
import matplotlib.pyplot as plt
from matplotlib.figure import Figure                       
from matplotlib.axes import Axes                           
from matplotlib.lines import Line2D                        
from matplotlib.backends.backend_agg import FigureCanvasAgg

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx


# Time the execution of code
start_time = time.time()


# here app_id = ZueRJM2if1GktTKaeQHF
# here app_code = kcCwlrO_DXCmtx7Xz-SQrg

# Read traffic database
sourcepath = '/nobackup/users/patel/Traffic'
#outputFile = sourcepath + '/bing_traffic.csv'

# Finding specific roads
#f = open(outputFile,'w')
#f.write("# Traffic information for specific roads from HERE api. \n")
#f.write("idx, shp, free_flow_speed, item_code, item_description, item_length, jam_factor, speed, timestamp \n")
#roadname = ['Outer Ring Road', 'Inner Ring Road','Ring', 'NH-1', 'NH-2', 'NH-8',\
#                'NH-10', 'NH-24', 'NH-91', 'NH-58', 'NH']

outputFile2 = sourcepath + '/bing_shape_name.txt'
read_df = pd.read_csv(outputFile2)
read_df = read_df.loc[:, ~read_df.columns.str.contains('^Unnamed')]
big_df = pd.DataFrame()

def roundTime(dt=None, roundTo=1800):
    """
    Round a datetime object to any time laps in seconds
    dt : datetime.datetime object, default now. NOTE: convert Timestamp objects first to datetime with .to_pydatetime()
    roundTo : Closest number of seconds to round to, default 10 minutes.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
    http://stackoverflow.com/questions/3463930/how-to-round-the-minute-of-a-datetime-object-python
    """
    if dt == None : dt = datetime.datetime.now()
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)



hourly_avg = []
for name in (glob.glob(sourcepath + '/special//traffic_*')): #1511861720 1514982661
    print('\n Reading file: \t', name)
    try:
        json_data = json.load(open(name))
        
        # Extremeties
    
        # json data has 5 keys but only created timestamp and RWS is important here.
        # Only these two keys will be extracted.
        
        time_stamp = json_data['CREATED_TIMESTAMP'] # Time for which traffic information is accessed.
        #Try1
        in_fun = (pd.to_datetime(time_stamp)).to_pydatetime()
        out_fun = roundTime(in_fun, 1800)
      
        if out_fun.minute == 0:
            print ('\n Accepted file %s with timestamp %s',  ntpath.basename(name), str(out_fun))
            top_DE = []
            top_LI = []
            le = []
            de = []
            pc = []
            ff = []
            sp = []
            jf = []
            sh = []
            tt = []
            for a in range(len(json_data['RWS'])):
                for b in range(len(json_data['RWS'][a]['RW'])):
                    for c in range(len(json_data['RWS'][a]['RW'][b])):
                        # Top branch of the traffic tree
                        top_DE.append(json_data['RWS'][a]['RW'][b]['DE']) # Description of top branch (Bigger road segment I think)
                        top_LI.append(json_data['RWS'][a]['RW'][b]['LI']) # Linear code of top branch
                        # Penetrating deeper into FIS
                        for d in range(len(json_data['RWS'][a]['RW'][b]['FIS'])):
                            # Going to FI level 
                            for e in range(len(json_data['RWS'][a]['RW'][b]['FIS'][d]['FI'])):
                                #print ('\n anu', anu)
                                # Saving TMC information - Spatial description
                                le.append(json_data['RWS'][a]['RW'][b]['FIS'][d]['FI'][e]['TMC']['LE']) # Length of item
                                de.append(json_data['RWS'][a]['RW'][b]['FIS'][d]['FI'][e]['TMC']['DE']) # Decription of item
                                pc.append(json_data['RWS'][a]['RW'][b]['FIS'][d]['FI'][e]['TMC']['PC']) # Code of item
                                # SHP is a list. Could be longer. Added here as a single list containing multiple list.
                                try:
                                    sh.append(json_data['RWS'][a]['RW'][b]['FIS'][d]['FI'][e]['SHP'])
                                except:
                                    sh.append(float('NaN'))
                                    #sh.append(read_df['coordinates'][anu])
                                # CF is a list. Just a precaution incase the list is longer
                                for fcount in range(len(json_data['RWS'][a]['RW'][b]['FIS'][d]['FI'][e]['CF'])):
                                    # Saving CF information - speed and jam
                                    ff.append(json_data['RWS'][a]['RW'][b]['FIS'][d]['FI'][e]['CF'][fcount]['FF'])
                                    sp.append(json_data['RWS'][a]['RW'][b]['FIS'][d]['FI'][e]['CF'][fcount]['SP'])
                                    jf.append(json_data['RWS'][a]['RW'][b]['FIS'][d]['FI'][e]['CF'][fcount]['JF'])
                                    tt.append(json_data['CREATED_TIMESTAMP'])

            new_dict = {'item_description':de, 'item_code':pc, 'item_length':le, \
            'free_flow_speed':ff, 'speed':sp, 'jam_factor':jf, 'coordinates':sh, 'timestamp':tt}
            df = pd.DataFrame.from_dict(new_dict)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            tstamp = pd.DatetimeIndex(df['timestamp'])
            
            # Roundoff time
            roundt = []
            for i in range(len(tstamp)):
                in_fun = tstamp[i].to_pydatetime()
                out_fun = roundTime(in_fun, 1800)
                roundt.append(out_fun)
            
            df['roundtime'] = roundt
            
            temp = pd.DatetimeIndex(df['roundtime'])
            df['date'] = temp.date
            df['time'] = temp.time
            del df['timestamp']
            df.set_index('coordinates', inplace=True)
            df.drop_duplicates(inplace=True)
            df['coordinates'] = df.index
            #df.to_csv(f, mode='a', header=False)
            #f.close()
            
            
            # Hourly average for all line segments
            hourly_avg.append(df)
#            hp.to_csv(sourcepath + '/hourly_pulse.csv')
    except:
        print('\n This file was empty. Moving to next file.')
        pass

hourly_avg = pd.concat(hourly_avg, axis=0)


# Average by time
hpulse = hourly_avg.groupby(['time', 'item_description', 'item_length']).mean()
#Index has important information. Place it in columns.
hpulse['based'] = hpulse.index
hpulse[['time','item_description', 'item_length']] = pd.DataFrame(hpulse.based.values.tolist(), index= hpulse.index)
read_df.set_index('coordinates', inplace=True)
read_df.drop_duplicates(inplace=True)
read_df['coordinates'] = read_df.index
hp = pd.merge(hpulse, read_df, on =['item_length','item_description'])


# Plot code
#fig = Figure(figsize=[14,14])
#ax = Axes(fig, [.1,.1,.8,.8])
#
#jet = cm = plt.get_cmap('jet') 
#cNorm  = colors.Normalize(vmin=0, vmax=10)
#scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
#print (scalarMap.get_clim())
#
##hrbox = list(range(24))
#hrbox = [14]
#minbox = [00]
#for x in hrbox:
#    auto_time = datetime.time(x,00)
#    print ('autotime', auto_time)
#    h = hp[hp['time'] == auto_time]
#    h.reset_index(inplace=True)
#    for i in range(len(h['coordinates'])):
#        print ('i',i)
#        big_box = ast.literal_eval(h.coordinates[i])
#        for j in range(len(big_box)):
#            box = big_box[j]['value']
#            box = box[0].strip()
#            #box = box.encode('utf-8')
#            bs = box.split(' ')
#            bss = [b.split(',') for b in bs]
#            ln = [tuple(float(n) for n in bss[i]) for i in range(len(bss))]
#            
#            line1_xs = []
#            line1_ys = []
#            (line1_xs, line1_ys) = zip(*ln)
#               
#            fig.add_axes(ax)
#            colorVal = scalarMap.to_rgba(h['jam_factor'][i])
#            ax.add_line(Line2D(line1_xs, line1_ys, linewidth=2, color=(colorVal)))
#            ax.set_xlim(28.35,28.85)
#            ax.set_ylim(76.8,77.6)
#            ax.set_title(str(auto_time))
#            #ax.annotate(str(df['jam_factor'][i]), (line1_xs, line1_ys))
#            canvas = FigureCanvasAgg(fig)                              
#    canvas.print_figure(sourcepath + '/Images/' + ' jam_factor average ' + str(auto_time) + "_hourly_pulse.png")  


##Write shp coordinates in a file so they can be used when not avialable
#outputFile2 = sourcepath + '/bing_shape_name.txt'
#df.to_csv(outputFile2, columns=['coordinates', 'item_description', 'item_length'])

print("--- %s seconds ---" % (time.time() - start_time))
sys.exit()

