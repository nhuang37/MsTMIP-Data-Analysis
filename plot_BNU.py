import sys
import numpy as np
import pandas as pd
from numpy import meshgrid
import netCDF4
import matplotlib.pyplot as plt
from matplotlib import colors as c
from netCDF4 import num2date, date2num, date2index
from mpl_toolkits.basemap import Basemap, shiftgrid
import datetime
import time

from matplotlib import rcParams
import skill_metrics as sm
from sys import version_info

from os import listdir
from os.path import isfile, join
import collections

#helper function to subset kenya & tanzania region by area lon/lat index:
def slice(latbounds, lonbounds, latt, lonn):	
	# latitude lower and upper index
	latli = np.argmin( np.abs( latt - latbounds[0] ) )
	latui = np.argmin( np.abs( latt - latbounds[1] ) ) 

	# longitude lower and upper index
	lonli = np.argmin( np.abs( lonn - lonbounds[0] ) )
	lonui = np.argmin( np.abs( lonn - lonbounds[1] ) )  
	return latli, latui, lonli, lonui

#helper function for multi_model mean files and subset 1982-2010 (-348) /2000-2010 (-132)
def multi_model_mean_slice(path):
	mean = np.loadtxt(path).reshape((372,360,720))
	mean_sub = np.concatenate([mean[-132:,latli_k:latui_k ,lonli_k:lonui_k].flatten(), mean[-132:,latli_t:latui_t ,lonli_t:lonui_t].flatten()])
	mean_sub[mean_sub < 0] = 0 
	return mean_sub.flatten()

def main():
	#import actual observation dataframe (BNU product)
	f = netCDF4.Dataset('BNU_2000-2010.nc4')
	LAI = f.variables['LAI']
	latt,lonn = f.variables['latitude'][:], f.variables['longitude'][:]

	#slice out Kenya
	latli_k, latui_k, lonli_k, lonui_k = slice([-4.75 , 4.75], [33.25, 41.75], latt, lonn)
	#slice out Tanzania
	latli_t, latui_t, lonli_t, lonui_t = slice([ -11.75 , -1.25 ], [ 28.25 , 40.75 ], latt, lonn)


	#observation slice by concatenating Kenya & Tanzania
	fSubset = np.concatenate ([ (f.variables['LAI'][ : , latli_k:latui_k , lonli_k:lonui_k ] ).flatten(),
								(f.variables['LAI'][ : , latli_t:latui_t , lonli_t:lonui_t ] ).flatten()])

	#import all BG1 model dataframe, subset to 2000-2010
	model_dict = {}
	mypath = '/Users/nhuang37/Desktop/NYU DS/Yr1 Summer/Data/LAI/BG1'
	pathlist = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('nc4')]
	for file in pathlist:
	    print(file)
	    name = file.split('_')[0] + file.split('_')[-1] #tidy up the name
	    f_model = netCDF4.Dataset(mypath+str('/')+file) 
	    f_model.set_auto_mask(False) #missing value = -9999.0
	    # subset 2000-2010
	    time = f_model.variables['time']
	    dates = num2date(time[:], time.units)
	    start_date = datetime.datetime(2000, 1, 1)
	    f_model_sub = np.concatenate ([f_model.variables['LAI'][np.where(dates[dates> start_date])[0], latli_k:latui_k , lonli_k:lonui_k ].flatten(),
	    							f_model.variables['LAI'][np.where(dates[dates> start_date])[0], latli_t:latui_t , lonli_t:lonui_t ].flatten()]) 
	    #set missing value to 0
	    f_model_sub[f_model_sub == -9999.0] = 0 
	    model_dict[name] = f_model_sub

	# add multi-model mean slice to the model dictionary
	model_dict['BG1_mean'] = multi_model_mean_slice('/Users/nhuang37/Desktop/NYU DS/Yr1 Summer/Data/LAI/BG1/BG1_mean.txt')
	model_dict['SG3_mean'] = multi_model_mean_slice('/Users/nhuang37/Desktop/NYU DS/Yr1 Summer/Data/LAI/BG1/SG3_mean.txt')

	# add LAI3G (another actual observation product) for comparison
	LAI3G = netCDF4.Dataset('/Users/nhuang37/Desktop/NYU DS/Yr1 Summer/Data/LAI/Taylor Diagram/LAI3G_regrid.nc')
	# subset 2000-2010 (-132), Kenya & Tanzania
	model_dict['LAI3G'] = np.concatenate ([LAI3G.variables['LAI'][-132:, latli_k:latui_k , lonli_k:lonui_k ].flatten(),
								LAI3G.variables['LAI'][-132:, latli_t:latui_t , lonli_t:lonui_t ].flatten()]) / 1000 #divide by 1000 to rescale

	# Make the Taylor Diagram
	# Set the figure properties (optional)
	rcParams["figure.figsize"] = [8.0, 6.4]
	rcParams['lines.linewidth'] = 1 # line width for plots
	rcParams.update({'font.size': 8}) # font size of axes text

	# Calculate statistics for Taylor diagram
	# The first array element (e.g. taylor_stats1[0]) corresponds to the 
	# reference series while the second and subsequent elements
	# (e.g. taylor_stats1[1:]) are those for the predicted series.
	for i,model in enumerate(model_dict):
		#remove all 0 elements in model & ref (0 represents ocean grid cells, not useful for LAI correlation)
		taylor_stats = sm.taylor_statistics(model_dict[model][fSubset>0],fSubset[fSubset>0]) 
		if i == 0:
			sdev, crmsd, ccoef = [taylor_stats['sdev'][0]], [taylor_stats['crmsd'][0]],[taylor_stats['ccoef'][0]]
		sdev.append(taylor_stats['sdev'][1])
		crmsd.append(taylor_stats['crmsd'][1])
		ccoef.append(taylor_stats['ccoef'][1])
	sdev, crmsd, ccoef = np.array(sdev), np.array(crmsd), np.array(ccoef)

	#get labels
	label = list(model_dict.keys())
	label.insert(0, 'obs') 
	#sort by correlation ccoef
	result = sorted(zip(label,sdev,crmsd,ccoef), key=lambda x: x[3], reverse=True)
	#unzip the result with sorted order
	label,sdev,crmsd,ccoef = zip(*result) 
	sdev, crmsd, ccoef = np.array(sdev), np.array(crmsd), np.array(ccoef)
	#print out the result
	print(label, ccoef)

	#plot Taylor Diagram
	sm.taylor_diagram(sdev,crmsd,ccoef,markerLabel = list(label), markerLabelColor = 'r', 
	                      markerLegend = 'on', markerColor = 'r',
	                      styleOBS = '-', colOBS = 'r', markerobs = 'o',
	                      markerSize = 6, tickRMS = [0.0, 1.0, 2.0, 3.0],
	                      tickRMSangle = 115, showlabelsRMS = 'on',
	                      titleRMS = 'on', titleOBS = 'Ref', checkstats = 'on')
	plt.savefig('taylor_BG1_BNU_LAI3G.png',dpi=300)


if __name__ == '__main__':
  main()
