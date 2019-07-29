import sys
import numpy as np
from numpy import meshgrid
import netCDF4
import matplotlib.pyplot as plt
from matplotlib import colors as c
from netCDF4 import num2date, date2num, date2index
from mpl_toolkits.basemap import Basemap, shiftgrid
import datetime

from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
from matplotlib import cm

from os import listdir
from os.path import isfile, join

from mkt import test #calculate trend statistics: Man-Kendall test
from PyEMD import EEMD #EEMD decomposition
from pyunicorn import timeseries #surrogate data testing

import geopandas as gpd
import shapely
from shapely.geometry import Polygon, mapping

#helper functions to determine trend type: monotonically increasing, maximum (concave), minimum (convex), monotonically decreasing
def check_increasing(seq):
    # This will check if it is monotonically increasing:
    for i in range(len(seq)-1):
        if seq[i] <= seq[i+1]:
            return True
        else:
            return False

def check_decreasing(seq):
    # This will check if it is monotonically decreasing:
    for i in range(len(seq)-1):
        if seq[i] >= seq[i+1]:
            return True
        else:
            return False

def green_to_brown(seq):
    dx = np.diff(seq)
    idx = np.where(dx[1:] * dx[:-1] < 0)[0][0] + 1 #extrema index
    #check if it's green to brown or brown to green:
    if (seq[idx-1] <= seq[idx]) & (seq[idx+1] <= seq[idx]):
        return True, idx
    else:
        return False, idx

def timeAsymmetry(ts):
	#calculate time-asymmetry statistic: https://www.rdocumentation.org/packages/nonlinearTseries/versions/0.2.1/topics/timeAsymmetry
    ta = np.mean(ts[0:(len(ts)-1)] * ts[1:(len(ts))]**2 - ts[0:(len(ts)-1)]**2 * ts[1:(len(ts))])
    return ta


def nonlinear_trend_test(trend):
	#Use surrogate method to test for EEMD trend significance
    t = len(trend)
    ts = trend.reshape(1,-1) #reshape into 2d array for surrogate methods
    ts_sur = timeseries.surrogates.Surrogates(ts) #generate an instance of surrogate class
    sur_list = [ts] #original time series is the first item in the surrogate list
    # Assign EEMD to `eemd` variable
    eemd = EEMD()
    eemd.noise_seed(12345)
    #detect extrema using parabolic method
    emd = eemd.EMD
    emd.extrema_detection="parabol"
    for i in range(19): #0.05 significance level for one-sided test
        sur = ts_sur.refined_AAFT_surrogates(ts, 100) #Return surrogates using the iteratively refined amplitude adjusted Fourier transform method.
        #detrend surrogate
        eIMFs_n = eemd.eemd(sur.flatten(), np.arange(t))
        sur_trend = eIMFs_n[-1]	    		
        sur_list.append(sur_trend)
    ta_list = [timeAsymmetry(ts.flatten()) for ts in sur_list] #test statistic: time asymmetry    
    return np.array(ta_list)

def create_grid(xmin,ymin,xmax,ymax):
	#create polygon grid geo-pandas dataframe
	width = 0.5
	height = 0.5
	rows = int(np.ceil((ymax-ymin) /  height))
	cols = int(np.ceil((xmax-xmin) / width))
	XleftOrigin = xmin
	XrightOrigin = xmin + width
	YtopOrigin = ymax
	YbottomOrigin = ymax- height
	polygons = []
	for i in range(cols):
	    Ytop = YtopOrigin
	    Ybottom =YbottomOrigin
	    for j in range(rows):
	        polygons.append(Polygon([(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)])) 
	        Ytop = Ytop - height
	        Ybottom = Ybottom - height
	    XleftOrigin = XleftOrigin + width
	    XrightOrigin = XrightOrigin + width
	grid = gpd.GeoDataFrame({'geometry':polygons})
	return grid, rows, cols



def protected_mask(path_nc4, path_protected_shp, lats, lons, mean_file):
	#read data and variables
	f = netCDF4.Dataset(path_nc4) 
	f.set_auto_mask(False) #missing value = -9999.0
	LAI = f.variables['LAI']
	time = f.variables['time']
	dates = num2date(time[:], time.units)
	latt,lonn = f.variables['lat'][:], f.variables['lon'][:]
	#get actual lon/lat range
	latbounds = list(latt[(latt <= lats[1]) & (latt >= lats[0])])
	lonbounds = list(lonn[(lonn <= lons[1]) & (lonn >= lons[0])])
	
	#create map polygon
	grid, rows, cols = create_grid(lonbounds[0],latbounds[0],lonbounds[-1],latbounds[-1])
	
	#read proctected shape file
	points_shape_map = gpd.read_file(path_protected_shp)
	points_shape_map.to_crs(epsg=4326, inplace=True)

	#identify grid cells in the shape file intersect with polygons 
	data = []
	for index, protected in points_shape_map.iterrows():
	    for index2, cell in grid.iterrows():
	        if protected['geometry'].intersects(cell['geometry']):
	            data.append({'geometry': cell['geometry']})
	df = gpd.GeoDataFrame(data,columns=['geometry'])

	#and drop duplicates by convert to wkb
	df["geometry"] = df["geometry"].apply(lambda geom: geom.wkb)
	df = df.drop_duplicates(["geometry"])
	# convert back to shapely geometry
	df["geometry"] = df["geometry"].apply(lambda geom: shapely.wkb.loads(geom))

	#map back to lat,lon list
	#extract all coordinates from geometry column
	g = [i for i in df.geometry]
	#map all polygons to coordinates
	all_coords = [mapping(item)["coordinates"] for item in g]
	#loop through all coordinates to find corresponding lat, lon tuple
	list_protected = []
	for coords in all_coords:
	    idx_tup = coords[0][0]
	    idx_lat = np.where(latt==idx_tup[1])[0][0]
	    idx_lon = np.where(lonn==idx_tup[0])[0][0]
	    list_protected.append([idx_lat,idx_lon])

	#get all monthly data, perform detrending using EEMD, and perform surrogate test for significance
	# Assign EEMD to `eemd` variable 
	eemd = EEMD()
	eemd.noise_seed(12345)
	#detect extrema using parabolic method
	emd = eemd.EMD
	emd.extrema_detection="parabol"
	#initial result dataframe
	lai_monthly_mkt, lai_monthly_cat = [], []
	for i in range(12): 
	    t = mean_file.shape[1] #get time-length, lat length, lon length
	    lai_cat = np.zeros((len(latt), len(lonn))) #initiate global mask for categorization
	    lai_mkt = np.zeros((len(latt), len(lonn))) #initiate global mask for stipling significance

	    for coords in list_protected:
	    	lat, lon = coords[0], coords[1]
	    	lai_monthly = mean_file[i,:,lat,lon]  #subset per month per lat lon, using the calculated mean_file
	    	if lai_monthly.all() < 0: #missing data present
	    		lai_cat[lat,lon] = 0
	    		lai_mkt[lat,lon] = 0
	    	else:
	    		eIMFs_n = eemd.eemd(lai_monthly, np.arange(t))
	    		trend = eIMFs_n[-1]
	    		if check_increasing(trend):
	    			lai_cat[lat,lon] = 1
	    			#test for significance: 
	    			ta_list = nonlinear_trend_test(trend)
	    			if np.argmax(ta_list) == 0:
	    				lai_mkt[lat,lon] = 1
	    		elif check_decreasing(trend):
	    			lai_cat[lat,lon] = -1
	    			#test for significance: 
	    			ta_list = nonlinear_trend_test(trend)
	    			if np.argmin(ta_list) == 0:
	    				lai_mkt[lat,lon] = -1
	    		else:
	    			lai_cat[lat,lon] =  5 #flag out inflection shape trend series if any

	    lai_monthly_mkt.append(lai_mkt)
	    lai_monthly_cat.append(lai_cat)
	#convert the result list back to arrays
	lai_monthly_mkt = np.array(lai_monthly_mkt)
	lai_monthly_cat = np.array(lai_monthly_cat)  

	# subset the region of interest by lon lat indexes
	# latitude lower and upper index
	latli = np.argmin( np.abs( latt - lats[0] ) )
	latui = np.argmin( np.abs( latt - lats[1] ) ) 

	# longitude lower and upper index
	lonli = np.argmin( np.abs( lonn - lons[0] ) )
	lonui = np.argmin( np.abs( lonn - lons[1] ) )  

	lai_cat = lai_monthly_cat[:,latli:latui,lonli:lonui]
	lai_mkt = lai_monthly_mkt[:,latli:latui,lonli:lonui]

	# Write the array to disk
	with open('cat_sg3_ken_sur.txt', 'w') as outfile:
		outfile.write('# Array shape: {0}\n'.format(lai_cat.shape))
		for data_slice in lai_cat:
			np.savetxt(outfile, data_slice, fmt='%-7.2f')
			outfile.write('# New slice\n')
	# Write the array to disk
	with open('mkt_sg3_ken_sur.txt', 'w') as outfile:
		outfile.write('# Array shape: {0}\n'.format(lai_mkt.shape))
		for data_slice in lai_mkt:
			np.savetxt(outfile, data_slice, fmt='%-7.2f')
			outfile.write('# New slice\n')
	#print out any infletion point index		
	print(np.where(lai_cat==5))
	return lai_cat, lai_mkt, latbounds, lonbounds, '20 SG3 models mean'


def plot(category_mask, stiple_mask, title, lats, lons, latbounds, lonbounds):
	#customize color map
	top = cm.get_cmap('Greens', 128)
	bottom = cm.get_cmap('copper', 128)
	newcolors = np.vstack((bottom(np.linspace(0, 1, 128)),top(np.linspace(0, 1, 128))))
	newcmp = ListedColormap(newcolors, name='GrBr')

	#create plots
	fig,axs = plt.subplots(ncols=6, nrows=2, sharex=True, sharey=True, figsize=(20,5))
	axs = axs.ravel()
	fig.suptitle(title + str(' trend_categorization'), fontsize=20)

	for i in range(12):
	    map = Basemap(projection='merc',llcrnrlat=lats[0],urcrnrlat=lats[1],\
	                llcrnrlon=lons[0],urcrnrlon=lons[1], resolution='h', ax=axs[i])
	    map.drawcountries()
	    map.drawcoastlines(color='#0000ff')
	    llons, llats = np.meshgrid(lonbounds, latbounds)
	    x,y = map(llons,llats)
	    cs = map.pcolormesh(x,y,category_mask[i], cmap=newcmp,vmin=-5, vmax=5) 
	    mask_pos = np.ma.masked_less_equal(stiple_mask[i], 0) #show significance when stiple mask = -1
	    mask_neg = np.ma.masked_greater_equal(stiple_mask[i], 0) #show significance when stiple mask = 1
	    st = map.pcolor(x,y,mask_pos, hatch='...', alpha=0.)
	    st = map.pcolor(x,y,mask_neg, hatch='...', alpha=0.)
	    fig.colorbar(cs,ax=axs[i], fraction=0.046, pad=0.04)
	fig.savefig(str('trend test stippling plot_bg1_tan_sur.png'),bbox_inches='tight')

def main():
	#get nc4 files
	mypath = '/Users/nhuang37/Desktop/NYU DS/Yr1 Summer/Data/LAI/BG1'
	#get kenya and tanzania shape files to identify protected areas
	kenya_shp = '/Users/nhuang37/Desktop/NYU DS/Yr1 Summer/Data/shape/WDPA_Jul2019_KEN-shapefile/WDPA_Jul2019_KEN-shapefile-polygons.shp'
	tanzania_shp = '/Users/nhuang37/Desktop/NYU DS/Yr1 Summer/Data/shape/WDPA_Jul2019_TZA-shapefile/WDPA_Jul2019_TZA-shapefile-polygons.shp'
	#import the LAI mean of SG3 models
	LAI_mean = np.loadtxt('/Users/nhuang37/Desktop/NYU DS/Yr1 Summer/Data/LAI/BG1/SG3_mean.txt')
	LAI_mean = LAI_mean.reshape((12,31,360,720))
	#specify the plotting area lat/lon
	lats = [-4.75,4.75]  # [ -11.75 , -1.25 ] #
	lons = [33.25, 41.75]  #[ 28.25 , 40.75 ] #
	category_mask, stiple_mask, latbounds, lonbounds, title  = protected_mask(mypath, tanzania_shp, lats, lons, LAI_mean)
	plot(category_mask, stiple_mask, title, lats, lons, latbounds, lonbounds)


if __name__ == '__main__':
  main()
