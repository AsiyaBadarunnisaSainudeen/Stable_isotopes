#!/usr/bin/env python


import pydap.client
import numpy as np
import matplotlib.pyplot as plt
import pprint
import xarray as xr
import cftime
from glob import glob

#########################################################
########################################################


df_sst = xr.open_dataset('ERA5.skt.2018_2023.nc')
df_sss = xr.open_dataset('salinity.nc')

#print(df_sss)
#latitude = df_sst.latitude
#longitude = df_sst.longitude
#time = df_sst.time

#sss = xr.DataArray(dsss, dims = ['time', 'latitude', 'longitude'])

region = 'NA'

if region == 'GOM': #GulfofMexico

    dsst = df_sst.sel(latitude = slice(0, -10), longitude = slice(270, 280))

if region == 'NA': #North America

    dsss = df_sss.sel(LAT94_159 = slice(15, 45), LONN170_N50 = slice(-130, -100))
    dsst = df_sst.sel(latitude = slice(45, 15), longitude = slice(-130, -100))



sst = dsst.sst[0:80, 0,:,:]
sss = dsss.SALT[0:80, 0,:,:]


#print (sss.shape, sst.shape)


dsss_seas = sss.groupby("TIME.season").mean(dim="TIME")
dsst_seas = sst.groupby("time.season").mean(dim="time")

#print(dsss_seas, dsst_seas)

clim_sst = sst.groupby('time.month').mean('time')
clim_sss = sss.groupby('TIME.month').mean('TIME')

anom_sst = sst.groupby('time.month') - clim_sst
anom_sss = sss.groupby('TIME.month') - clim_sss

anom_sst = sst
anom_sss = sss

#dsss_seas_anom = anom_sss.groupby("TIME.season").mean(dim="TIME")
#dsst_seas_anom = anom_sst.groupby("time.season").mean(dim="time")

############################################
#PLOTTING FUNCTION (ABS)#
############################################
############################################
def plot_spatial (var, lat, lon, CMAP, levels, title, label, ofile):

    ### Importing all the essential packages ###
    ############################################

    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import cartopy.feature as cfeature

    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib import cm
    from matplotlib.patches import Polygon
    import matplotlib.patches as mpatches

    from netCDF4 import Dataset, MFDataset
    import cartopy as cart
    import cartopy.feature as cf

    ############################################


    var = var
    trans = ccrs.PlateCarree()

    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(9, 6))
    ax = plt.subplot(111, projection=proj)
  
    cmap = plt.cm.get_cmap(f'{CMAP}')

    if len(levels) == 1:
        cf = ax.contourf(lon, lat, var[:,:], cmap=cmap,
                         extend='both',transform=trans)
        cbar = fig.colorbar(cf,shrink= 1,orientation='horizontal',
                        fraction=0.07, anchor=(0.0,0.0),pad=0.07)


    else:
        ticks = []
        for i in range(0,len(levels), 1):
            ticks.append(levels[i])
            print (ticks)
        #ticks = levels
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        print (f'{levels}')
        cf = ax.contourf(lon, lat, var[:,:], cmap=cmap, extend='both',
                         transform=trans,levels=levels,norm=norm)
        cbar = fig.colorbar(cf,shrink= 0.95,orientation='horizontal',
                fraction=0.07,anchor=(0.5,0.0),pad=0.07,ticks=ticks)

    cbar.ax.tick_params(direction='inout',
                    labelsize=11,
                    length=12,
                    width=2,
                    colors='k',
                    grid_color='k',
                    grid_alpha=0.7)


    ax.set_title(f' {title}', fontsize = 15, weight='bold')
        
    clabel = f'{label}'
    cbar.set_label(clabel, size=10,  weight='bold')


    ax.coastlines()


    gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels=True,
                  linewidth = 2,
                  color='xkcd:black', alpha=0.4, linestyle=':')
    #gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    #                  linewidth=1.5,
    #                  color='k', alpha=0.9, linestyle=':')

    gl.top_labels = False
    gl.left_labels = True
    gl.right_labels=False
    gl.xformatter = LONGITUDE_FORMATTER    # Gives the degree sign
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 14,  'weight': 'bold'}
    gl.ylabel_style = {'size': 14, 'weight': 'bold'}
    
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES, linestyle=':', edgecolor='black', linewidth=0.5)
    ax.coastlines()
    plt.savefig(ofile)
    print(f'Output file: {ofile}')
    plt.show()

############################################
#PLOTTINF FUNCTION (ABS)#
############################################
############################################

""" 
seasons = ['DJF', 'JJA', 'MAM', 'SON']
fields = ['sst', 'sss']

#for plotting seasonal means
for seas in seasons:
    for field in fields:
        if field == 'sst':
            dfd = dsst_seas
            ##dfd = dsst_seas_anom
            lat = dfd['latitude']
            lon = dfd['longitude']
            CMAP = 'RdBu_r'
            levels = np.arange(284, 304, 2)
            unit = 'degree K'
            title = f'{seas} {field} mean'
            label = f'{field} in {unit}'
            
        if field == 'sss':
            dfd = dsss_seas
            ##dfd = dsss_seas_anom
            lat = dfd['LAT94_159']
            lon = dfd['LONN170_N50']
            CMAP = 'Oranges'
            #levels = [0]
            levels = np.arange(32, 36, 0.4)
            unit = 'psu'
            title = f'{seas} {field} mean'
            label = f'{field} in {unit}'

        ofile = f'{seas}.{field}.mean.pdf'
        var = dfd.sel(season = f'{seas}')
        
        plot_spatial (var, lat, lon, CMAP, levels, title, label, ofile)
"""     

locations = ['C1', 'C2', 'C3', 'B1', 'B2']
#locations = ['C1']
print(clim_sst)
#clim_sss)
#print(anom_sst, anom_sst)


def getIdx(var, point):
    return int(np.argmin(np.abs(var-point)))

LAT94_159 = clim_sss.LAT94_159.values
LONN170_N50 = clim_sss.LONN170_N50.values

latitude = clim_sst.latitude.values
longitude = clim_sst.longitude.values

for loc in locations:
     
    if loc == 'C1': 

        lat_index = getIdx(LAT94_159, 41)
        lon_index = getIdx(LONN170_N50, -126)

        lat_index1 = getIdx(latitude, 41)
        lon_index1 = getIdx(longitude, -126)
        
    if loc == 'C2':
        
        lat_index = getIdx(LAT94_159, 35)
        lon_index = getIdx(LONN170_N50, -121)

        lat_index1 = getIdx(latitude, 35)
        lon_index1 = getIdx(longitude, -121)

    if loc == 'C3':
        
        lat_index = getIdx(LAT94_159, 27)
        lon_index = getIdx(LONN170_N50, -115)

        lat_index1 = getIdx(latitude, 27)
        lon_index1 = getIdx(longitude, -115)
        
    if loc == 'B1':
        
        lat_index = getIdx(LAT94_159, 31)
        lon_index = getIdx(LONN170_N50, -114)

        lat_index1 = getIdx(latitude, 31)
        lon_index1 = getIdx(longitude, -114)

    if loc == 'B2':
        
        lat_index = getIdx(LAT94_159, 25)
        lon_index = getIdx(LONN170_N50, -110)

        lat_index1 = getIdx(latitude, 25)
        lon_index1 = getIdx(longitude, -110)


    clim_sss1 = clim_sss[:, lat_index, lon_index]
    clim_sst1 = clim_sst[:, lat_index1, lon_index1]

    anom_sss1 = anom_sss[:, lat_index, lon_index]
    anom_sst1 = anom_sst[:, lat_index1, lon_index1]
    
    print(loc)
    #print(clim_sss1.shape)
    #print(clim_sst1.shape)

    #print(anom_sss1.values)
    #print(anom_sst1.values)
    
    for i in range(len(anom_sss1.values)):
        #print (loc)
        print(anom_sst1[i].time),
    
    
