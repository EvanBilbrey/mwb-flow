# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:09:21 2020

DNRC Monthly Runoff Model
@author: Todd Blythe - Hydrologist
"""

######## Gray and McCabe WaterBalance Model ########
### Version 1.0 ###
### Lolo Creek Specific ###

### Packages ###
import pandas as pd
import numpy as np
import xarray as xr
import salem
import glob
from rasterio import features
from affine import Affine
import geopandas as gpd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider
from datetime import datetime

Wdir = r'F:\MT_DNRC_Working\WB_Model'
os.chdir(Wdir)

### NOTE: shapefiles and GridMET netcdf file directories are hardcoded in
### CatchmetGridmet script - !change in future version!
from datautils.process_gridmet import CatchmetGridmet

shp_ids = {'Sleeman and Mormon Creeks': 5,
           'Lower Lolo Creek': 6,
           'Middle Lolo Creek': 3,
           'South Fork Lolo Creek': 8,
           'Graves Creek': 2,
           'Lolo Creek Headwaters': 7}


def get_meteorData(shp_dir, shp_info, start_dt, end_dt):
    shp_direct = shp_dir

    ## Basin Name, ID
    NAME, iD = shp_info

    ## need to include date formatting/parser

    ## Create Date Range
    Dts = pd.date_range(start_dt, end_dt, freq='M')

    ## append data arrays to lists
    prc = []
    temp = []
    ## collect spatial reference info
    SR = []

    print("Aquiring, clipping, and resampling input meteorological data")
    for date in tqdm(Dts):
        gridpr = CatchmetGridmet(date.year, date.month, shp_dirct, _id=iD, variable='pr')
        pr_array = gridpr.get_monthly_gridmet()
        prc.append(pr_array[0])
        gridtemp = CatchmetGridmet(date.year, date.month, shp_dirct, _id=iD, variable='temp')
        temp_array = gridtemp.get_monthly_gridmet()
        temp.append(temp_array[0])
        if date == Dts[-1]:
            SRS = gridtemp.get_spatial_ref()
            SR.append(SRS)

    ## create dictionary of all multidimensional data with date range
    mdat = {'temp': temp, 'precip': prc, 'dates': Dts}  # define as self.IN_Dict
    ## dictionary of spatial reference information
    CSR = SR[0]

    class _mdat:

        def __init__(self, data_dic, sr_dic):
            self.IN_Dict = data_dic
            self.basin_name = NAME
            self.shape_ID = iD
            self.DF = pd.DataFrame({k: data_dic[k] for k in ('temp', 'precip')},
                                   index=data_dic['dates'])
            self.dts = data_dic['dates']
            self.DATES = data_dic['dates'].strftime('%b %Y')
            self.temp_arr = np.ma.masked_values(data_dic['temp'], -987654321)
            self.prec_arr = np.ma.masked_values(data_dic['precip'], -987654321)
            self.xOrigin, self.yOrigin = sr_dic['origin']
            self.nrows = self.temp_arr.shape[1]
            self.ncols = self.temp_arr.shape[2]
            self.pixelWidth = sr_dic['cell_width']
            self.pixelHeight = sr_dic['cell_height']
            self.col_coords = np.linspace(0, self.ncols, num=self.ncols, endpoint=False)
            self.xcoords = (self.col_coords * self.pixelWidth) + self.xOrigin
            self.row_coords = np.linspace(0, self.nrows, num=self.nrows, endpoint=False)
            self.ycoords = (self.row_coords * self.pixelHeight) + self.yOrigin
            self.Xgrid, self.Ygrid = np.meshgrid(self.xcoords, self.ycoords)

        def plot_date(self, variable, month, year, plot_type, **kwargs):

            if variable not in list(self.IN_Dict.keys()):
                raise ValueError(
                    f'{variable} not included in available variables\nplease choose from {list(self.IN_Dict.keys())[0:-1]}')

            if (year, month) not in zip(self.DF.index.year, self.DF.index.month):
                raise ValueError(
                    f'{year}-{month} not in date range, choose date between {self.DATES[0]} and {self.DATES[-1]}')

            if plot_type not in ['image', 'contour']:
                raise ValueError('Invalid Plot Type: please choose "image" or "contour"')

            fig = plt.figure(figsize=(11, 8), dpi=150)
            ax = plt.subplot(111)

            if plot_type == 'contour':
                # cmap = plt.cm.get_cmap("summer")
                cp = ax.contourf(self.Xgrid, self.Ygrid, self.DF[f'{variable}'][f'{year}-{month}'].iloc[0], **kwargs)
            if plot_type == 'image':
                cp = ax.imshow(self.DF[f'{variable}'][f'{year}-{month}'].iloc[0],
                               origin='upper',
                               extent=tuple(np.array([self.xOrigin + (self.pixelWidth * 0.5),
                                                      self.xcoords[-1] + (self.pixelWidth * 0.5),
                                                      self.ycoords[-1] + (self.pixelHeight * 0.5),
                                                      self.yOrigin + (self.pixelHeight * 0.5)])),
                               **kwargs)
            ax.set_aspect(aspect='equal')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.08)
            cb = fig.colorbar(cp, cax=cax)  # Add a colorbar to a plot
            cb.ax.tick_params(labelsize=14)
            ax.set_xlabel('EASTING (ft)', fontsize=18)
            ax.set_ylabel('NORTHING (ft)', fontsize=18)
            ax.set_ylim(self.yOrigin - (self.pixelWidth * self.nrows), self.yOrigin)
            ax.set_xlim(self.xOrigin, self.xOrigin + (self.pixelWidth * self.ncols))
            ax.tick_params(axis='both', which='major', labelsize=14)

        ## need to make update so that colorbar rescales and such need to remove old one and
        ## replot using cb.remove()
        def multi_dim_plots(self, variable, plot_type, **kwargs):
            fig = plt.figure(figsize=(11, 8), dpi=150)
            ax = plt.subplot(111)
            fig.subplots_adjust(left=0.25, bottom=0.25)

            if variable == 'precip':
                cube = self.prec_arr
            if variable == 'temp':
                cube = self.temp_arr

            # select first image
            axis = 0
            s = [slice(0, 1) if i == axis else slice(None) for i in range(3)]
            im = cube[s].squeeze()

            # custom index-date formatter
            # N = len(dts)
            # def format_date(x, pos=None):
            #     thisind = np.clip(int(x + 0.5), 0, N - 1)
            #     return dts[thisind].strftime('%b %Y')

            if plot_type == 'contour':
                cp = ax.contourf(self.Xgrid, self.Ygrid, im, **kwargs)
                ax.set_aspect(aspect='equal')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.08)
                cb = fig.colorbar(cp, cax=cax)  # Add a colorbar to a plot
                cb.ax.tick_params(labelsize=14)
                ax.set_xlabel('EASTING (ft)', fontsize=16)
                ax.set_ylabel('NORTHING (ft)', fontsize=16)
                ax.set_ylim(self.yOrigin - (self.pixelWidth * self.nrows), self.yOrigin)
                ax.set_xlim(self.xOrigin, self.xOrigin + (self.pixelWidth * self.ncols))
                ax.tick_params(axis='both', which='major', labelsize=14)

                # define slider
                axcolor = 'gray'
                ax2 = fig.add_axes([0.25, 0.1, 0.65, 0.01], facecolor=axcolor)

                slider = Slider(ax2, 'Date', 0, cube.shape[axis] - 1,
                                valinit=0, valfmt='%i', valstep=1, facecolor='white',
                                edgecolor='black')
                t1 = ax2.text(-0.065, -1.8, self.DATES[slider.val], size=11, transform=ax2.transAxes)

                def update(val):
                    ind = int(slider.val)
                    s = [slice(ind, ind + 1) if i == axis else slice(None)
                         for i in range(3)]
                    im = cube[s].squeeze()
                    ax.clear()
                    ax.contourf(self.Xgrid, self.Ygrid, im, **kwargs)
                    ax.set_xlabel('EASTING (ft)', fontsize=16)
                    ax.set_ylabel('NORTHING (ft)', fontsize=16)
                    cb.set_clim(im.min(), im.max())
                    t1.set_text(self.DATES[ind])
                    fig.canvas.draw()
                    cb.draw_all()

                slider.on_changed(update)

                plt.show()

            if plot_type == 'image':
                cp = ax.imshow(im,
                               origin='upper',
                               extent=tuple(np.array([self.xOrigin + (self.pixelWidth * 0.5),
                                                      self.xcoords[-1] + (self.pixelWidth * 0.5),
                                                      self.ycoords[-1] + (self.pixelHeight * 0.5),
                                                      self.yOrigin + (self.pixelHeight * 0.5)])),
                               **kwargs)
                ax.set_aspect(aspect='equal')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.08)
                cb = fig.colorbar(cp, cax=cax)  # Add a colorbar to a plot
                cb.ax.tick_params(labelsize=14)
                ax.set_xlabel('EASTING (ft)', fontsize=16)
                ax.set_ylabel('NORTHING (ft)', fontsize=16)
                ax.set_ylim(self.yOrigin - (self.pixelWidth * self.nrows), self.yOrigin)
                ax.set_xlim(self.xOrigin, self.xOrigin + (self.pixelWidth * self.ncols))
                ax.tick_params(axis='both', which='major', labelsize=14)

                # define slider
                axcolor = 'gray'
                ax2 = fig.add_axes([0.25, 0.1, 0.65, 0.01], facecolor=axcolor)

                slider = Slider(ax2, 'Date', 0, cube.shape[axis] - 1,
                                valinit=0, valfmt='%i', valstep=1, facecolor='white',
                                edgecolor='black')
                t1 = ax2.text(-0.065, -1.8, self.DATES[slider.val], size=11, transform=ax2.transAxes)

                def update(val):
                    ind = int(slider.val)
                    s = [slice(ind, ind + 1) if i == axis else slice(None)
                         for i in range(3)]
                    im = cube[s].squeeze()
                    cp.set_data(im)
                    cb.set_clim(im.min(), im.max())
                    t1.set_text(self.DATES[ind])
                    fig.canvas.draw()
                    cb.draw_all()

                slider.on_changed(update)

                plt.show()

    C = _mdat(mdat, CSR)
    return C


## gather meterological data
st_dt = '2016-01-01'
end_dt = '2019-12-31'
shape_dir = r'F:\MT_DNRC_Working\WB_Model\lolo_data\shp\Lolo_WB_Model_Calibration_Catchments_32611.shp'
shape = list(shp_ids.items())[-1]
MData = get_meteorData(shape_dir, shape, st_dt, end_dt)
### Calibration Parameters ###
Tsnow = -5
Train = 5
drf = 0.06
alpha = 0.6
rfactor = 0.5
STC = 76  # Soil Storage Capacity (mm)

## Streamflow Time Series
Qflw_all = pd.read_csv(r'streamflow\Model_Subcatchment_Runoff_cms.csv',
                       index_col='Date',
                       parse_dates=True)
qflw = Qflw_all[MData.basin_name]
# test
qlfw = qflw.loc[MData.dts]

# Headwaters Latitude = 46.739440

### Model Computations ###
## Model, input is dictionary similar to previous mdat dictionary
## = {temp, precip, mon_streamflow, dates (months)} where temp and precip
##    are ndarrays, dates is a vectors/1d array, and mon_Q is
##    a pandas series with dates as index and flow in m^3/s
moddict = {'temp': MData.temp_arr,
           'precip': MData.prec_arr,
           'dates': MData.dts,
           'mon_Q': qflw}
