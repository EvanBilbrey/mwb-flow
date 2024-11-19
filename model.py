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


shp_ids = {'Sleeman and Mormon Creeks':5, 
           'Lower Lolo Creek':6, 
           'Middle Lolo Creek':3,
           'South Fork Lolo Creek':8,
           'Graves Creek':2,
           'Lolo Creek Headwaters':7}

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
    mdat = {'temp':temp, 'precip':prc, 'dates':Dts} # define as self.IN_Dict
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
                raise ValueError(f'{variable} not included in available variables\nplease choose from {list(self.IN_Dict.keys())[0:-1]}')
            
            if (year, month) not in zip(self.DF.index.year, self.DF.index.month):
                raise ValueError(f'{year}-{month} not in date range, choose date between {self.DATES[0]} and {self.DATES[-1]}')
            
            if plot_type not in ['image', 'contour']:
                raise ValueError('Invalid Plot Type: please choose "image" or "contour"')
            
            fig = plt.figure(figsize=(11,8), dpi=150)
            ax = plt.subplot(111)
            
            if plot_type == 'contour':
            # cmap = plt.cm.get_cmap("summer")
                cp = ax.contourf(self.Xgrid, self.Ygrid, self.DF[f'{variable}'][f'{year}-{month}'].iloc[0], **kwargs)
            if plot_type == 'image':
                cp = ax.imshow(self.DF[f'{variable}'][f'{year}-{month}'].iloc[0],
                               origin='upper',
                               extent=tuple(np.array([self.xOrigin+(self.pixelWidth*0.5),
                                            self.xcoords[-1]+(self.pixelWidth*0.5),
                                            self.ycoords[-1]+(self.pixelHeight*0.5),
                                            self.yOrigin+(self.pixelHeight*0.5)])),
                               **kwargs)
            ax.set_aspect(aspect='equal')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.08)
            cb = fig.colorbar(cp, cax=cax) # Add a colorbar to a plot
            cb.ax.tick_params(labelsize=14)
            ax.set_xlabel('EASTING (ft)', fontsize=18)
            ax.set_ylabel('NORTHING (ft)', fontsize=18)
            ax.set_ylim(self.yOrigin-(self.pixelWidth*self.nrows), self.yOrigin)
            ax.set_xlim(self.xOrigin, self.xOrigin+(self.pixelWidth*self.ncols))
            ax.tick_params(axis='both', which='major', labelsize=14)

        ## need to make update so that colorbar rescales and such need to remove old one and 
        ## replot using cb.remove()
        def multi_dim_plots(self, variable, plot_type, **kwargs):
            fig = plt.figure(figsize=(11,8), dpi=150)
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
            
            #custom index-date formatter
            # N = len(dts)
            # def format_date(x, pos=None):
            #     thisind = np.clip(int(x + 0.5), 0, N - 1)
            #     return dts[thisind].strftime('%b %Y')
            
            if plot_type == 'contour':
                cp = ax.contourf(self.Xgrid, self.Ygrid, im, **kwargs)
                ax.set_aspect(aspect='equal')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.08)
                cb = fig.colorbar(cp, cax=cax) # Add a colorbar to a plot
                cb.ax.tick_params(labelsize=14)
                ax.set_xlabel('EASTING (ft)', fontsize=16)
                ax.set_ylabel('NORTHING (ft)', fontsize=16)
                ax.set_ylim(self.yOrigin-(self.pixelWidth*self.nrows), self.yOrigin)
                ax.set_xlim(self.xOrigin, self.xOrigin+(self.pixelWidth*self.ncols))
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
                               extent=tuple(np.array([self.xOrigin+(self.pixelWidth*0.5),
                                                    self.xcoords[-1]+(self.pixelWidth*0.5),
                                                    self.ycoords[-1]+(self.pixelHeight*0.5),
                                                    self.yOrigin+(self.pixelHeight*0.5)])),
                               **kwargs)
                ax.set_aspect(aspect='equal')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.08)
                cb = fig.colorbar(cp, cax=cax) # Add a colorbar to a plot
                cb.ax.tick_params(labelsize=14)
                ax.set_xlabel('EASTING (ft)', fontsize=16)
                ax.set_ylabel('NORTHING (ft)', fontsize=16)
                ax.set_ylim(self.yOrigin-(self.pixelWidth*self.nrows), self.yOrigin)
                ax.set_xlim(self.xOrigin, self.xOrigin+(self.pixelWidth*self.ncols))
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
STC = 76 # Soil Storage Capacity (mm)

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
moddict = {'temp':MData.temp_arr, 
           'precip':MData.prec_arr, 
           'dates':MData.dts,  
           'mon_Q':qflw}

class WB_Model():
    
    def __init__(self, data_dict, soil_stor_cap, latitude, init_per, Temp_sno=0,  
                 Temp_rain=5, dr_frac=0.06, mlt_rate=0.6, rfactor=0.5):
        self.STC = soil_stor_cap
        self.snotemp = Temp_sno
        self.raintemp = Temp_rain
        self.DRF = dr_frac
        self.alpha = mlt_rate
        self.rfactor = rfactor
        self.latitude = latitude
        self.init_per = init_per
        self.STo = 0.0
        self.snw_st_init = 0.0
        self.ST_inter = 0.0
        
        # Calibration Parameter Ranges
        self.snotemp_rng = [-10, 0]
        self.raintemp_rng = [0, 10]
        self.DRF_rng = [0.01, 0.99]
        self.alpha_rng = [0.01, 0.99]
        self.rfactor_rng = [0.01, 0.99]
        
        self.T = data_dict['temp'] - 273.15  ## Convert to Celcius
        self.P = data_dict['precip']
        self.dates = data_dict['dates']
        self.monthly_Q = data_dict['mon_Q']
        self.days_in_mnth = self.dates.days_in_month
       
        # gridded data for WB variables
        self.snowfall = None
        self.snow_stor = None
        self.P_snow_melt = None
        self.snowmelt = None
        self.soil_m = None
        self.PET = None
        self.AET = None
        self.rain = None
        self.direct_runoff = None
        self.surplus_runoff = None
        self.runoff = None
        
        
    def _day_length(self):
        rad_lat = np.radians(self.latitude)
        days = self.dates - pd.Timedelta(15, unit='D')
        J = days.dayofyear.values
        decl = 0.409*np.sin((2*np.pi/365)*J - 1.39)
        sunset_ang = np.arccos(-np.tan(rad_lat)*np.tan(decl))
        Dl = (24/np.pi)*sunset_ang
        return Dl    

    def _snow_input(self, Tsnow, Train):
        ## Snow Storage
        ## multidimensional array indices
        sno_ind = np.where(self.T <= Tsnow)
        mix_ind = np.where((self.T > Tsnow) & (self.T < Train))
        rain_ind = np.where(self.T >= Train)
        Psnow = self.P * 1.0
        Psnow[sno_ind] = Psnow[sno_ind]
        Psnow[mix_ind] = Psnow[mix_ind] * ((Train - self.T[mix_ind])/(Train - Tsnow))
        Psnow[rain_ind] = 0.0
        return Psnow
        
    def _rain(self):
        ## Rain Fraction
        Prain = self.P - self.snowfall
        return Prain
        
    def _direct_runoff(self, drf):
        ## Direct Runoff
        Dir_RO = drf * self.rain    # Water Balance Runoff Variable
        return Dir_RO
        
    def _snowmelt(self, alpha, Tsnow):
        ## Snowmelt
        Sno_M = alpha * (self.T - Tsnow) * self.days_in_mnth.values[:,None,None]      # Water Balance Runoff Variable
        Sno_M[Sno_M < 0.0] = 0.0
        return Sno_M

    def _PET(self):
        ## Potential Evapotranspiration
        ## did not use Gray and McGabe formula because what they present for
        ##  water vapor density produces incorrect values (see Lu et al. 2005)
        e_sat = 6.108 * np.exp((17.27 * self.T) / (self.T + 237.3))
        rho_sat = 216.7 * (e_sat / (self.T + 273.3))
        D = self._day_length()
        Kpec = 1.0
        PET = 0.1651 * self.days_in_mnth.values[:,None, None] * (D[:,None,None]/12) * rho_sat * Kpec
        return PET

    def _soilmoisture_storage(self, dates, time_offset, Tsnow, Train, drf, alpha, rfactor):
        self.snowfall = self._snow_input(Tsnow, Train)
        self.rain = self._rain()
        self.direct_runoff = self._direct_runoff(drf)
        self.P_snow_melt = self._snowmelt(alpha, Tsnow)
        self.PET = self._PET()
        
        smstor = []
        sm_in = []
        AET_l = []
        surp_r = []
        snowstor = []
        snwmlt = []
        for i in tqdm(np.arange(time_offset, len(dates))):
            
            if i == 0:
                STi_1 = self.STo
                sno_stori_1 = self.snw_st_init
                STmi_1 = self.ST_inter
            else:
                STi_1 = smstor[i-1]
                sno_stori_1 = snowstor[i-1]
                STmi_1 = sm_in[i-1]
            
            ## Liquid water input to Soil Moisture Storage
            snow_storage = sno_stori_1 + self.snowfall[i,:,:] - self.P_snow_melt[i,:,:]
            
            hi_snwm = np.where(snow_storage > 0.0)
            lo_snwm = np.where(snow_storage <= 0.0)
            
            snwm = self.P_snow_melt[i,:,:] * 1.0
            snwm[hi_snwm] = self.P_snow_melt[i,:,:][hi_snwm]
            if type(sno_stori_1) == float:
                snwm[lo_snwm] = sno_stori_1 + self.snowfall[i,:,:][lo_snwm]
            else:
                snwm[lo_snwm] = sno_stori_1[lo_snwm] + self.snowfall[i,:,:][lo_snwm]

            snow_storage[snow_storage < 0.0] = 0.0
            
            P_tot = (self.rain[i,:,:] - self.direct_runoff[i,:,:]) + snwm

            ## multidimensional array indices
            hi_pet = np.where(P_tot >= self.PET[i,:,:])
            lo_pet = np.where(P_tot < self.PET[i,:,:])
            
            AET = self.PET[i,:,:] * 1.0
            AET[hi_pet] = AET[hi_pet]
            if type(STi_1) == float:
                AET[lo_pet] = P_tot[lo_pet] + (abs(P_tot[lo_pet] - self.PET[i,:,:][lo_pet]) * (STi_1/self.STC))
            else:
                AET[lo_pet] = P_tot[lo_pet] + (abs(P_tot[lo_pet] - self.PET[i,:,:][lo_pet]) * (STi_1[lo_pet]/self.STC))
            STi = STi_1 + P_tot - AET
            STmi = STi * 1.0
            sm_in.append(STmi)
            STi[STi < 0] = 0.0
            STi[STi > self.STC] = self.STC
            ## Surplus Surface Runoff
            SurRun_o = STmi_1 - self.STC
            if type(STmi_1) != float:
                SurRun_o[SurRun_o < 0] = 0.0
            else:
                SurRun_o = 0.0
            SurRun_i = STmi - self.STC
            SurRun_i[SurRun_i < 0] = 0.0
            SR = (SurRun_i * rfactor) + (SurRun_o * (1-rfactor))
            surp_r.append(SR)
            smstor.append(STi)
            AET_l.append(AET)
            snowstor.append(snow_storage)
            snwmlt.append(snwm)
        ST = np.ma.array(smstor, mask=self.P[time_offset:len(dates),:,:].mask)
        ST_Int = np.ma.array(sm_in, mask=self.P[time_offset:len(dates),:,:].mask)
        AET = np.ma.array(AET_l, mask=self.P[time_offset:len(dates),:,:].mask)
        SR = np.ma.array(surp_r, mask=self.P[time_offset:len(dates),:,:].mask)
        SnwStor = np.ma.array(snowstor, mask=self.P[time_offset:len(dates),:,:].mask)
        snowmlt = np.ma.array(snwmlt, mask=self.P[time_offset:len(dates),:,:].mask)

        return dict(SoilStor=ST, Interm_SoilStor=ST_Int, AET=AET, SurplusRunoff=SR, SnowStorage=SnwStor, Snowmelt=snowmlt)
    
    def _model_warmup(self, dates, time_offset, Tsnow, Train, drf, alpha, rfactor):
        RES = self._soilmoisture_storage(dates, time_offset, Tsnow, Train, drf, alpha, rfactor)
        return (RES['SoilStor'][-1,:,:], RES['SnowStorage'][-1,:,:], RES['Interm_SoilStor'][-1,:,:])
        
    def run_model(self, Tsnow=None, Train=None, drf=None, alpha=None, rfactor=None):
        kwargs = dict(Tsnow=Tsnow,
                      Train=Train,
                      drf=drf,
                      alpha=alpha,
                      rfactor=rfactor)
        
        selfkwargs = dict(Tsnow=self.snotemp,
                          Train=self.raintemp,
                          drf=self.DRF,
                          alpha=self.alpha,
                          rfactor=self.rfactor)
        
        for key in kwargs:
            if kwargs[key] == None:
                kwargs[key] = selfkwargs[key]
                 
        warmup_dates = self.dates[0:self.init_per]
        warmup_off = np.where(self.dates == warmup_dates[0])[0][0]

        w_args = [warmup_dates, warmup_off, kwargs['Tsnow'], kwargs['Train'], kwargs['drf'], kwargs['alpha'], kwargs['rfactor']]
        # w_args = dict(dates=warmup_dates, time_offset=warmup_off)

        args = [self.dates, 0, kwargs['Tsnow'], kwargs['Train'], kwargs['drf'], kwargs['alpha'], kwargs['rfactor']]
        
        # wf_kwargs = w_args.update(kwargs)
        # f_kwargs = args.update(kwargs)
        
        # model warm up
        print("Initializing Model")
        self.STo, self.snw_st_init, self.ST_inter = self._model_warmup(*w_args)
        # model run
        print("Running Model")
        Reslt = self._soilmoisture_storage(*args)
        # assign final arrays
        self.AET = Reslt['AET']
        self.soil_m = Reslt['SoilStor']
        self.surplus_runoff = Reslt['SurplusRunoff']
        self.snow_stor = Reslt['SnowStorage']
        self.snowmelt = Reslt['Snowmelt']
        # compute total runoff
        Tot_Runoff = self.surplus_runoff + self.direct_runoff
        self.runoff = Tot_Runoff
        # convert to streamflow in cms
        Qarrays = ((Tot_Runoff/1000) * 30**2) / (86400 * self.days_in_mnth.values[:,None, None])
        Qvec = Qarrays.sum(axis=(1,2))
        Qs = pd.Series(Qvec, index=self.dates)
        return Qs
    
    def calibrate(self, st_date, end_date, p, pmin, m=None, q=None, alpha=None, beta=None):
        """
        Calibrates Water Balance Model based on streamflow data using
        SCE-UA algorithm [see Duan et al. (1994)]
        
        n = number of parameters being calibrated (in this case 5)

        Parameters
        ----------
        st_date : Start date of the calibration period (YYYY-MM-DD)
        end_date : End date of the calibration period (YYYY-MM-DD)
        p     : Number of Complexes.
        pmin  : Minimum number of Complexes.
        m     : Number of points in a Complex (default = 2n+1).
        q     : Number of points in Subcomplex, ie sampled from Complex (default n+1).
        alpha : Number of consecutive offspring produced by a Subcomplex (default = 1).
        beta  : Number of evolution steps taken by each Complex (default = 2n+1)

        Returns
        -------
        Pandas Series of streamflow for best model run

        """
        
        n = 5
        p = p
        pmin = pmin
        
        if m == None:
            m = 2*n + 1
        else:
            m = m
        
        if q == None:
            q = n + 1
        else:
            q = q
        
        if alpha == None:
            alpha = 1
        else:
            alpha = alpha
        
        if beta == None:
            beta = 2*n + 1
        else:
            beta = beta
        
        s = p*m
        
        sno_t = np.round(np.random.default_rng().uniform(self.snotemp_rng[0],self.snotemp_rng[1],s),0)
        rain_t = np.round(np.random.default_rng().uniform(self.raintemp_rng[0],self.raintemp_rng[1],s),0)
        drf = np.random.default_rng().uniform(self.DRF_rng[0],self.DRF_rng[1],s)
        alpha = np.random.default_rng().uniform(self.alpha_rng[0],self.alpha_rng[1],s)
        rfac = np.random.default_rng().uniform(self.rfactor_rng[0],self.rfactor_rng[1],s)
        
        params = list(zip(sno_t, rain_t, drf, alpha, rfac))
        
        NaSut = []
        for param in params:
            Result = self.run_model(Tsnow=param[0], 
                                    Train=param[1], 
                                    drf=param[2], 
                                    alpha=param[3], 
                                    rfactor=param[4])
            Comp = pd.concat([self.monthly_Q, Result], axis=1)
            Compna = Comp.dropna()
            SSE = (Comp.iloc[:,0] - Comp.iloc[:,1])**2
            DEN = (Comp.iloc[:,0] - Comp.iloc[:,0].mean())**2
            E = 1.0 - (SSE.sum()/DEN.sum())
            NaSut.append(E)
        
        Fun_res = pd.DataFrame()
        ## make dataframe of results
        ## sort based on NatSut
        ## populate complexes
        ## sample complexes for sub-complexes using trapezoidal/triangle distribution
        
        
        warmup_dates = self.dates[0:self.init_per]
        warmup_off = np.where(dts == warmup_dates[0])[0][0]
        calib_dates = self.monthly_Q.index
        cal_off = np.where(dts == calib_dates[0])[0][0]
        #for loop
        for i in range(len(sno_t)):
            w_kwargs = dict(dates=warmup_dates, 
                            time_offset=warmup_off, 
                            Tsnow=sno_t[i],
                            Train=rain_t[i],
                            drf=drf[i],
                            alpha=alpha[i])
        # model warm up
            self.STo = _model_warmup(**w_kwargs)
        # model run
            w_kwargs = dict(dates=calib_dates, 
                            time_offset=cal_off, 
                            Tsnow=sno_t[i],
                            Train=rain_t[i],
                            drf=drf[i],
                            alpha=alpha[i])
            ST, AET
        # calculate model fit
        
        
mod = WB_Model(moddict, 50, 46.739440, 12)
qf = mod.run_model()


 
n1 = np.array([[[1,1],[1,1]],[[1,1],[1,1]]])
n2 = np.array([2,3])
n3 = n1*n2[:,None,None]
s = n3.sum(axis=(1,2))
