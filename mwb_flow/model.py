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
from tqdm import tqdm


#TODO edit code to accept xarray.datasets instead of a dictionary
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
        # TODO edit code to accept xarray.datasets instead of a dictionary
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


 
### Model Computations ###
## Model, input is dictionary similar to previous mdat dictionary
## = {temp, precip, mon_streamflow, dates (months)} where temp and precip
##    are ndarrays, dates is a vectors/1d array, and mon_Q is
##    a pandas series with dates as index and flow in m^3/s
moddict = {'temp': MData.temp_arr,
           'precip': MData.prec_arr,
           'dates': MData.dts,
           'mon_Q': qflw}
