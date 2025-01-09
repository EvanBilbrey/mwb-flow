# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:09:21 2020

DNRC Monthly Runoff Model
@author: Todd Blythe - Hydrologist
"""
# TODO Need to double check all math and results for correctness
# TODO Update/add documentation
######## Gray and McCabe WaterBalance Model ########
### Version 1.0 ###
### Lolo Creek Specific ###

### Packages ###
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm


class WB_Model():
    # TODO change __init__ parameters to accept array of parameter values for each location
    def __init__(self, data_arr, init_per=12, soil_stor_cap=75, temp_sno=0,
                 temp_rain=5, dr_frac=0.06, mlt_rate=0.6, sr_factor=0.5):
        self.STC = np.repeat(soil_stor_cap, len(data_arr.coords["location"].values))
        self.snotemp = np.repeat(temp_sno, len(data_arr.coords["location"].values))
        self.raintemp = np.repeat(temp_rain, len(data_arr.coords["location"].values))
        self.DRF = np.repeat(dr_frac, len(data_arr.coords["location"].values))
        self.melt_coef = np.repeat(mlt_rate, len(data_arr.coords["location"].values))
        self.SRF = np.repeat(sr_factor, len(data_arr.coords["location"].values))
        self.latitude = data_arr.coords["lat"].values
        self.init_per = init_per
        self.STo = np.repeat(0, len(data_arr.coords["location"].values))
        self.snw_st_init = np.repeat(0, len(data_arr.coords["location"].values))
        self.Sur_init = np.repeat(0, len(data_arr.coords["location"].values))
        
        # Calibration Parameter Ranges
        self.snotemp_rng = [-10, 0]
        self.raintemp_rng = [0, 10]
        self.DRF_rng = [0.01, 0.99]
        self.melt_coef_rng = [0.01, 0.99]
        self.SRF_rng = [0.01, 0.99]
        self.T = data_arr.mo_temp.values
        self.P = data_arr.mo_precip.values
        self.val_Q = data_arr.mo_discharge.values
        self.locations = data_arr['location'].values
        self.times = pd.to_datetime(data_arr.coords["time"].values)
        self.days_in_mnth = pd.to_datetime(data_arr.coords["time"].values).daysinmonth.to_numpy()

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
        self.total_runoff = None

    def _day_length(self):
        rad_lat = np.tile(np.radians(self.latitude), (len(self.times), 1))
        days = self.times - pd.Timedelta(15, unit='D')
        j = np.transpose(np.tile(days.dayofyear.values, (len(self.latitude), 1)))
        decimal = 0.409*np.sin((2*np.pi/365)*j - 1.39)
        sunset_ang = np.arccos(-np.tan(rad_lat)*np.tan(decimal))
        day_len = (24/np.pi)*sunset_ang
        return day_len

    def _snow_input(self, Tsnow, Train):
        Tsnow_arr = np.tile(Tsnow, (len(self.times), 1))
        Train_arr = np.tile(Train, (len(self.times), 1))

        sno_ind = np.where(self.T <= Tsnow_arr)
        mix_ind = np.where((self.T > Tsnow_arr) & (self.T < Train))
        rain_ind = np.where(self.T >= Train_arr)
        Psnow = self.P * 1.0  # Create 2D array and replace if T is greater than Tsnow
        Psnow[sno_ind] = Psnow[sno_ind]
        Psnow[mix_ind] = Psnow[mix_ind] * (
                    (Train_arr[mix_ind] - self.T[mix_ind]) / (Train_arr[mix_ind] - Tsnow_arr[mix_ind]))
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
        Sno_M = alpha * (self.T - Tsnow) * np.transpose(np.tile(self.days_in_mnth, (len(self.latitude), 1)))
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
        PET = (0.1651 * (D / 12) * rho_sat * Kpec * np.transpose(np.tile(self.days_in_mnth, (len(self.latitude), 1))))
        return PET

    def _soilmoisture_storage(self, dates, time_offset, Tsnow, Train, drf, alpha, rfactor):
        self.snowfall = self._snow_input(Tsnow, Train)
        self.rain = self._rain()
        self.direct_runoff = self._direct_runoff(drf)
        self.P_snow_melt = self._snowmelt(alpha, Tsnow)
        self.PET = self._PET()

        smstor = []  # soil moisture storage
        snowstor = []  # snow storage
        surstor = []  # surplus storage
        snwmlt = []  # snowmelt
        AET_l = []  # AET list
        surplus_r = []  # surplus runoff

        for i in tqdm(np.arange(time_offset, len(dates))):
            if i == time_offset:
                STi_1 = self.STo
                sno_stori_1 = self.snw_st_init
                Suri_1 = self.Sur_init
            else:
                if time_offset == 0:
                    STi_1 = smstor[i - 1]
                    sno_stori_1 = snowstor[i - 1]
                    Suri_1 = surstor[i - 1]
                elif time_offset == self.init_per:
                    STi_1 = smstor[i - (self.init_per + 1)]
                    sno_stori_1 = snowstor[i - (self.init_per + 1)]
                    Suri_1 = surstor[i - (self.init_per + 1)]
                else:
                    print(f"Error in warmup argument. Evaluate row {i}")
            
            # Liquid water input to Soil Moisture Storage
            snow_storage = sno_stori_1 + self.snowfall[i, :] - self.P_snow_melt[i, :]
            hi_snwm = np.where(snow_storage > 0.0)
            lo_snwm = np.where(snow_storage <= 0.0)

            snwm = self.P_snow_melt[i, :] * 1.0
            snwm[hi_snwm] = self.P_snow_melt[i, :][hi_snwm]
            snwm[lo_snwm] = sno_stori_1[lo_snwm] + self.snowfall[i, :][lo_snwm]
            snow_storage[snow_storage < 0.0] = 0.0

            # Ptotal calc
            P_tot = (self.rain[i, :] - self.direct_runoff[i, :]) + snwm  # Premain is nested in this eq. Error in paper.
            hi_pet = np.where(P_tot >= self.PET[i, :])
            lo_pet = np.where(P_tot < self.PET[i, :])

            # AET and ST calc
            AET = self.PET[i, :] * 1.0
            STi = STi_1 * 1.0

            ## High AET
            AET[hi_pet] = AET[hi_pet]
            STi[hi_pet] = STi_1[hi_pet] + (P_tot[hi_pet] - AET[hi_pet])
            ## Low AET
            STi[lo_pet] = STi_1[lo_pet] - (abs(P_tot[lo_pet] - self.PET[i, :][lo_pet]) * (
                        STi_1[lo_pet] / self.STC[lo_pet]))  # STi_1 starts as 0 so will always be 0 if P_tot is < PET
            STi[STi < 0] = 0
            STW = STi_1[lo_pet] - STi[lo_pet]
            AET[lo_pet] = P_tot[lo_pet] + STW

            # Surplus Runoff
            Suri = np.repeat(0, len(self.locations))
            hi_s = np.where(STi > self.STC)
            Suri[hi_s] = STi[hi_s] - self.STC[hi_s]
            Sur_RO = (Suri + Suri_1) * rfactor  # Calculate surplus runoff for month
            Suri = Suri + Suri_1 - Sur_RO  # Calculate remaining runoff for month

            smstor.append(STi)
            surstor.append(Suri)
            snowstor.append(snow_storage)
            snwmlt.append(snwm)
            AET_l.append(AET)
            surplus_r.append(Sur_RO)

        soil_storage = np.stack(smstor, axis=0)
        surplus_storage = np.stack(surstor, axis=0)
        snow_storage = np.stack(snowstor, axis=0)
        snowmelt = np.stack(snwmlt, axis=0)
        AET = np.stack(AET_l, axis=0)
        surplus_runoff = np.stack(surplus_r, axis=0)

        res = xr.Dataset(
            {
                "soil_storage": (['time', 'location'], soil_storage, {'standard_name': 'soil storage',
                                                                      'units': 'mm'}),
                "surplus_storage": (['time', 'location'], surplus_storage, {'standard_name': 'surplus storage',
                                                                            'units': 'mm'}),
                "snow_storage": (['time', 'location'], snow_storage, {'standard_name': 'snow_storage',
                                                                      'units': 'mm'}),
                "snowmelt": (['time', 'location'], snowmelt, {'standard_name': 'snowmelt',
                                                              'units': 'mm'}),
                "AET": (['time', 'location'], AET, {'standard_name': 'AET',
                                                    'units': 'mm'}),
                "surplus_runoff": (['time', 'location'], surplus_runoff, {'standard_name': 'surplus_runoff',
                                                                          'units': 'mm'})

            },
            coords={
                "location": (['location'], self.locations, {'long_name': 'location_identifier',
                                                                         'cf_role': 'timeseries_id'}),
                # Keep the order of xds
                "time": self.times[time_offset:len(dates)]
            },
            attrs={
                "featureType": 'timeSeries',
            }
        )

        return res

    def _model_warmup(self, dates, time_offset, Tsnow, Train, drf, alpha, rfactor):
        RES = self._soilmoisture_storage(dates, time_offset, Tsnow, Train, drf, alpha, rfactor)
        return RES['soil_storage'].values[-1,:], RES['snow_storage'].values[-1,:], RES['surplus_storage'].values[-1,:]
        
    def run_model(self, Tsnow=None, Train=None, drf=None, alpha=None, rfactor=None):
        kwargs = dict(Tsnow=Tsnow,
                      Train=Train,
                      drf=drf,
                      alpha=alpha,
                      rfactor=rfactor)
        
        selfkwargs = dict(Tsnow=self.snotemp,
                          Train=self.raintemp,
                          drf=self.DRF,
                          alpha=self.melt_coef,
                          rfactor=self.SRF)
        
        for key in kwargs:
            if kwargs[key] == None:
                kwargs[key] = selfkwargs[key]
                 
        warmup_dates = self.times[0:self.init_per]

        w_args = [warmup_dates, 0, kwargs['Tsnow'], kwargs['Train'], kwargs['drf'], kwargs['alpha'], kwargs['rfactor']]
        args = [self.times, self.init_per, kwargs['Tsnow'], kwargs['Train'], kwargs['drf'], kwargs['alpha'], kwargs['rfactor']]

        # model warm up
        print("Initializing Model")
        self.STo, self.snw_st_init, self.Sur_init = self._model_warmup(*w_args)
        # model run
        print("Running Model")
        result = self._soilmoisture_storage(*args)
        # assign final arrays
        self.AET = result['AET'].values
        self.soil_m = result['soil_storage'].values
        self.surplus_runoff = result['surplus_runoff'].values
        self.snow_stor = result['snow_storage'].values
        self.snowmelt = result['snowmelt'].values
        # compute total runoff
        Tot_RO = self.surplus_runoff + self.direct_runoff[self.init_per:len(self.times)]
        self.total_runoff = Tot_RO
        Tot_RO_arr = xr.DataArray(Tot_RO, coords=result.coords, attrs={'standard_name': 'Total Runoff', 'units': 'mm'})
        Tot_RO_arr.name = 'total_runoff'

        p = self.P[self.init_per:len(self.times)]
        p_arr = xr.DataArray(p, coords=result.coords, attrs={'standard_name': 'Precipitation', 'units': 'mm'})
        p_arr.name = 'precipitation'
        result_xds = xr.merge([result, Tot_RO_arr, p_arr])

        return result_xds
