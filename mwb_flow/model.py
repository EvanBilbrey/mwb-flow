# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:09:21 2020

DNRC Monthly Runoff Model
@author: Todd Blythe - Hydrologist
"""

# TODO Update/add documentation
######## Gray and McCabe WaterBalance Model ########

### Packages ###
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm


class WaterBalanceModel:
    """
    monthly water balance model
    :param data: xr.Dataset - model input file returned from datafile.CreateInputFile()
    :param init_steps: int - number of time steps used in model warm up
    :param soil_storage_cap: float - soil storage capacity in mm
    :param snow_temp: float - air temperature in degrees Celsius in which snowfall occurs below
    :param rain_temp: float - air temperature in degrees Celsius in which rainfall occurs above
    :param direct_runoff_frac: float - decimal fraction of rainfall that becomes direct runoff
    :param melt_rate: float - melt rate coefficient as a decimal fraction
    :param surplus_runoff_fac: float - decimal fraction of surplus storage that becomes surplus runoff
    """

    # TODO change __init__ parameters to accept array of parameter values for each location
    def __init__(self, data, init_steps=12, soil_storage_cap=75, snow_temp=0, rain_temp=5, direct_runoff_frac=0.06,
                 melt_rate=0.6, surplus_runoff_fac=0.5):
        # initialization arguments
        self.init_steps = init_steps
        self.soilstoragecap = np.repeat(soil_storage_cap, len(data.coords["location"].values))
        self.snowtemp = np.repeat(snow_temp, len(data.coords["location"].values))
        self.raintemp = np.repeat(rain_temp, len(data.coords["location"].values))
        self.directrunofffrac = np.repeat(direct_runoff_frac, len(data.coords["location"].values))
        self.meltrate = np.repeat(melt_rate, len(data.coords["location"].values))
        self.surplusrunofffact = np.repeat(surplus_runoff_fac, len(data.coords["location"].values))
        # data inputs
        self.latitude = data.coords["lat"].values
        self.locations = data['location'].values
        self.times = pd.to_datetime(data.coords["time"].values)
        self.days_in_mnth = pd.to_datetime(data.coords["time"].values).daysinmonth.to_numpy()
        self.t = data.mo_temp.values
        self.p = data.mo_precip.values
        self.q = data.mo_dischg.values
        # soil snow and surplus t0 values
        self.soil_st_init = np.repeat(0, len(data.coords["location"].values))
        self.snow_st_init = np.repeat(0, len(data.coords["location"].values))
        self.surp_st_init = np.repeat(0, len(data.coords["location"].values))
        # calibration parameter ranges
        self.soilstoragecap_rng = None
        self.snowtemp_rng = [-10, 0]
        self.raintemp_rng = [0, 10]
        self.directrunofffrac_rng = [0.01, 0.99]
        self.meltrate_rng = [0.01, 0.99]
        self.surplusrunofffact_rng = [0.01, 0.99]
        # method output variables
        self.soil_capacity = None
        self.snowfall = None
        self.rain = None
        self.direct_runoff = None
        self.snowmelt = None
        self.pet = None
        # model output
        self.aet_result = None
        self.soil_storage_result = None
        self.snow_storage_result = None
        self.snow_storage_result = None
        self.snowmelt_result = None
        self.surplus_runoff_result = None
        self.total_runoff_result = None

    @staticmethod
    def _soil_capacity(sc):
        return sc

    def _snowfall(self, ts, tr):
        # format ts and tr to 2d array
        ts_arr = np.tile(ts, (len(self.times), 1)) * 1.0
        tr_arr = np.tile(tr, (len(self.times), 1)) * 1.0
        # index arrays for snow mixed and rain
        s_ind = np.where(self.t <= ts_arr)
        m_ind = np.where((self.t > ts_arr) & (self.t < tr))
        r_ind = np.where(self.t >= tr_arr)
        # calculate snowfall
        ps = self.p * 1.0
        ps[s_ind] = ps[s_ind]
        ps[m_ind] = ps[m_ind] * ((tr_arr[m_ind] - self.t[m_ind]) / (tr_arr[m_ind] - ts_arr[m_ind]))
        ps[r_ind] = 0.0
        return ps

    def _rain(self):
        pr = self.p - self.snowfall
        return pr

    def _direct_runoff(self, drf):
        dir_ro = drf * self.rain
        return dir_ro

    def _snowmelt(self, mr, ts):
        # calculate snowmelt from degree days method
        sm = mr * (self.t - ts) * np.transpose(np.tile(self.days_in_mnth, (len(self.latitude), 1)))
        sm[sm < 0.0] = 0.0
        return sm

    def _day_length(self):
        rad_lat = np.tile(np.radians(self.latitude), (len(self.times), 1))
        days = self.times - pd.Timedelta(15, unit='D')
        j = np.transpose(np.tile(days.dayofyear.values, (len(self.latitude), 1)))
        decimal = 0.409 * np.sin((2 * np.pi / 365) * j - 1.39)
        sunset_ang = np.arccos(-np.tan(rad_lat) * np.tan(decimal))
        day_len = (24.0 / np.pi) * sunset_ang
        return day_len

    def _pet(self):
        # did not use Gray and McGabe formula because what they present for
        # water vapor density produces incorrect values (see Lu et al. 2005)
        e_sat = 6.108 * np.exp((17.27 * self.t) / (self.t + 237.3))
        rho_sat = 216.7 * (e_sat / (self.t + 273.3))
        d = self._day_length()
        k_pec = 1.0
        pet = (0.1651 * (d / 12.0) * rho_sat * k_pec * np.transpose(np.tile(self.days_in_mnth, (len(self.latitude), 1))))
        return pet

    def _soilmoisture_storage(self, dates, time_offset, slst_c, snow_t, rain_t, dir_f, mlt_r, srp_f):
        # calculate method variables
        self.soil_capacity = self._soil_capacity(sc=slst_c)
        self.snowfall = self._snowfall(ts=snow_t, tr=rain_t)
        self.rain = self._rain()
        self.direct_runoff = self._direct_runoff(drf=dir_f)
        self.snowmelt = self._snowmelt(mr=mlt_r, ts=snow_t)
        self.pet = self._pet()

        # create soil snow and surplus storage lists
        soil_stor = []
        snow_stor = []
        surp_stor = []
        # create snowmelt aet and surplus runoff lists
        snwmlt_list = []
        aet_list = []
        surpro_list = []

        # loop through time steps
        for i in np.arange(time_offset, len(dates)):
            if i == time_offset:
                soil_i1 = self.soil_st_init
                snow_i1 = self.snow_st_init
                surp_i1 = self.surp_st_init
            else:
                if time_offset == 0:
                    soil_i1 = soil_stor[i - 1]
                    snow_i1 = snow_stor[i - 1]
                    surp_i1 = surp_stor[i - 1]
                elif time_offset == self.init_steps:
                    soil_i1 = soil_stor[i - (self.init_steps + 1)]
                    snow_i1 = snow_stor[i - (self.init_steps + 1)]
                    surp_i1 = surp_stor[i - (self.init_steps + 1)]
                else:
                    print(f"Error in warmup argument. Evaluate row {i}")

            # calculate snowmelt
            snow_i = (snow_i1 + self.snowfall[i, :]) - self.snowmelt[i, :]
            hi_snwm = np.where(snow_i > 0.0)
            lo_snwm = np.where(snow_i <= 0.0)
            snwm = self.snowmelt[i, :] * 1.0
            snwm[hi_snwm] = self.snowmelt[i, :][hi_snwm]
            snwm[lo_snwm] = snow_i1[lo_snwm] + self.snowfall[i, :][lo_snwm]
            snow_i[snow_i < 0.0] = 0.0

            # calculate p_tot and aet
            # Premain is nested in this eq. Error in paper
            p_tot = (self.rain[i, :] - self.direct_runoff[i, :]) + snwm
            hi_pet = np.where(p_tot >= self.pet[i, :])
            lo_pet = np.where(p_tot < self.pet[i, :])

            # calculate aet for hi_pet scenarios
            aet = self.pet[i, :] * 1.0
            soil_i = soil_i1 * 1.0

            # calculate soil storage for hi_pet scenarios
            aet[hi_pet] = aet[hi_pet]
            soil_i[hi_pet] = soil_i1[hi_pet] + (p_tot[hi_pet] - aet[hi_pet])
            # calculate soil_i for lo_pet scenarios
            soil_i[lo_pet] = soil_i1[lo_pet] - (abs(p_tot[lo_pet] - self.pet[i, :][lo_pet]) * (
                        soil_i1[lo_pet] / self.soil_capacity[lo_pet]))
            soil_i[soil_i < 0] = 0.0

            # calculate surplus storage and surplus runoff
            surp_i = surp_i1 * 1.0
            hi_soil = np.where(soil_i > self.soil_capacity)
            lo_soil = np.where(soil_i <= self.soil_capacity)
            surp_i[hi_soil] = soil_i[hi_soil] - self.soil_capacity[hi_soil]
            soil_i[hi_soil] = self.soil_capacity[hi_soil]
            surp_i[lo_soil] = 0.0
            surp_ro = (surp_i + surp_i1) * srp_f
            surp_i = (surp_i + surp_i1) - surp_ro

            # calculate aet for lo_pet scenarios
            soil_w = soil_i1[lo_pet] - soil_i[lo_pet]
            aet[lo_pet] = p_tot[lo_pet] + soil_w

            # append storage and results lists
            soil_stor.append(soil_i)
            snow_stor.append(snow_i)
            surp_stor.append(surp_i)
            snwmlt_list.append(snwm)
            aet_list.append(aet)
            surpro_list.append(surp_ro)

        res = xr.Dataset(
            {
                "soil_storage": (
                    ['time', 'location'], np.stack(soil_stor, axis=0),
                    {'standard_name': 'soil storage', 'units': 'mm'}),
                "surplus_storage": (
                    ['time', 'location'], np.stack(surp_stor, axis=0),
                    {'standard_name': 'surplus storage', 'units': 'mm'}),
                "snow_storage": (
                    ['time', 'location'], np.stack(snow_stor, axis=0),
                    {'standard_name': 'snow_storage', 'units': 'mm'}),
                "snowmelt": (
                    ['time', 'location'], np.stack(snwmlt_list, axis=0), {'standard_name': 'snowmelt', 'units': 'mm'}),
                "AET": (['time', 'location'], np.stack(aet_list, axis=0), {'standard_name': 'AET', 'units': 'mm'}),
                "surplus_runoff": (
                    ['time', 'location'], np.stack(surpro_list, axis=0),
                    {'standard_name': 'surplus_runoff', 'units': 'mm'})
            },
            coords={
                "location": (
                    ['location'], self.locations, {'long_name': 'location_identifier', 'cf_role': 'timeseries_id'}),
                "time": self.times[time_offset:len(dates)]
            },
            attrs={
                "featureType": 'timeSeries',
            }
        )
        return res

    def _model_warmup(self, dates, time_offset, slst_c, snow_t, rain_t, dir_f, mlt_r, srp_f):
        res = self._soilmoisture_storage(dates, time_offset, slst_c, snow_t, rain_t, dir_f, mlt_r, srp_f)
        return res['soil_storage'].values[-1, :], res['snow_storage'].values[-1, :], res['surplus_storage'].values[-1,
                                                                                     :]

    def run_model(self, slst_c=None, snow_t=None, rain_t=None, dir_f=None, mlt_r=None, srp_f=None) -> xr.Dataset:
        """
        run the monthly water balance model
        :return: xr.Dataset - water balance members
        """
        kwargs = dict(slst_c=slst_c, snow_t=snow_t, rain_t=rain_t, dir_f=dir_f, mlt_r=mlt_r, srp_f=srp_f)
        self_kwargs = dict(slst_c=self.soilstoragecap, snow_t=self.snowtemp, rain_t=self.raintemp,
                           dir_f=self.directrunofffrac, mlt_r=self.meltrate, srp_f=self.surplusrunofffact)

        for key in kwargs:
            if kwargs[key] is None:
                kwargs[key] = self_kwargs[key]
        # warm up and model arguments
        warmup_dates = self.times[0: self.init_steps]
        w_args = [warmup_dates, 0, kwargs['slst_c'], kwargs['snow_t'], kwargs['rain_t'], kwargs['dir_f'],
                  kwargs['mlt_r'], kwargs['srp_f']]
        args = [self.times, self.init_steps, kwargs['slst_c'], kwargs['snow_t'], kwargs['rain_t'], kwargs['dir_f'],
                kwargs['mlt_r'], kwargs['srp_f']]

        # run warm up to calculate soil snow and surplus storage starting values
        self.soil_st_init, self.snow_st_init, self.surp_st_init = self._model_warmup(*w_args)

        # model run
        result = self._soilmoisture_storage(*args)
        self.aet_result = result['AET'].values
        self.soil_storage_result = result['soil_storage'].values
        self.surplus_runoff_result = result['surplus_runoff'].values
        self.snow_storage_result = result['snow_storage'].values
        self.snowmelt_result = result['snowmelt'].values

        # calculate total runoff from results
        ro_tot = self.surplus_runoff_result + self.direct_runoff[self.init_steps: len(self.times)]
        self.total_runoff_result = ro_tot
        ro_tot_arr = xr.DataArray(ro_tot, coords=result.coords, attrs={'standard_name': 'Total Runoff', 'units': 'mm'})
        ro_tot_arr.name = 'total_runoff'

        p = self.p[self.init_steps: len(self.times)]
        p_arr = xr.DataArray(p, coords=result.coords, attrs={'standard_name': 'Precipitation', 'units': 'mm'})
        p_arr.name = 'precipitation'
        result_xds = xr.merge([result, ro_tot_arr, p_arr])

        return result_xds
