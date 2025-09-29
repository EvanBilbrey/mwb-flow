import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
from typing import Union
import warnings

from config import INPUT_VARS
from config import DSET_COORDS
from prep.metdata import get_gridmet_at_points


class CreateInputFile:
    """
    class for formatting input data for model.WaterBalanceModel()
    :param m_data: xr.Dataset - dataset returned from metdata.get_gridmet_for_polygons()
    :param q_data: xr.Dataset - daily discharge in cfs in 'discharge' titled DataArray
    """

    def __init__(self, m_data=None, q_data=None):
        self.m_data = m_data
        self.q_data = q_data

        self.data = xr.merge([self.m_data, self.q_data])
        self.input_file = self.format_data()

    def _check_format(self):
        vars = [x for x in INPUT_VARS if x not in list(self.input_file.data_vars)]
        coords = [x for x in DSET_COORDS if x not in list(self.input_file.coords)]
        if len(vars) != 0:
            warnings.warn("There are missing or mislabeled variables in the dataset. See the following:")
            print("MISSING VARIABLES", *vars, sep='\n')
        else:
            print("All necessary variables exist and are labeled properly.")

        if len(coords) != 0:
            warnings.warn("There are missing or mislabeled coordinates in the dataset. See the following:")
            print("MISSING COORDINATES", *coords, sep='\n')
        else:
            print("All necessary coordinates exist and are labeled properly")

    def _format_data(self):

        temp = ((self.data['min_temp'] + self.data['max_temp']) / 2) - 273.15
        mo_temp = temp.resample(time="MS").mean()
        mo_temp.name = 'mo_temp'
        mo_temp = mo_temp.assign_attrs(standard_name='Monthly Temperature', units='Celsius')

        area_m2 = self.data['area'].values * 1000000  # convert polygon areas from km^2 to m^2
        precip = (self.data['precip_volume'] / np.tile(area_m2, (
            len(self.data['time']), 1))) * 1000  # convert precip volume (m^3) to length (m) and then to mm
        mo_precip = precip.resample(time="MS").sum()
        mo_precip.name = 'mo_precip'
        mo_precip = mo_precip.assign_attrs(standard_name='Monthly Precipitation', units='mm')

        dischg = np.round(self.data['discharge'] / 35.3146667,
                          decimals=2) * 86400  # convert mean ft^3/second to m^3/second and then to m^3/day
        dischg = (dischg / np.tile(area_m2, (
            len(dischg['time']), 1))) * 1000  # convert precip volume (m^3) to length (m) and then to mm
        mo_dischg = dischg.resample(time="MS").sum()
        mo_dischg.name = 'mo_dischg'
        mo_dischg = mo_dischg.assign_attrs(standard_name='Monthly Discharge', units='mm')

        output = xr.merge([mo_temp, mo_precip, mo_dischg])

        return output
