import pandas as pd
import geopandas as gpd
import xarray as xr
from typing import Union
import warnings

from config import INPUT_VARS
from config import DSET_COORDS
from prep.metdata import get_gridmet_at_points


MET_SOURCES = [
    'gridmet',
    'daymet',
    'from_file'
]


class CreateInputFile:
    """

    Prepares/provides methods to prepare a xarray Dataset for use in the mwb_flow model based on input sites/locations
    where the model will estimate streamflow (e.g., watershed polygon layer in the form of a geopandas GeoDataFrame).
    mwb_flow required inputs are meterology and streamflow at monthly frequencies.
    These datasets are required as input to format the datafile.

    :param geoms: geopandas.GeoDataFrame - geometries for which to compute evaporation estimates (point or polygon)
    :param index_col: str or None(default) - defines the column name in geoms that will be used as the unique identifier for
        each geometry location. None defaults to the GeoDataFrame index.
    :param met_data: xarray.Dataset or None (default) - if Dataset is provided it is used to construct the input
        datafile from file. If None, met_source must be provided to build initial input datafile.
    :param met_source: str - from accepted sources ['gridmet', 'daymet', 'from_file']
    :return: A formatted datafile class that contains necessary info to save a properly formatted mwb_flow input netcdf.
    """

    def __init__(
            self,
            geoms: Union[gpd.GeoDataFrame, None],
            index_col: Union[str, None] = None,
            met_data: Union[xr.Dataset, None] = None,
            met_source = 'gridmet'):

        self.data = self._create_metinputs(geoms, index_col, met_data, met_source)

    def _create_metinputs(self,
                          geoms: gpd.GeoDataFrame,
                          index_col: Union[str, None],
                          met_data: Union[xr.Dataset, None] = None,
                          met_source: str = 'gridmet') -> xr.Dataset:
        """
        Functin to format a .netcdf meteorology file for input into mwb_flow.
        __________________

        Valid met_source = ['gridmet', 'daymet', 'from_file']

        :param geoms: geopandas.GeoDataFrame - geometries where meteorology data will be extracted.
        :param met_data: xarray.Dataset or None(default) - Pre-formatted as discrete sampling locations netcdf with dims
            (time, location) for each variable. Must also contain coordinate
            vars ["lat", "lon", "elev", "location", "time"].
        :param met_source: str - a string representing the source of meteorology data to use include
        :return: xarray.DataSet with meteorology variables necessary for Penman Equation calculations at each geometry
            location in geoms.
        """

        if met_source == 'from_file':
            if met_data is None:
                raise ValueError("met_data was not defined.")
            else:
                metinputs = met_data

        elif met_source == 'gridmet':
            # This is where the addition of start, end dates could be added to provide an alternative to the default
            #   behavior of the met_source = 'gridmet' of downloading the entire GridMET POR.
            metinputs = get_gridmet_at_points(geoms, index_col)

        elif met_source == 'daymet':
            print("Sorry, daymet is not yet available. Defaulting to gridment source.")
            metinputs = get_gridmet_at_points(geoms, index_col)

        else:
            raise ValueError("Meteorology source is invalid. Valid entries:{0}".format(MET_SOURCES))

        return metinputs

    def add_variable(self, data, variable_name, var_attrs=None):
        """
        Function to add a variable to an existing input dataset. Only accepts one variable at a time.
        :param data: pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset - pandas objects must be formatted with
        multiindex levels = [time, location] where location index matches those of the class.data object's location ids.
        Time must be datetime64[ns]. Can also be xarray object formatted identical to the class.data object.
        :param variable_name: str - name to assign to new variable from available variables
            ['precip', 'mo_precip', 'min_temp', 'max_temp', 'mean_temp', 'mo_temp', 'mo_discharge']
        :param var_attrs: dict or None(default) - dictionary of attributes and associated values for the new variable
        (usually at minimum includes 'standard_name' and 'units')
        :return: None - updates class data object with new variable
        """
        accepted_vars = ['precip', 'mo_precip', 'min_temp', 'max_temp', 'mean_temp', 'mo_temp', 'mo_discharge']
        if variable_name not in accepted_vars:
            raise ValueError("Variable name not compatible, select from: "
                             "['precip', 'mo_precip', 'min_temp', 'max_temp', 'mean_temp', 'mo_temp', 'mo_discharge']")

        if isinstance(data, pd.DataFrame):
            data.index.names = ['time', 'location']
            if len(data.columns) > 1:
                print("Too many columns in dataframe, due to ambiguity, no variable was loaded.")
            else:
                print("Variable {0} loaded.".format(data.columns[0]))
                var_nam = data.columns[0]
                if var_nam != variable_name:
                    data.columns = [variable_name]
                    print("Variable {0} renamed to {1}".format(var_nam, variable_name))
                else:
                    print("Variable {0} did not need renaming.".format(var_nam))

                new_ds = xr.merge([self.data, data.to_xarray()])
                new_ds[variable_name].attrs = var_attrs
                self.data = new_ds
                print("New variable added.")
        elif isinstance(data, pd.Series):
            data.index.names = ['time', 'location']
            data = pd.DataFrame(data, columns=[variable_name])
            print("Series loaded and converted to DataFrame with {0} variable.".format(variable_name))
            new_ds = xr.merge([self.data, data.to_xarray()])
            new_ds[variable_name].attrs = var_attrs
            self.data = new_ds
            print("New variable added.")
        elif isinstance(data, xr.Dataset) or isinstance(data, xr.DataArray):
            data.name = variable_name
            new_ds = xr.merge([self.data, data])
            if var_attrs is None:
                self.data = new_ds
                print("New variable added.")
            else:
                new_ds[variable_name].attrs = var_attrs
                self.data = new_ds
                print("New variable added.")
        else:
            print("Data type is neither pandas Series or Dataframe, no variable loaded.")

    def save_datafile(self, pthname):
        self.data.to_netcdf(pthname)


def check_format(xrdset):
    vars = [x for x in INPUT_VARS if x not in list(xrdset.data_vars)]
    coords = [x for x in DSET_COORDS if x not in list(xrdset.coords)]
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

# Default behavior create input datafile from gridmet given static reservoir variables and gridmet POR
if __name__ == '__main__':

    pass
