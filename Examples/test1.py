import os
from pathlib import Path
import geopandas as gpd
import xarray as xr

mwb_flow_dir = r'C:\Users\CND905\Downloaded_Programs\mwb_flow'
os.chdir(mwb_flow_dir)
#
from prep.datafile import CreateInputFile
from prep.metdata import get_gridmet_at_points
from prep.datafile import check_format

exres_pth = Path(r'C:\Users\CND905\Downloaded_Programs\mwb_flow\Examples\data\Lolo_WB_Model_Calibration_Catchments_32611.shp')
exres = gpd.read_file(exres_pth)
# This file is in crs 32611 (WGS84 UTM zone 11N), need it to be 4326 for getting GridMET.
exres = exres.to_crs(4326)

# This example provides a unique identifier column rather than using the default DataFrame index.
exres_met = get_gridmet_at_points(exres, 'gageID', start='2016-01-01', end='2016-12-31')

test = CreateInputFile(geoms=None, met_data=exres_met, met_source='from_file')
test.data

# Calculate daily mean air temperature then followed by the monthly mean air temperature
mean_temp = ((exres_met.min_temp + exres_met.max_temp) / 2) - 273.15  # also convert to Celcius from GridMET native Kelvin
mean_temp
monthly_temp = mean_temp.resample(time = "MS").mean()

# Convert to a DataArray with attributes and title
Monthly_Temp = xr.DataArray(monthly_temp, coords=monthly_temp.coords, attrs={'standard_name': 'Monthly_Temperature', 'units': 'Celcius'})
Monthly_Temp.name = 'mo_temp'
Monthly_Temp

# Calculate the monthly precipitation
monthly_precip = exres_met.precip.resample(time = "MS").sum()

# Convert to a DataArray with attributes and title
Monthly_Precip = xr.DataArray(monthly_precip, coords=monthly_precip.coords, attrs={'standard_name': 'Monthy Precipitation', 'units': 'mm'})
Monthly_Precip.name = 'mo_precip'
Monthly_Precip

metdata_input = xr.merge([Monthly_Temp, Monthly_Precip])
check_format(metdata_input)

Monthly_Q = xr.load_dataarray(r'C:\Users\CND905\Downloaded_Programs\mwb_flow\prep\q_datafile_output.nc')
Monthly_Q
