import os
from pathlib import Path
import geopandas as gpd
import xarray as xr

mwb_flow_dir = r'C:\Users\CND905\Downloaded_Programs\mwb_flow'
os.chdir(mwb_flow_dir)

from prep.datafile import CreateInputFile
from prep.metdata import get_gridmet_at_points
from prep.datafile import check_format


exres_pth = Path(r'C:\Users\CND905\Downloaded_Programs\mwb_flow\Examples\data\HWWY_reservoirs_5070.shp')
exres = gpd.read_file(exres_pth)
# this file is in crs 5070, need it to be 4326 for getting GridMET
exres = exres.to_crs(4326)

# This example provides a unique identifier column rather than using the default DataFrame index.
exres_met = get_gridmet_at_points(exres, 'Permanent_', start='2021-01-01', end='2022-12-31')

#Create an object of daily data
day_Idat = CreateInputFile(geoms = None, met_data = exres_met, met_source = 'from_file')

# Calculate daily mean air temperature then followed by the monthly mean air temperature
mean_temp = ((day_Idat.data.min_temp + day_Idat.data.max_temp) / 2) - 273.15  # also convert to Celcius from GridMET native Kelvin
monthly_temp = mean_temp.resample(time = "MS").mean()
# Convert to a DataArray with attributes and title
Monthly_Temp = xr.DataArray(monthly_temp, coords=monthly_temp.coords, attrs={'standard_name': 'Monthly_Temperature', 'units': 'Celcius'})
Monthly_Temp.name = 'monthly_temp'

# Calculate the monthly precipitation
monthly_precip = day_Idat.data.precip.resample(time = "MS").sum()
# Convert to a DataArray with attributes and title
Monthly_Precip = xr.DataArray(monthly_precip, coords=monthly_precip.coords, attrs={'standard_name': 'Monthy Precipitation', 'units': 'mm'})
Monthly_Precip.name = 'monthly_precip'

monthly_Idat = CreateInputFile(geoms=None, met_data=Monthly_Temp, met_source='from_file')
monthly_Idat.add_variable(Monthly_Precip, variable_name='monthly_precip', var_attrs={'standard_name': 'Monthy Precipitation', 'units': 'mm'})

print(monthly_Idat.data)
check_format(monthly_Idat.data)

#monthly_Idat.save_datafile(r'C:\Users\CND905\Downloaded_Programs\mwb-flow\Examples\prep_Example1_PullMetdata_outpu.nc')