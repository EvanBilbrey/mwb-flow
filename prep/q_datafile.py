import xarray as xr
import numpy as np
import pandas as pd
import os




# TODO make a class for discharge data with arguments for datasource and date ranges
start_date = "2016-08-01"
end_date = "2016-11-30"
dir = r'C:\Users\CND905\Downloaded_Programs\mwb_flow\Examples\data\discharge _copy'

# create lists of file names and Id labels
file_list = []
label_list = []
for file in os.listdir(dir):
    if ".csv" in file:
        file_list.append(file)
        # TODO write code to slice the full gage number from files
        label_list.append(file[6:10])

# open files by looping through file_list then formatting into monthly discharge by location
q_data = []
for i in range(0, len(file_list)):

    df = pd.read_csv(os.path.join(dir, file_list[i]), skiprows=1, usecols=[0, 1])

    # make column headers uniform for all gages
    df.rename(columns={list(df)[0]: "time", list(df)[1]: "discharge"}, inplace=True)

    # convert time column to correct data type
    df["time"] = df["time"].astype("datetime64[ns]")

    # convert ft^3/s to m^3/s
    df["discharge"] = df["discharge"] / 35.3146667

    # convert to monthly average
    df = df.resample('MS', on="time").mean()
    df = df.reset_index()

    # add Start and Stop dates for each dataset
    # TODO create method to flag months with low data availability
    mask = (df["time"] >= start_date) & (df["time"] <= end_date)
    df = df.loc[mask]

    # add a column for gage Id that matches the file name
    location = [str(label_list[i])] * len(df)
    df["location"] = location
    df["location"] = df["location"].astype(int)

    # combine all df
    q_data.append(df)


q_data = pd.concat(q_data, axis=0, ignore_index=True)

# convert to a xarray DataArray
q_data = q_data.set_index(["time", "location"])
q_data = q_data.to_xarray()
q_data = xr.DataArray(q_data.discharge, attrs={'standard_name': 'Monthly Discharge', 'units': 'm^3/s'})
q_data.name = "mo_discharge"
print(q_data)

# Save as .nc file until above code can be used as class to make object
q_data.to_netcdf('q_datafile_output.nc')





