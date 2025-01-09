import xarray as xr
import numpy as np
import pandas as pd
import os
import datetime




# TODO make a class for discharge data with arguments for datasource and date ranges
start_date = "2016-01-01"
end_date = "2019-12-31"
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
    print(i)
    df = pd.read_csv(os.path.join(dir, file_list[i]), skiprows=1, usecols=[0, 1])

    # make column headers uniform for all gages
    df.rename(columns={list(df)[0]: "time", list(df)[1]: "q"}, inplace=True)

    # Calculate seconds between consecutive timestamps
    df["time"] = df["time"].astype("datetime64[ns]")
    tdelt = [(df["time"][i+1] - df["time"][i]).seconds for i in range(len(df["time"]) - 1)]
    tdelt.append(tdelt[-1])

    # Calculate monthly volume
    df["q"] = df["q"] / 35.3146667 # convert ft^3/s to m^3/s
    df['q_vol'] = df['q'] * tdelt
    df = df.resample('MS', on="time").sum()
    df = df.drop('q', axis=1)
    df = df.reset_index()
# TODO write code to flag months without full discharge records
    # Add Start and Stop dates for each dataset
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
q_data = xr.DataArray(q_data.q_vol, attrs={'standard_name': 'Monthly Discharge Volume', 'units': 'm^3'})
q_data.name = "mo_discharge_vol"
print(q_data)

# Save as .nc file until above code can be used as class to make object
q_data.to_netcdf('q_datafile_output.nc')





