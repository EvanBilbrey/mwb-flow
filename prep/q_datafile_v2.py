import xarray as xr
import numpy as np
import pandas as pd
import os


# Function to make lists of file names and stream gage number identifiers.
def make_qfile_lists(directory):
    # create lists of file names and Id labels
    file_list = []
    label_list = []
    for file in os.listdir(directory):
        if ".csv" in file:
            file_list.append(file)
            # TODO write code to slice the full gage number from files
            label_list.append(file[6:10])
    return file_list, label_list

# # Test make_qfile_list function
# file_list, label_list = make_qfile_lists(directory=r'C:\Users\CND905\Downloaded_Programs\mwb_flow\Examples\data\discharge _copy')
# print(file_list)
# print(label_list)


# Function to format stream gage data .cvs files downloaded from DNRC website
def format_qdata(data, start_date, end_date, loc_label):
    df = data
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
    # loc_label needs update
    location = [str(loc_label)] * len(df)
    df["location"] = location

    return df

# # Test format_qdata function
# test_data = pd.read_csv(os.path.join(dir, file_list[0]), skiprows=1, usecols=[0, 1])
# test_output = format_qdata(data=test_data, start_date="2016-01-01", end_date="2016-12-31", loc_label=label_list[0])
# print(test_output)













file_list, label_list = make_qfile_lists(directory=r'C:\Users\CND905\Downloaded_Programs\mwb_flow\Examples\data\discharge _copy')

q_data = []
for i in range(0, len(file_list)):
    df = pd.read_csv(os.path.join(dir, file_list[i]), skiprows=1, usecols=[0, 1])
    df = format_qdata(data=df, start_date="2016-01-01", end_date="2016-12-31", loc_label=label_list[i])
    q_data.append(df)

q_data = pd.concat(q_data, axis=0, ignore_index=True)

# convert to a xarray DataArray
q_data = q_data.set_index(["time", "location"])
q_data = q_data.to_xarray()
q_data = xr.DataArray(q_data.discharge, attrs={'standard_name': 'Monthly Discharge', 'units': 'm^3/s'})
q_data.name = "mo_discharge"
print(q_data)

# # Save as .nc file until above code can be used as class to make object
# q_data.to_netcdf('q_datafile_output.nc')
