import xarray as xr
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
# USGS imports
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import dataretrieval.nwis as nwis

# Pull data from stage gages - Units are CFS
DEFAULT_DATES = ('1979-01-01', (datetime.today() - timedelta(days=2)).strftime("%Y-%m-%d"))


class Get_Flow_Data():

    def __init__(self, start=DEFAULT_DATES[0], end=DEFAULT_DATES[1], source=None, directory=None, site_number=None,
                 data_frequency='iv', parameter_code='00060', summarize_frequency='daily', units='cfs'):

        self.start = start
        self.end = end

        if source in ['stage_file', 'get_usgs']:
            self.source = source
            if source == 'stage_file' and directory is None:
                raise ValueError("directory must be defined when source='stage_file'.")
            elif source == 'get_usgs' and site_number is None:
                raise ValueError("site_number must be defined when source='get_usgs'.")
        else:
            raise ValueError(f"source='{source}' input not compatible, select from: ['stage_file', 'get_usgs']")
        self.dir = directory
        self.sid = site_number.to_list()

        if data_frequency == 'iv' and parameter_code == '00060':
            self.data_freq = data_frequency
            self.paramcd = parameter_code
        else:
            raise ValueError("class is only compatible with frequency='iv' and parameter_code='00060' arguments")

        if summarize_frequency == 'daily':
            self.summ_freq = summarize_frequency
        else:
            raise ValueError("summarize_frequency='daily' input is the only option")
        self.units = units

        self.imported_dataframe = None

    def _get_stage_csv(self):
        file_list = []
        label_list = []
        for file in os.listdir(self.dir):
            if ".csv" in file:
                file_list.append(file)
                # TODO write code to slice the full gage number from files
                label_list.append(file[0:10])
        q_data = []
        for i in range(0, len(file_list)):
            df = pd.read_csv(os.path.join(self.dir, file_list[i]), skiprows=1, usecols=[0, 1])
            df.rename(columns={list(df)[0]: "time", list(df)[1]: "q"}, inplace=True)

            # add a column for gage number that matches the file name
            sid = [str(label_list[i])] * len(df)
            df["siteID"] = sid
            df["siteID"] = df["siteID"].astype(str)

            # combine all df
            q_data.append(df)
        dataframe = pd.concat(q_data, axis=0, ignore_index=True)

        return pd.DataFrame(dataframe)

    def _get_usgs_flows(self):
        site_list = self.sid
        q_data = []
        for site in range(len(site_list)):
            print(f'getting site {site_list[site]}, {self.start} - {self.end}...')
            gage_df = nwis.get_record(sites=site_list[site], service=self.data_freq,
                                 start=self.start, end=self.end, parameterCd=self.paramcd)
            gage_df.index = pd.DatetimeIndex([pd.to_datetime(i, utc=True) for i in gage_df.index])

            if gage_df.empty:
                print(site_list[site], self.start, self.end, ' is empty')
                return None

            freq_map = {'dv': '00060_Mean', 'iv': '00060'}
            q_col = freq_map[self.data_freq]
            gage_df = gage_df.rename(columns={q_col: 'q'})

            gage_df['q'] = np.where(gage_df['q'] < 0, np.zeros_like(gage_df['q']) * np.nan, gage_df['q'])
            gage_df = gage_df[['q']]

            gage_df = gage_df.reset_index()
            gage_df = gage_df.rename(columns={'index': 'time'})
            gage_df["siteID"] = [site_list[site]] * len(gage_df)
            gage_df["siteID"] = gage_df["siteID"].astype(str)

            # combine all df
            q_data.append(gage_df)
        df = pd.concat(q_data, axis=0, ignore_index=True)

        return pd.DataFrame(df)


        # print(f'getting site {self.sid}, {self.start} - {self.end}...')
        # df = nwis.get_record(sites=self.sid, service=self.data_freq,
        #                      start=self.start, end=self.end, parameterCd=self.paramcd)
        # df.index = pd.DatetimeIndex([pd.to_datetime(i, utc=True) for i in df.index])
        #
        # if df.empty:
        #     print(self.sid, self.start, self.end, ' is empty')
        #     return None
        #
        # freq_map = {'dv': '00060_Mean', 'iv': '00060'}
        # q_col = freq_map[self.data_freq]
        # df = df.rename(columns={q_col: 'q'})
        #
        # df['q'] = np.where(df['q'] < 0, np.zeros_like(df['q']) * np.nan, df['q'])
        # df = df[['q']]
        #
        # df = df.reset_index()
        # df = df.rename(columns={'index': 'time'})
        # df["siteID"] = [self.sid] * len(df)
        # df["siteID"] = df["siteID"].astype(str)
        #
        # return pd.DataFrame(df)

    def _format_flow(self):

        # resample discharge my daily_mean or instantaneous_sum. Returns pd.dataframe with q as ft^3
        combine_list = []
        id_list = self.imported_dataframe['siteID'].unique().tolist()
        if self.summ_freq == 'daily':
            for site in range(len(id_list)):
                gage_df = pd.DataFrame(self.imported_dataframe.loc[self.imported_dataframe['siteID'] == id_list[site]])
                gage_df.time = pd.to_datetime(gage_df.time)
                gage_df = gage_df.reset_index().set_index("time").drop(['siteID', 'index'], axis=1)
                gage_df = gage_df.resample('D').mean()
                gage_df['siteID'] = [id_list[site]] * len(gage_df)
                combine_list.append(gage_df)
            df = pd.concat(combine_list, axis=0, ignore_index=False)
            df['q'] = df['q'] * 86400

        elif self.summ_freq == "instant":
            for site in range(len(id_list)):
                gage_df = (pd.DataFrame(self.imported_dataframe.loc[self.imported_dataframe['siteID'] == id_list[site]])
                           .reset_index())
                gage_df.time = pd.to_datetime(gage_df.time)
                tdelt = [(gage_df["time"][i + 1] - gage_df["time"][i]).seconds for i in range(len(gage_df["time"]) - 1)]
                tdelt.append(tdelt[-1])
                gage_df['q'] = gage_df['q'] * tdelt
                combine_list.append(gage_df)
            df = pd.concat(combine_list, axis=0, ignore_index=False)
            df = df.set_index("time").drop(['index'], axis=1)

        else:
            raise ValueError(f"frequency='{self.summ_freq}' is not yet available.")

        # Convert from ft^3 to m^3
        if self.units == 'cfs':
            df["q"] = round(df["q"] / 35.3146667, 2)
        else:
            raise ValueError(
                f"units='{self.units}' are not compatible with function. Convert input data to ft^3/s first.")

        # make dataframe into xarray.Dataset
        df = pd.DataFrame(df)
        df = df.rename(columns={"siteID": 'location', 'q': "discharge_volume"})
        df = df.reset_index().set_index(["location", "time"]).to_xarray()
        df['time'] = pd.DatetimeIndex(df['time'].values)

        return df

    def create_dataframe(self):
        if self.source == 'stage_file':
            imported_df = self._get_stage_csv()
        elif self.source == 'get_usgs':
            imported_df = self._get_usgs_flows()
        self.imported_dataframe = imported_df

        output = self._format_flow()

        return output

