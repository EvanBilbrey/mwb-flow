from chmdata.thredds import GridMet
from chmdata.thredds import BBox
from datetime import datetime, timedelta
import py3dep
from tqdm import tqdm
import pandas as pd
import xarray as xr
from shapely.geometry import Point
from prep.utils import get_gridmet_cells
from config import GRIDMET_PARAMS
import GRIDtools as gt

from geocube.api.core import make_geocube

import rioxarray
import xvec
import shapely


# TODO - Default date did go to previous day...seems inconsistent with THREDDS,
#   sometimes it creates a mismatch in the dates and variable series, NEED TO SEE
#   if this is fixable bug in chmdata.thredds or if the end date needs an exception
#   in this script.
DEFAULT_DATES = ('1979-01-01', (datetime.today() - timedelta(days=2)).strftime("%Y-%m-%d"))


def get_gridmet_at_points(in_geom,
                          gdf_index_col=None,
                          start = DEFAULT_DATES[0],
                          end = DEFAULT_DATES[1],
                          crs = 4326) -> xr.Dataset:
    """
    Function takes a list of GridMET data parameters, start date, end date, and a Geopandas GeoDataFrame of point or
    polygon geometries and returns a discrete station formatted xarray dataset of necessary GridMET data to run mwb_flow
    for each point geometry or averaged over each polygon geometry.
    :param in_geom: geopandas.GeoDataFrame - contains geometry
    :param gdf_index_col: str - name of column in GeoDataFrame to use as a unique identifier for each geometry
        default is None, in which case the index will be used
    :param start: str "%Y-%m-%d" - Starting date of data extraction period
    :param end: str "%Y-%m-%d" = Ending date of data extraction period
    :param crs: int or str - EPSG code for crs, default is 4326
    :return: an xarray dataset for discrete locations (stations)
    """
    if gdf_index_col is not None:
        ixcol = gdf_index_col
    else:
        in_geom['ixcol'] = in_geom.index
        ixcol = 'ixcol'

    location_ids = in_geom[ixcol].to_list()

    if (in_geom.geometry.geom_type == 'Point').all():
        coords = list(zip(in_geom.geometry.x, in_geom.geometry.y))
    elif (in_geom.geometry.geom_type == 'Polygon').all():
        coords = list(zip(in_geom.geometry.centroid.x, in_geom.geometry.centroid.y))
    else:
        coords = None
        raise ValueError("Mixed geometry types were found in the input GeoDataFrame. Mixed Geometry is not supported.")

    loc_lat = []
    loc_lon = []
    loc_elev = py3dep.elevation_bycoords(coords, crs=crs)  # only 4326 or NAD83 works with py3dep

    if isinstance(loc_elev, list):
        loc_elev = loc_elev
    else:
        loc_elev = [loc_elev]

    loc_gdf = in_geom[['{0}'.format(ixcol), 'geometry']]

    print("Retrieving GridMET cells...")
    gmt_cells = get_gridmet_cells(loc_gdf)
    unq_cells = gmt_cells['cell_id'].unique()
    print("{0} unique GridMET cells found for {1} input features.".format(len(unq_cells), len(loc_gdf[ixcol])))

    gmt_cntrs = gmt_cells.drop_duplicates(subset='cell_id').centroid

    pr = []
    tmmn = []
    tmmx = []

    cdsets = {}
    print("Fetching GridMET data for unique cells...")
    for cell in tqdm(unq_cells, desc='Cells'):
        clon = gmt_cntrs[cell].x
        clat = gmt_cntrs[cell].y
        datasets = []
        for p in GRIDMET_PARAMS:
            s = start
            e = end
            ds = GridMet(p, start=s, end=e, lat=clat, lon=clon).get_point_timeseries()
            datasets.append(ds)
        cdsets[cell] = datasets

    for i in range(len(coords)):  ## left off here, need to then allocate unique cells back to geoms, average if polygon
        c = coords[i]
        loc = location_ids[i]
        gmtcell_ids = gmt_cells[gmt_cells[ixcol] == loc]
        lon, lat = c
        loc_lat.append(lat)
        loc_lon.append(lon)


        if len(gmtcell_ids.index) > 1:

            prm = []
            tmmnm = []
            tmmxm = []

            for cid in gmtcell_ids['cell_id']:
                dset = cdsets[cid]

                prm.append(dset[GRIDMET_PARAMS.index('pr')])
                tmmnm.append(dset[GRIDMET_PARAMS.index('tmmn')])
                tmmxm.append(dset[GRIDMET_PARAMS.index('tmmx')])


            prm_d = pd.concat(prm)
            tmmnm_d = pd.concat(tmmnm)
            tmmxm_d = pd.concat(tmmxm)

            # TODO - use appropriate spatial summary statistics in the future, not just average over the input polygon
            #   but area weighted volume/cumulative total for precip/solar radiation (could derive from GRIDtools package)
            pr.append(prm_d.groupby(prm_d.index).mean())
            tmmn.append(tmmnm_d.groupby(tmmnm_d.index).mean())
            tmmx.append(tmmxm_d.groupby(tmmxm_d.index).mean())

        else:
            dset = cdsets[gmtcell_ids['cell_id'].values[0]]
            pr.append(dset[GRIDMET_PARAMS.index('pr')])
            tmmn.append(dset[GRIDMET_PARAMS.index('tmmn')])
            tmmx.append(dset[GRIDMET_PARAMS.index('tmmx')])

    xds = xr.Dataset(
        {
            "precip": (['time', 'location'], pd.concat(pr, axis=1), {'standard_name': 'Precipitation',
                                                                     'units': 'mm'}),
            "min_temp": (['time', 'location'], pd.concat(tmmn, axis=1), {'standard_name': 'Minimum Temperature',
                                                                     'units': 'Kelvin'}),
            "max_temp": (['time', 'location'], pd.concat(tmmx, axis=1), {'standard_name': 'Maximum Temperature',
                                                                     'units': 'Kelvin'})
        },
        coords={
            "lat": (['location'], loc_lat, {'standard_name': 'latitude',
                                            'long_name': 'location_latitude',
                                            'units': 'degrees',
                                            'crs': '4326'}),
            "lon": (['location'], loc_lon, {'standard_name': 'longitude',
                                            'long_name': 'location_longitude',
                                            'units': 'degrees',
                                            'crs': '4326'}),
            "elev": (['location'], loc_elev, {'standard_name': 'elevation',
                                            'long_name': 'location_elevation',
                                            'units': 'meters'}),
            "location": (['location'], location_ids, {'long_name': 'location_identifier',
                                            'cf_role': 'timeseries_id'}),
            "time": pr[0].index
        },
        attrs={
            "featureType": 'timeSeries',
        }
    )

    return xds

def get_gridmet_for_polygons_with_xvec(in_geom,
                          gdf_index_col,
                          start = DEFAULT_DATES[0],
                          end = DEFAULT_DATES[1],
                          crs = 4326) -> xr.Dataset:
    """
    :return: Function takes a list of GridMET data parameters, start date, end date, and a Geopandas GeoDataFrame of
    polygon geometries and returns a discrete station formatted xarray dataset of average or area weighted GridMET data to run mwb_flow
    for each polygon geometry.
    :param in_geom: geopandas.GeoDataFrame - contains geometry
    :param gdf_index_col: str - name of column in GeoDataFrame to use as a unique identifier for each geometry
    :param start: str "%Y-%m-%d" - Starting date of data extraction period
    :param end: str "%Y-%m-%d" = Ending date of data extraction period
    :param crs: int or str - EPSG code for crs, default is 4326
    :return: a xarray dataset for discrete locations (stations)
    """

    gmet_list = []
    for p in GRIDMET_PARAMS:
        bnds = in_geom.total_bounds
        gmet = GridMet(variable=p, start=start, end=end,
                       bbox=BBox(bnds[0] - 0.5, bnds[2] + 0.5, bnds[3] + 0.5, bnds[1] - 0.5))
        gmet = gmet.subset_nc(return_array=True)
        gmet_input = gmet[list(gmet.data_vars)[0]]
        if p == 'pr':
            vol_xds = gt.grid_area_weighted_volume(gmet_input, in_geom, gdf_index_col)
            vol_xds = vol_xds.transpose('location', 'time').drop('area')
        else:
            gmet_list.append(gmet_input)

    ds = xr.merge(gmet_list)
    ds = ds.xvec.zonal_stats(geometry=in_geom.geometry, x_coords="lon", y_coords="lat", stats="mean", method="iterate")

    loc_df = pd.DataFrame(in_geom.gageID).set_index(in_geom.geometry)
    loc_df = loc_df.reindex(in_geom.geometry.values)

    lat_df = pd.DataFrame((in_geom.geometry.bounds['miny'] + in_geom.geometry.bounds['maxy']) / 2).set_index(
        in_geom.geometry)
    lat_df = lat_df.reindex(in_geom.geometry.values)

    xds = xr.Dataset(
        {
            "max_temp": (['location', 'time'], ds['daily_maximum_temperature'].values,
                         {'standard_name': 'Maximum Temperature',
                          'units': 'Kelvin'}),
            "min_temp": (['location', 'time'], ds['daily_minimum_temperature'].values,
                         {'standard_name': 'Maximum Temperature',
                          'units': 'Kelvin'})
        },
        coords={
            "lat": (['location'], list(lat_df.iloc[:, 0]), {'standard_name': 'latitude',
                                                            'long_name': 'location_latitude',
                                                            'units': 'degrees',
                                                            'crs': '4326'}),
            "location": (['location'], list(loc_df.iloc[:, 0]), {'long_name': 'location_identifier',
                                                                 'cf_role': 'timeseries_id'}),
            "time": ds['time'].values
        },
        attrs={
            "featureType": 'timeSeries',
        }
    )

    if 'pr' in GRIDMET_PARAMS:
        output = xr.merge([xds, vol_xds])
    else:
        output = xds

    return output

def get_gridmet_for_polygons(in_geom,
                          gdf_index_col,
                          start = DEFAULT_DATES[0],
                          end = DEFAULT_DATES[1],
                          crs = 4326) -> xr.Dataset:
    """
    :return: Function takes a list of GridMET data parameters, start date, end date, and a Geopandas GeoDataFrame of
    polygon geometries and returns a discrete station formatted xarray dataset of average or area weighted GridMET data to run mwb_flow
    for each polygon geometry.
    :param in_geom: geopandas.GeoDataFrame - contains geometry
    :param gdf_index_col: str - name of column in GeoDataFrame to use as a unique identifier for each geometry
    :param start: str "%Y-%m-%d" - Starting date of data extraction period
    :param end: str "%Y-%m-%d" = Ending date of data extraction period
    :param crs: int or str - EPSG code for crs, default is 4326
    :return: a xarray dataset for discrete locations (stations)
    """

    var_list = []
    for p in GRIDMET_PARAMS:
        bnds = in_geom.total_bounds
        gmet = GridMet(variable=p, start=start, end=end,
                       bbox=BBox(bnds[0] - 0.5, bnds[2] + 0.5, bnds[3] + 0.5, bnds[1] - 0.5))
        gmet = gmet.subset_nc(return_array=True)
        gmet_input = gmet[list(gmet.data_vars)[0]]

        if p == 'pr':
            vol_xds = gt.grid_area_weighted_volume(gmet_input, in_geom, gdf_index_col)
            vol_xds = vol_xds.drop('area')
        else:
            # This is done for each output of the above code
            in_geom["gageID"] = in_geom.gageID.astype(int)
            gmet_input = gmet_input.rio.write_crs(input_crs=crs).rio.clip(in_geom.geometry.values, in_geom.crs)
            gmet_input.name = p

            grid_out = make_geocube(vector_data=in_geom, measurements=["gageID"], like=gmet_input).set_coords('gageID')

            for date in range(0, len(gmet_input.time.values)):
                gmet_ts = gmet_input[date, :, :]
                grid_ts = grid_out

                grid_ts[p] = (grid_out.dims, gmet_ts.values, gmet_ts.attrs, gmet_ts.encoding)
                grid_ts = grid_out.drop("spatial_ref").groupby(grid_out.gageID).mean()

                xda = grid_ts[p]
                xda = xda.expand_dims({"time": 1}).assign_coords(time=('time', [gmet_ts.time.values]))
                var_list.append(xda)
        xds = xr.merge(var_list)

    lat_df = pd.DataFrame((in_geom.geometry.bounds['miny'] + in_geom.geometry.bounds['maxy']) / 2).set_index(
        in_geom.gageID)
    lat_df = lat_df.reindex(list(xds.gageID.values.astype(int)))

    xds = xr.Dataset(
        {
            "max_temp": (['time', 'location'], xds["tmmx"].values, {'standard_name': 'Maximum Temperature',
                                                                    'units': 'Kelvin'}),
            "min_temp": (['time', 'location'], xds["tmmn"].values, {'standard_name': 'Maximum Temperature',
                                                                    'units': 'Kelvin'})
        },
        coords={
            "lat": (['location'], list(lat_df.iloc[:, 0]), {'standard_name': 'latitude',
                                                            'long_name': 'location_latitude',
                                                            'units': 'degrees',
                                                            'crs': '4326'}),
            "location": (['location'], xds['gageID'].values.astype(int), {'long_name': 'location_identifier',
                                                                          'cf_role': 'timeseries_id'}),
            # Keep the order of xds
            "time": xds['time'].values
        },
        attrs={
            "featureType": 'timeSeries',
        }
    )

    if 'pr' in GRIDMET_PARAMS:
        output = xr.merge([xds, vol_xds])  # vol_xds reorders to match xds
    else:
        output = xds

    return output
