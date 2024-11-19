import geopandas as gp
import pandas as pd
import numpy as np
import rioxarray as rio
import xarray
from shapely.geometry import Polygon
from pathlib import Path
import os

# def clip_grid(in_geom, in_grid, clip_to_shape=False):
#     vgm = gp.read_file(in_geom)
#     grd = rio.open_rasterio(in_grid)
#     if type(grd) is list:
#         grd = grd[1]
#     geom_crs = vgm.crs
#     if clip_to_shape:
#         clipped = grd.rio.clip(vgm.geometry, geom_crs)
#     else:
#         geometries = [
#             {
#                 'type': 'Polygon',
#                 'coordinates': [[
#                     [vgm.bounds.minx[0], vgm.bounds.maxy[0]],
#                     [vgm.bounds.minx[0], vgm.bounds.miny[0]],
#                     [vgm.bounds.maxx[0], vgm.bounds.miny[0]],
#                     [vgm.bounds.maxx[0], vgm.bounds.maxy[0]],
#                     [vgm.bounds.minx[0], vgm.bounds.maxy[0]]
#                 ]]
#             }
#         ]
#         clipped = grd.rio.clip(geometries, geom_crs)
#
#     return clipped


def grid_area_weighted_volume(dataset, in_geom, out_fp, save_shp_to_file=False):
    """
    Takes raster or multidimensional (.nc) datasets and input geometry in the same coordinate reference
    system and returns an area weighted volume from depth valued datasets (e.g., precip).
    :param dataset: an xarray dataset or dataarray instance, must have defined spatial_ref and be in units of meters,
    time dimension/coords must be labeled 'time'
    :param in_geom: a GeoDataFrame of input shapefile/geometry in the same spatial_ref as xarray data
    :param out_fp: the filepath to output .csv data and shapefile if specified
    :param save_shp_to_file: boolean, default = False, if True will save the grid shapefile to the out filepath
    :return: creates .csv files of volume for the input vector geometries based on the gridded data
    """
    # output path
    out_file_path = Path(out_fp)
    # get grid resolution
    xres, yres = dataset.rio.resolution()

    # get the array bounds as variables for new geometry
    gminx, gminy, gmaxx, gmaxy = dataset.rio.bounds()

    # shape of dataset
    nrows = dataset.shape[1]
    ncols = dataset.shape[2]

    nshp_cols = list(np.linspace(gminx, gmaxx, ncols+1))
    nshp_rows = np.linspace(gminy, gmaxy, nrows+1)
    nshp_rows = np.flip(nshp_rows)

    polygons = []
    for y in nshp_rows[:-1]:
        for x in nshp_cols[:-1]:
            polygons.append(Polygon([(x,y), (x+xres,y), (x+xres, y+yres), (x, y+yres)]))

    intersection_areas = []
    main_shape_id = []
    intersecting_shape_id = []
    for i, g1 in enumerate(in_geom.geometry.values):
        for j, g2 in enumerate(polygons):
            if g1.intersects(g2):
                intersection_areas.append(g1.intersection(g2).area)
                main_shape_id.append(i)
                intersecting_shape_id.append(j)

    s_areas = pd.DataFrame({'Cell_Area': intersection_areas}, index=intersecting_shape_id)
    polys = pd.DataFrame({'geometry': polygons})

    # Create shapefile
    grd_shp = gp.GeoDataFrame(pd.concat([s_areas, polys], axis=1))
    grd_shp.set_crs(in_geom.crs.to_epsg(), inplace=True)
    grd_shp.sort_index(inplace=True)
    grd_shp['GID'] = grd_shp.index

    if save_shp_to_file:
        grd_shp.to_file(out_file_path / (dataset.name + '_grid.shp'))

    area_arry = grd_shp['Cell_Area'].values.reshape(nrows, ncols)
    md_arry = dataset.values
    vol_grd = area_arry * md_arry
    vol_v = np.nansum(vol_grd, axis=(1,2))

    if str(dataset.indexes['time'].dtype) != 'datetime64[ns]':
        dttmidx = dataset.indexes['time'].to_datetimeindex()
    else:
        dttmidx = pd.DatetimeIndex(dataset.indexes['time'])

    VOL_SRS = pd.DataFrame({dataset.name + ' volume (cubic meters)': vol_v}, index=dttmidx)
    VOL_SRS.to_csv(out_file_path / (dataset.name + ' Weighted Area Volume_WY{0}.csv'.format(VOL_SRS.index[-1].year)))


# def nsidc_to_polygon(file, save_to_file=False)




if __name__ == '__main__':

    # filepath where output is saved
    outdir = Path('D:\\Modeling\\Water Supply Forecasting\\CSKT Post Cr abv McDonald Lake\\Meteor_Data')
    if not outdir.exists():
        os.makedirs(outdir)

    nsidc_p = Path('D:\\Spatial_Data\\Statewide_Data\\Raster\\NSIDC\\NSIDC-0719')
    grdmt_p = Path('D:\\Spatial_Data\\Statewide_Data\\Raster\\GridMET\\Precip')
    file = Path('D:\\Spatial_Data\\Statewide_Data\\Raster\\GridMET\\Precip\\pr_1994.nc')
    shp_fl = Path('D:\\ArcGIS_Projects\\Flathead Basin\\Post Creek\\Vector\\Post_Cr_abv_McDonald_DrainageArea.shp')

    in_geom = gp.read_file(shp_fl)
    in_geom_rpj = in_geom.to_crs("EPSG:5070")
    in_geom_wgs84 = in_geom.to_crs("EPSG:4326")
    in_geom_nad83 = in_geom.to_crs("EPSG:4269")

    for file in nsidc_p.glob('*.nc'):
        nas_snw = xarray.open_dataset(file)
        nas_swe = nas_snw.SWE
        nas_swe.rio.write_crs(nas_snw.crs.spatial_ref, inplace=True)
        nas_swe_rn = nas_swe.rename({'lon': 'x', 'lat':'y'})
        nas_swe_clip = nas_swe_rn.rio.clip_box(
                minx=in_geom_nad83.bounds.minx[0],
                miny=in_geom_nad83.bounds.miny[0],
                maxx=in_geom_nad83.bounds.maxx[0],
                maxy=in_geom_nad83.bounds.maxy[0]
        )
        nas_swe_rpj = nas_swe_clip.rio.reproject("EPSG:5070")
        nas_swe_rpj_m = nas_swe_rpj / 1000.0

        grid_area_weighted_volume(nas_swe_rpj_m, in_geom_rpj, outdir)

    for file in grdmt_p.glob('*.nc'):
        gmet = rio.open_rasterio(file, mask_and_scale=True)
        gmet_rn = gmet.rename({'day': 'time'})
        gmet_clip = gmet_rn.rio.clip_box(
            minx=in_geom_wgs84.bounds.minx[0],
            miny=in_geom_wgs84.bounds.miny[0],
            maxx=in_geom_wgs84.bounds.maxx[0],
            maxy=in_geom_wgs84.bounds.maxy[0]
        )
        gmet_rpj = gmet_clip.rio.reproject("EPSG:5070")
        gmet_rpj_m = gmet_rpj / 1000.0

        grid_area_weighted_volume(gmet_rpj_m, in_geom_rpj, outdir)

sntl = pd.read_csv(outdir.parents[0] / 'Mission_Mtns_SNOTEL_Sites_POR_20230303.txt', parse_dates=True, comment='#', index_col=0)
M = sntl.loc[(sntl.index.month == 3) & (sntl.index.day ==1)]
A = sntl.loc[(sntl.index.month == 4) & (sntl.index.day ==1)]
My = sntl.loc[(sntl.index.month == 5) & (sntl.index.day ==1)]

M.to_csv(outdir.parents[0] / 'MissionMtns_MARCH1_SNOTEL.csv')
A.to_csv(outdir.parents[0] / 'MissionMtns_APRIL1_SNOTEL.csv')
My.to_csv(outdir.parents[0] / 'MissionMtns_MAY1_SNOTEL.csv')

d = sntl.index
m = sntl.index.month.values >= 10
y = sntl.index.year.values
y[m] = y[m] + 1
mm = (sntl.index.month.values >= 3) & (sntl.index.month.values < 10)
y[mm] = y[mm] * 10
Indx = pd.Index(y)

SDF = sntl.groupby(Indx).mean()
SDF.to_csv(outdir.parents[0] / 'MissionMtn_SNOTEL_octtoMARCH_Summary.csv')

### SNODAS Test
