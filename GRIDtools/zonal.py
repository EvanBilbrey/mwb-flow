## Zonal Statistics for various geometry types

import geopandas as gpd
import rasterio as rio
from rasterstats import zonal_stats
import numpy as np
import pandas as pd


def calc_zonal_stats(in_geom, in_grid, **kwargs):
    # get geometry in same reference system
    geom = gpd.read_file(in_geom)
    with rio.open(in_grid) as src:
        affine = src.transform
        array = src.read(1)
        crs = src.crs
        nodata = src.nodata
    array[array == nodata] = np.nan

    # check crs
    if geom.crs.to_authority() == crs.to_authority():
        pass
    else:
        geom = geom.to_crs(crs)

    zs = zonal_stats(geom.geometry, array, affine=affine, **kwargs)
    fgd = geom.join(pd.DataFrame(zs))

    # return GeoDataFrame with stats added as additional attributes
    return fgd
