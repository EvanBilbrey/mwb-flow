import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point

# format = minx, miny, maxx, maxy
from config import GRIDMET_BOUNDS
# GRIDMET Grid properties
from config import GRIDMET_NROWS
from config import GRIDMET_NCOLS
from config import GRIDMET_XRES
from config import GRIDMET_YRES


def get_gridmet_cells(geom):
    """
    Gets the GridMET cells for each input coordinate point.
    :param geom: geopandas.GeoDataFrame that includes point or polygon geometries column and
        a column of location ID's for each point
    :return: geopandas.GeoDataFrame - product of spatial join between GridMET cells and points containing
    GridMET cell ID's, input point ID's, and input geometry cell center point geometry column
    """
    nshp_cols =np.linspace(GRIDMET_BOUNDS[0], GRIDMET_BOUNDS[2], GRIDMET_NCOLS+1)
    nshp_rows = np.linspace(GRIDMET_BOUNDS[1], GRIDMET_BOUNDS[3], GRIDMET_NROWS+1)
    nshp_rows = np.flip(nshp_rows)

    gridmet_cells = []
    for y in nshp_rows[:-1]:
        for x in nshp_cols[:-1]:
            gridmet_cells.append(Polygon([(x, y),
                                          (x+GRIDMET_XRES, y),
                                          (x+GRIDMET_XRES, y+GRIDMET_YRES),
                                          (x, y+GRIDMET_YRES)]))

    gmet_polys = gpd.GeoDataFrame({
        'cell_id': np.arange(len(gridmet_cells)),
        'geometry': gridmet_cells
    }, crs='EPSG:4326')

    rslt = gmet_polys.sjoin(geom, how="inner")

    return rslt

# tpnts = gpd.GeoDataFrame({
#     'ids': [1010, 1011, 1012, 1013, 1014, 1015],
#     'geometry': [Point(-106.862859, 44.829906),
#                  Point(-106.969769, 44.910018),
#                  Point(-106.802219, 45.082024),
#                  Point(-107.497037, 44.781413),
#                  Point(-107.207667, 44.540927),
#                  Point(-107.204430, 44.532385)]
# }, crs=4326)
#
# t = get_gridmet_cells(tpnts)
