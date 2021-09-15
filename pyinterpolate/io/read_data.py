import geopandas as gpd
import numpy as np

from geopandas import points_from_xy


def read_txt(
        path: str, delim=',', epsg='4326', crs=None
) -> gpd.GeoDataFrame:
    """
    Function reads data from a text file. Provided data format should include: latitude, longitude, value. You should
        provide crs or epsg, if it's not provided then epsg:4326 is used as a default value (https://epsg.io/4326).
        Data read by a function is converted into GeoSeries.

    INPUT:

    :param path: (str) path to the file,
    :param delim: (str) delimiter which separates columns,
    :param epsg: (str) optional; if not provided and crs is None then algorithm sets epsg:4326 as a default value,
    :param crs: (str) optional;

    OUTPUT:

    :returns: (GeoDataFrame)"""

    data_arr = np.loadtxt(path, delimiter=delim)

    gdf = gpd.GeoDataFrame(data=data_arr, columns=['y', 'x', 'val'])
    gdf['geometry'] = points_from_xy(gdf['x'], gdf['y'])
    gdf.set_geometry('geometry', inplace=True)

    if crs is None:
        gdf.set_crs(epsg=epsg, inplace=True)
    else:
        gdf.set_crs(crs=crs, inplace=True)

    return gdf[['geometry', 'val']]
