import geopandas as gpd
import numpy as np


def get_total_value_of_area(areal_points):
    total = np.sum(areal_points[1][:, 2])
    return total

def set_areal_weights(areal_data: gpd.GeoDataFrame, areal_points: dict):
    """
    Function prepares array for the weighted semivariance calculation.

    INPUT:

    :param areal_data: (GeoDataFrame) with columns:
        ['geometry', 'area.id', 'area.value', 'area.centroid', 'area.centroid.x', 'area.centroid.y],
    :param areal_points: (dict) of points within areas in the form:
        {area_id, [point_position_x, point_position_y, value]}.

    OUTPUT:

    :return: (numpy array) of weighted points.
    """

    areal_ids = areal_data['area.id'].unique()

    # Set GeoDataFrame for calculations
    gdf = areal_data.copy()
    gdf['point.value.total'] = np.nan

    for _id in areal_ids:
        # Calculate total value of points within area
        total = np.sum(areal_points[_id][:, 2])
        idx = gdf[gdf['area.id'] == _id].index
        gdf.at[idx, 'point.value.total'] = total

    weighted_semivariance_input = gdf[['area.centroid.x', 'area.centroid.y', 'area.value', 'point.value.total']].values
    return weighted_semivariance_input
