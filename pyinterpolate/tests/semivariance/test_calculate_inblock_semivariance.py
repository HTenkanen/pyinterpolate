from pyinterpolate.semivariance.areal_semivariance.within_block_semivariance.calculate_inblock_semivariance\
    import calculate_inblock_semivariance

import numpy as np
import geopandas as gpd
from pyinterpolate.data_processing.data_preparation.prepare_areal_shapefile import prepare_areal_shapefile
from pyinterpolate.data_processing.data_preparation.get_points_within_area import get_points_within_area
from pyinterpolate.semivariance.semivariogram_estimation.calculate_semivariance import calculate_semivariance
from pyinterpolate.semivariance.semivariogram_fit.fit_semivariance import TheoreticalSemivariogram


def test_calculate_inblock_semivariance():
    # Data prepration
    areal_dataset = 'sample_data/test_areas_pyinterpolate.shp'
    subset = 'sample_data/test_points_pyinterpolate.shp'

    a_id = 'id'
    areal_val = 'value'
    points_val = 'value'

    # Get maximum range and set step size

    gdf = gpd.read_file(areal_dataset)

    total_bounds = gdf.geometry.total_bounds
    total_bounds_x = np.abs(total_bounds[2] - total_bounds[0])
    total_bounds_y = np.abs(total_bounds[3] - total_bounds[1])

    max_range = min(total_bounds_x, total_bounds_y)
    step_size = max_range / 4

    lags = np.arange(0, max_range, step_size * 2)

    areal_data_prepared = prepare_areal_shapefile(areal_dataset, a_id, areal_val)
    points_in_area = get_points_within_area(areal_dataset, subset, areal_id_col_name=a_id,
                                            points_val_col_name=points_val)

    # Get areal centroids with data values
    areal_centroids = areal_data_prepared[:, 2:]
    areal_centroids = np.array([[x[0], x[1], x[2]] for x in areal_centroids])

    gamma = calculate_semivariance(areal_centroids, lags, step_size)

    # Get theoretical semivariogram model
    ts = TheoreticalSemivariogram(areal_centroids, gamma)

    ts.find_optimal_model(number_of_ranges=8)

    # Get centroids to calculate experimental semivariance

    inblock_semivariance = calculate_inblock_semivariance(points_in_area, ts)
    inblock_semivariance = np.array(inblock_semivariance)

    data_point = inblock_semivariance[inblock_semivariance[:, 0] == 1][0]
    assert (int(data_point[1]) == 111)


if __name__ == '__main__':
    test_calculate_inblock_semivariance()
