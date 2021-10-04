import numpy as np
import pandas as pd
from collections import defaultdict

from pyinterpolate.distance.calculate_distances import calc_point_to_point_distance,\
    calc_block_to_block_distance
from pyinterpolate.transform.set_areal_weights import get_total_value_of_area


def prepare_kriging_data(unknown_position, data_array, number_of_neighbours=10):
    """
    Function prepares data for kriging - array of point position, value and distance to an unknown point.

    INPUT:

    :param unknown_position: (numpy array) position of unknown value,
    :param data_array: (numpy array) known positions and their values,
    :param number_of_neighbours: (int) number of the closest locations to the unknown position.

    OUTPUT:
    :return: (numpy array) dataset with position, value and distance to the unknown point:
        [[x, y, value, distance to unknown position], [...]]
    """

    # Distances to unknown point
    r = np.array([unknown_position])

    known_pos = data_array[:, :-1]
    dists = calc_point_to_point_distance(r, known_pos)

    # Prepare data for kriging
    kriging_output_array = np.c_[data_array, dists.T]
    kriging_output_array = kriging_output_array[kriging_output_array[:, -1].argsort()]
    prepared_data = kriging_output_array[:number_of_neighbours]

    return prepared_data


def prepare_poisson_kriging_data(unknown_area, points_within_unknown_area,
                                 known_areas, points_within_known_areas,
                                 number_of_neighbours, max_search_radius,
                                 weighted=False):
    """
    Function prepares data for centroid based Poisson Kriging.

    INPUT:

    :param unknown_area: (numpy array) unknown area in the form:
        [area_id, polygon, centroid x, centroid y],
    :param points_within_unknown_area: (numpy array) points and their values within the given area:
        [area_id, [point_position_x, point_position_y, value]],
    :param known_areas: (numpy array) known areas in the form:
        [area_id, polygon, centroid x, centroid y, aggregated value],
    :param points_within_known_areas: (numpy array) points and their values within the given area:
        [area_id, [point_position_x, point_position_y, value]],
    :param number_of_neighbours: (int) minimum number of neighbours to include in the algorithm,
    :param max_search_radius: (float) maximum search radius (if number of neighbours within this search radius is
        smaller than number_of_neighbours parameter then additional neighbours are included up to number of neighbors),
    :param weighted: (bool) distances weighted by population (True) or not (False).

    OUTPUT:

    :return: (numpy array) distances from known locations to the unknown location:
        [id_known, coordinate x, coordinate y, value, distance to unknown, aggregated points values within an area].
    """

    # Prepare data
    cx_cy = unknown_area[2:-1]
    r = np.array(cx_cy)

    known_centroids = known_areas.copy()
    kc_ids = known_centroids[:, 0]
    kc_vals = known_centroids[:, -1]
    kc_pos = known_centroids[:, 2:-1]

    # Build set for Poisson Kriging

    if weighted:
        known_areas_pts = points_within_known_areas.copy()

        dists = []  # [id_known, dist]

        for pt in known_areas_pts:
            d = calc_block_to_block_distance([pt, points_within_unknown_area])
            dists.append([d[0][0][1]])
        s = np.ravel(np.array(dists)).T
        kriging_data = np.c_[kc_ids, kc_pos, kc_vals, s]  # [id, coo_x, coo_y, val, dist_to_unkn]
    else:
        dists = calc_point_to_point_distance(kc_pos, [r])
        dists = dists.ravel()
        s = dists.T
        kriging_data = np.c_[kc_ids, kc_pos, kc_vals, s]  # [id, coo_x, coo_y, val, dist_to_unkn]

    # sort by distance
    kriging_data = kriging_data[kriging_data[:, -1].argsort()]

    # Get distances in max search radius
    max_search_pos = np.argmax(kriging_data[:, -1] > max_search_radius)
    output_data = kriging_data[:max_search_pos]

    # check number of observations

    if len(output_data) < number_of_neighbours:
        output_data = kriging_data[:number_of_neighbours]

    # get total points' value in each id from prepared datasets and append it to the array

    points_vals = []
    for rec in output_data:
        areal_id = rec[0]
        points_in_area = points_within_known_areas[points_within_known_areas[:, 0] == areal_id]
        total_val = get_total_value_of_area(points_in_area[0])
        points_vals.append(total_val)

    output_data = np.c_[output_data, np.array(points_vals)]
    return output_data


def _merge_vals_and_distances(known_vals, unknown_vals, distances_array):
    """
    Function prepares array of point values and respective distances for Poisson Kriging distance
    :param known_vals: (numpy array) list of known area point values - number of rows of output array,
    :param unknown_vals: (numpy array) list of unknown area point values - number of columns of output array,
    :param distances_array: (numpy array) distances array with the same number of rows as known_vals and
        the same number of columns as unknown_vals arrays,
    :return output_arr: (numpy array) array of [known point value, unknown point value, distance between points]
    """
    output = []
    for k_idx, value in enumerate(known_vals):
        for u_idx, u_value in enumerate(unknown_vals):
            output.append([value, u_value, distances_array[k_idx, u_idx]])
    output_arr = np.array(output)
    return output_arr


def prepare_ata_data(points_within_unknown_area,
                     known_areas, points_within_known_areas,
                     number_of_neighbours, max_search_radius):
    """
    Function prepares data for Area to Area Poisson Kriging.

    INPUT:

    :param points_within_unknown_area: (numpy array) points and their values within the unknown area:
        [area_id, [point_position_x, point_position_y, value of point]],
    :param known_areas: (numpy array) known areas in the form:
        [area_id, areal_polygon, centroid coordinate x, centroid coordinate y, value at specific location],
    :param points_within_known_areas: (numpy array) points and their values within the given area:
        [[area_id, [point_position_x, point_position_y, value of point]], ...],
    :param number_of_neighbours: (int) minimum number of neighbours to include in the algorithm,
    :param max_search_radius: (float) maximum search radius (if number of neighbours within this search radius is
        smaller than number_of_neighbours parameter then additional neighbours are included up to number of neighbors).

    OUTPUT:

    :return output_data: (dict) distances from known locations to the unknown location:
        {area id:
            {area.value: float,
            total.value: float,
            array: numpy array of points [known point value, unknown point value, distance]
            }
        }
    """

    # Initialize set

    kriging_data = known_areas[['area.id', 'area.value']].copy()
    kriging_data.set_index('area.id', inplace=True)

    # Build set for Area to Area Poisson Kriging - sort areas with distance

    dists = {}  # [id_known, dist to unknown]
    unknown_key = list(points_within_unknown_area.keys())[0]

    for area_id in points_within_known_areas:
        # Pass Dict with the known areas points and unknown areas points
        datadict = points_within_unknown_area.copy()
        datadict[area_id] = points_within_known_areas[area_id]
        d = calc_block_to_block_distance(datadict)
        dists[area_id] = d.loc[area_id, unknown_key]

    dists = pd.DataFrame.from_dict(dists, orient='index', columns=['distance'])
    kriging_data = kriging_data.join(dists)
    # s = np.ravel(np.array(dists)).T
    # kriging_data = np.c_[kriging_areas_ids, kriging_areal_values, s]  # [id, areal val, dist_to_unkn]

    # sort by distance
    kriging_data = kriging_data.sort_values('distance')

    # Get distances within max search radius
    output_data = kriging_data[kriging_data['distance'] < max_search_radius]

    # check number of observations

    if len(output_data) < number_of_neighbours:
        output_data = kriging_data.iloc[:number_of_neighbours]

    # for each of prepared id prepare distances list with points' weights for semivariogram calculation

    points_in_unknown_area = points_within_unknown_area[unknown_key][:, :-1]
    vals_in_unknown_area = points_within_unknown_area[unknown_key][:, -1]

    output_d = {}

    # Now get distances between all points area-to-area
    for area_id in output_data.index:
        known_points = points_within_known_areas[area_id][:, :-1]
        known_values = points_within_known_areas[area_id][:, -1]
        distances = calc_point_to_point_distance(points_a=known_points,
                                                 points_b=points_in_unknown_area)
        # Prepare Output
        merged = _merge_vals_and_distances(known_values, vals_in_unknown_area, distances)  # [known point value,
                                                                                           #  unknown point value,
                                                                                           #  distance between points]
        total_val = np.sum(known_values)
        output_d[area_id] = {
            'area.value': output_data['area.value'].loc[area_id],
            'total.value': total_val,
            'array': merged
        }

        """
        output_d = {
            area id: {
                area.value: float,
                total.value: float,
                array: numpy array of points [known point value, unknown point value, distance]
                }
            }
        """

    return output_d


def prepare_ata_known_areas(list_of_points_of_known_areas):
    """
    Function prepares known areas data for prediction.

    INPUT:

    :param list_of_points_of_known_areas: (dict) {
            area id: {
                area.value: float,
                total.value: float,
                array: numpy array of points [known point value, unknown point value, distance]
                }
            }

    OUTPUT:

    :return: (dict) list of arrays with areas and distances between them:
        {id base: {id other: [base point value, other point value,  distance between points]}}
    """
    keys = list_of_points_of_known_areas.keys()
    all_distances_dict = dict()

    for k1 in keys:
        all_distances_dict[k1] = defaultdict()
        points_in_base_area = list_of_points_of_known_areas[k1]['array'][:, :-1]
        vals_in_base_area = list_of_points_of_known_areas[k1]['array'][:, -1]

        for k2 in keys:
            points_in_other_area = list_of_points_of_known_areas[k2]['array'][:, :-1]
            vals_in_other_area = list_of_points_of_known_areas[k2]['array'][:, -1]

            distances_array = calc_point_to_point_distance(points_in_base_area, points_in_other_area)
            merged = _merge_vals_and_distances(vals_in_base_area, vals_in_other_area, distances_array)

            all_distances_dict[k1][k2] = merged

            list_of_distances_from_base[1].append([id_other, merged])
        all_distances_list.append(list_of_distances_from_base)

    return np.array(all_distances_list)


def prepare_distances_list_unknown_area(unknown_area_points):
    """
    Function prepares distances list of unknown (single) area.

    INPUT:

    :param unknown_area_points: [pt x, pt y, val].

    OUTPUT:

    :return: [point value 1, point value 2,  distance between points].
    """
    dists = calc_point_to_point_distance(unknown_area_points[:, :-1])
    vals = unknown_area_points[:, -1]

    merged = _merge_vals_and_distances(vals, vals, dists)
    return np.array(merged)


def _merge_point_val_and_distances(unknown_point_val, known_vals, distances_array):
    """
    Function prepares array of point values and respective distances for Poisson Kriging distance.

    INPUT:

    :param unknown_point_val: (float) unknown point value,
    :param known_vals: (numpy array) list of unknown area point values,
    :param distances_array: (numpy array) distances from unknown area point to known area points.

    OUTPUT:

    :return output_arr: (numpy array) array of [unknown point value, [known points values, distance between points]].
    """
    distances_array = distances_array[:, 0]
    otp = np.array([list(x) for x in zip(known_vals, distances_array)])

    output_arr = np.array([unknown_point_val, otp])
    return output_arr

def prepare_atp_data(points_within_unknown_area,
                     known_areas, points_within_known_areas,
                     number_of_neighbours, max_search_radius):
    """
    Function prepares data for Area to Point Poisson Kriging.

    INPUT:

    :param points_within_unknown_area: (numpy array) points and their values within the given area:
        [area_id, [point_position_x, point_position_y, value of point]],
    :param known_areas: (numpy array) known areas in the form:
        [area_id, areal_polygon, centroid coordinate x, centroid coordinate y, value at specific location],
    :param points_within_known_areas: (numpy array) points and their values within the given area:
        [[area_id, [point_position_x, point_position_y, value of point]], ...],
    :param number_of_neighbours: (int) minimum number of neighbours to include in the algorithm,
    :param max_search_radius: (float) maximum search radius (if number of neighbours within this search radius is
        smaller than number_of_neighbours parameter then additional neighbours are included up to number of neighbors).

    OUTPUT:

    :return output_data: (numpy array) distances from known locations to the unknown location:
        [
            id_known,
            areal value - count,
            [
                unknown point value,
                [known point values, distance],
            total point value count
            ],
            [array of unknown area points coordinates]
        ]
    """

    # Initialize set

    kriging_areas_ids = known_areas[:, 0]
    kriging_areal_values = known_areas[:, -1]

    # Build set for Area to Area Poisson Kriging - sort areas with distance

    known_areas_pts = points_within_known_areas.copy()

    dists = []  # [id_known, dist to unknown]

    for pt in known_areas_pts:
        d = calc_block_to_block_distance([pt, points_within_unknown_area])
        dists.append([d[0][0][1]])
    s = np.ravel(np.array(dists)).T
    kriging_data = np.c_[kriging_areas_ids, kriging_areal_values, s]  # [id, areal val, dist_to_unkn]

    # sort by distance
    kriging_data = kriging_data[kriging_data[:, -1].argsort()]

    # Get distances in max search radius
    max_search_pos = np.argmax(kriging_data[:, -1] > max_search_radius)
    output_data = kriging_data[:max_search_pos]

    # check number of observations

    if len(output_data) < number_of_neighbours:
        output_data = kriging_data[:number_of_neighbours]

    # for each of prepared id prepare distances list with points' weights for semivariogram calculation

    points_vals = []
    points_and_vals_in_unknown_area = points_within_unknown_area[1]
    for rec in output_data:
        areal_id = rec[0]
        areal_value = rec[1]
        known_area = points_within_known_areas[points_within_known_areas[:, 0] == areal_id]
        known_area = known_area[0]
        points_in_known_area = known_area[1][:, :-1]
        vals_in_known_area = known_area[1][:, -1]

        # Set distances array from each point of unknown area
        merged_points_array = []
        for u_point in points_and_vals_in_unknown_area:
            u_point_dists = calc_point_to_point_distance(points_in_known_area, [u_point[:-1]])
            u_point_val = u_point[-1]
            merged = _merge_point_val_and_distances(u_point_val, vals_in_known_area, u_point_dists)
            merged_points_array.append(merged)

        total_val = np.sum(known_area[1][:, 2])
        generated_array = [areal_id, areal_value, merged_points_array, total_val]  # [[id, value, [
                                                                                   # [unknown point value,
                                                                                   #     [known points values,
                                                                                   #      distances between points]],
                                                                                   # ...],
                                                                                   #  total known points value],
                                                                                   # [list of uknown point coords]]
        points_vals.append(generated_array)

    output_data = np.array(points_vals)
    return [output_data, points_within_unknown_area[1][:, :-1]]
