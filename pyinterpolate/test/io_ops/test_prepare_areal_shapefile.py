import unittest
import os
import numpy as np
from pyinterpolate.io import prepare_areal_shapefile


class TestPrepareArealShapefile(unittest.TestCase):

    def test_prepare_areal_shapefile(self):

        my_dir = os.path.dirname(__file__)
        path_to_areal_file = os.path.join(my_dir, '../sample_data/test_areas_pyinterpolate.shp')

        # Read without id column and without value column
        try:
            _ = prepare_areal_shapefile(path_to_areal_file)
        except TypeError:
            assert True
        else:
            assert False

        # Read with id column
        dataset_with_id = prepare_areal_shapefile(path_to_areal_file, id_column_name='id', dropnans=False)

        # Tests:
        # Must have 6 columns: geometry, area.id, area.value, area.centroid, area.centroid.x, area.centroid.y
        test_cols_dataset = len(dataset_with_id.columns) == 6
        self.assertTrue(test_cols_dataset, "Dataset should have 6 columns:"
                                           "[geometry, area.id, area.value, area.centroid, area.centroid.x, area.centroid.y]")


if __name__ == '__main__':
    unittest.main()
