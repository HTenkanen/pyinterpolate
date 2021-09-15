import unittest
import os
import geopandas as gpd
import numpy as np
from pyinterpolate.io.read_data import read_txt


class TestReadData(unittest.TestCase):

	def test_read_data(self):
		my_dir = os.path.dirname(__file__)
		path_to_the_data = os.path.join(my_dir, '../sample_data/poland_dem_gorzow_wielkopolski')
		data = read_txt(path_to_the_data)

		# Validate data
		# Check if data type is GeoDataFrame
		check_ndarray = isinstance(data, gpd.GeoDataFrame)
		self.assertTrue(check_ndarray, "Instance of a data type should be geopandas GeoDataFrame")

		# Check if geometry is valid
		is_valid_geometry = data.

		# Check dimensions
		self.assertEqual(data.shape[1], 3, "Shape of data should be 3 - x, y, value")


if __name__ == '__main__':
	unittest.main()
