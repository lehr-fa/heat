import os
import unittest

import heat as ht


class TestKMeans(unittest.TestCase):
    def test_fit_iris(self):
        # get some test data
        iris = ht.load(os.path.join(os.getcwd(), 'heat/datasets/data/iris.csv'), sep=';', split=0)

        # fit the clusters
        k = 3
        kmeans = ht.ml.cluster.KMeans(n_clusters=k)
        kmeans.fit(iris)

        # check whether the results are correct
        self.assertIsInstance(kmeans.cluster_centers_, ht.DNDarray)
        self.assertEqual(kmeans.cluster_centers_.shape, (k, iris.shape[1],))
