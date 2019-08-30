import torch

__all__ = [
    'DBSCAN'
]


class SpatialIndex:
    def __init__(self, x, eps):
        self.eps = eps
        device = x.device.torch_device()

        # compute initial order
        offset, _, _ = x.comm.chunk(x.shape, x.split)
        self.initial_order = torch.arange(offset, offset + x.lshape[0], device=device)

        # compute space dimensions
        self.minimums = x.min()._DNDarray__array
        self.maximums = x.max()._DNDarray__array

        # compute cell dimensions
        self.cells_in_dimension = ((self.maximums - self.minimums) / self.eps).type(torch.int64) + 1
        self.total_cells = torch.prod(self.cells_in_dimension)

        # swap dimensions
        self.swapped_dimensions = torch.arange(x.shape[1])
        self.swapped_dimensions, _ = self.swapped_dimensions.sort(descending=True)
        self.halo_size = self.total_cells / self.cells_in_dimension[self.swapped_dimensions[-1]]

        # compute cells
        space_indices = ((x._DNDarray__array - self.minimums) / self.epsilon).floor().type(torch.int64)
        accumulator = self.cells_in_dimension[self.swapped_dimensions].cumprod().roll(1)
        accumulator[0] = 1
        self.cells = (space_indices * accumulator).sum(dim=1)
        self.hist_bucket, self.hist_counts = self.cells.unique(return_counts=True)

        # sort by cell
        self.cells, indices = self.cells.sort()
        self.initial_order[indices]
        x._DNDarray__array = x._DNDarray__array[indices]

        # redistribute and re-index the data if there is more than one rank
        if x.comm.size > 1:
            # global histogram
            pass


class DBSCAN:
    """
    Perform DBSCAN clustering on two-dimensional matrices.

    DBSCAN - Density-Based Spatial Clustering of Applications with Noise. Finds core samples of high density and expands
    clusters from them. Is able to equally identify bordering points (i.e. at the edges of cluster) and noise points not
    belonging to any cluster. Is able to identify clusters of arbitrary shapes (e.g. rings, "bananas", s-shapes etc.).
    DBSCAN excells at finding clusters with similar cluster point density, for clusters with unequal density an
    iterative clustering with varying density thresholds is recommended.

    Parameters
    ----------
    eps : float, optional
        The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is
        not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN parameter to
        choose appropriately for your data set. A too large search radius may significantly reduce parallel processing
        performance. Defaults to 0.5.
    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This
        includes the point itself. Defaults to 5.
    p : float, optional
        The power of the Minkowski metric (p=2 is equivalent to the Euclidean distance) to be used to calculate
        distances between points. Defaults to 2.

    Attributes
    ----------
    core_points_ : ht.DNDarray, shape=[n_points]
        Boolean vector indicating whether a point p is a core label of cluster or not.
    labels_ : ht.DNDarray, shape=[n_points]
        Cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1.

    Examples
    --------
    >>> import heat as ht
    >>> from ht.ml.cluster import DBSCAN
    >>> x = ht.array([[1, 2], [2, 2], [2, 3],
    ...               [8, 7], [8, 8], [25, 80]])
    >>> clustering = DBSCAN(eps=3, min_samples=2).fit(x)
    >>> clustering.labels_
    array([ 0,  0,  0,  1,  1, -1])

    References
    ----------
    [1] Ester, M., H. P. Kriegel, J. Sander, and X. Xu, "A Density-Based Algorithm for Discovering Clusters in Large
    Spatial Databases with Noise", Proceedings of the 2nd International Conference on Knowledge Discovery and Data
    Mining, Portland, OR, AAAI Press, pp. 226-231, 1996.
    """
    def __init__(self, eps=0.5, min_samples=5, p=2):
        self.eps = eps
        self.min_samples = min_samples
        self.p = p

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and contained sub objects that are estimators.

        Returns
        -------
        params : dict
            Mapping of string to any parameter names mapped to their values.
        """
        return {
            'eps': self.eps,
            'min_samples': self.min_samples,
            'p': self.p
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        params : dict
            Mapping of parameter string to parameter value.

        Returns
        -------
        self : ht.ml.cluster.DBSCAN
            This estimator for chaining.
        """
        self.eps = params.get('eps', self.eps)
        self.min_samples = params.get('min_samples', self.min_samples)
        self.p = params.get('p', self.p)

        return self

    def fit(self, x):
        pass

    def fit_predict(self, x):
        pass
