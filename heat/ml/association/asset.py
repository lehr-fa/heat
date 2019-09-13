import enum
import heat as ht


class ASSET:
    """
    ASSET is a statistical method [1] for the detection of repeating sequences of synchronous events in parallel spike
    trains. Given a list `sts` of spike trains, the analysis comprises the following steps:

    1) Build the intersection matrix `imat` (optional) and the associated
       probability matrix `pmat` given a parametric bin size
    2) Compute the joint probability matrix jmat, using a suitable filter
    3) Using pmat and jmat, create a masked version of the intersection matrix
    4) Cluster significant elements of imat into diagonal structures
    5) Extract sequences of synchronous events associated to each worm

    Attributes
    ----------

    Examples
    --------

    References
    ----------
    [1] Torre, Canova, Denker, Gerstein, Helias, Gruen (submitted)
    """
    class Norm(enum.Enum):
        NONE = enum.auto()
        MINIMUM = enum.auto()
        EUCLIDEAN = enum.auto()
        SUM = enum.auto()

    def __init__(self, norm=Norm.SUM):
        if norm not in self.Norm:
            raise ValueError('norm must be an instance of ASSET.Norm, but was {}'.format(norm))
        self.norm = norm

    def _compute_intersection_matrix(self, bsts_x, bsts_y):
        # compute the number of spikes in each bin, for both time axes
        spikes_per_bin_x = bsts_x.sum(axis=0)
        spikes_per_bin_y = bsts_y.sum(axis=0)

        # compute the intersection matrix
        intersection_matrix = bsts_x.T @ bsts_y

        # normalize the intersection matrix
        if self.norm is self.Norm.MINIMUM:
            normalization_coefficient = ht.minimum(
                spikes_per_bin_x.expand_dims(1),
                spikes_per_bin_y.resplit().expand_dims(0)
            )
        elif self.norm is self.Norm.EUCLIDEAN:
            normalization_coefficient = ht.sqrt(
                ht.expand_dims(spikes_per_bin_x, 1) @ ht.expand_dims(spikes_per_bin_y, 0)
            )
        elif self.norm is self.Norm.SUM:
            normalization_coefficient = ht.sum(
                spikes_per_bin_x.expand_dims(1) + spikes_per_bin_y.resplit().expand_dims(0),
                axis=1
            ).expand_dims(1)
        else:
            normalization_coefficient = 1.0

        intersection_matrix /= normalization_coefficient

        return intersection_matrix

    def fit(self, bsts_x, bsts_y):
        intersection_matrix = self._compute_intersection_matrix(bsts_x, bsts_y)
