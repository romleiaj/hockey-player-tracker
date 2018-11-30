import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

# #############################################################################
# Compute clustering with MeanShift
def mean_shift(points):
    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(points, quantile=0.2, n_samples=10)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(points)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("Number of estimated clusters : %d" % n_clusters_)

    return n_clusters_, cluster_centers, labels
