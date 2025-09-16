
# OPTICS
optics_params = [
    {'min_samples': 18, 'xi': 0.007, 'min_cluster_size': 0.25},
    {'min_samples': 8, 'xi': 0.02, 'min_cluster_size': 0.05},
    {'min_samples': 4, 'xi': 0.04, 'min_cluster_size': 0.05},
    {'min_samples': 10, 'xi': 0.002, 'min_cluster_size': 0.15},
    {'min_samples': 10, 'xi': 0.006, 'min_cluster_size': 0.059},
    {'min_samples': 10, 'xi': 0.01, 'min_cluster_size': 0.02},
    {'min_samples': 25, 'xi': 0.013, 'min_cluster_size': 0.125}
]

# MeanShift
meanshift_params = []

# BIRCH
birch_params = [
    {'threshold': 3.2, 'branching_factor': 300, 'n_clusters': 2}
]