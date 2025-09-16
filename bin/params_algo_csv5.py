# OPTICS
optics_params = [
    {'min_samples': 11, 'xi': 0.05, 'min_cluster_size': 0.1},
    {'min_samples': 10, 'xi': 0.03, 'min_cluster_size': 0.15},
    {'min_samples': 5, 'xi': 0.035, 'min_cluster_size': 0.1},
    {'min_samples': 5, 'xi': 0.036, 'min_cluster_size': 0.1},
    {'min_samples': 12, 'xi': 0.032, 'min_cluster_size': 0.049},
    {'min_samples': 13, 'xi': 0.01, 'min_cluster_size': 0.086}
]

# MeanShift
meanshift_params = [
    {'bandwidth': 3.35},
    {'bandwidth': 3.39}
]

# BIRCH
birch_params = [
    {'threshold': 2.0, 'branching_factor': 100,'n_clusters': 4},
    {'threshold': 2.3, 'branching_factor': 120, 'n_clusters': 2},
    {'threshold': 2.65, 'branching_factor': 190, 'n_clusters': 2},
    {'threshold': 3.1, 'branching_factor': 280, 'n_clusters': 2},
    {'threshold': 2.47, 'branching_factor': 72, 'n_clusters': 5},
    {'threshold': 2.31, 'branching_factor': 291, 'n_clusters': 2}
]