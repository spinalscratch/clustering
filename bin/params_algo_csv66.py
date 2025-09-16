
# OPTICS
optics_params = [
    {"min_samples": 9, "xi": 0.01, "min_cluster_size": 0.15},
    {"min_samples": 10, "xi": 0.01, "min_cluster_size": 0.1},
    {"min_samples": 10, "xi": 0.02, "min_cluster_size": 0.05},
    {"min_samples": 34, "xi": 0.019, "min_cluster_size": 0.065},
    {"min_samples": 12, "xi": 0.007, "min_cluster_size": 0.05},
    {"min_samples": 5, "xi": 0.002, "min_cluster_size": 0.1},
    {"min_samples": 5, "xi": 0.01, "min_cluster_size": 0.15},
    {"min_samples": 6, "xi": 0.02, "min_cluster_size": 0.1},
    {"min_samples": 17, "xi": 0.03, "min_cluster_size": 0.15},
    {"min_samples": 4, "xi": 0.02, "min_cluster_size": 0.01},
    {"min_samples": 5, "xi": 0.007, "min_cluster_size": 0.02},
    {"min_samples": 14, "xi": 0.015, "min_cluster_size": 0.05},
    {"min_samples": 15, "xi": 0.019, "min_cluster_size": 0.04},
    {"min_samples": 10, "xi": 0.002, "min_cluster_size": 0.15},
    {"min_samples": 14, "xi": 0.029, "min_cluster_size": 0.072}
]

# MeanShift
meanshift_params = [
    {"bandwidth": 14.59}
]

# BIRCH
birch_params = [
    {"n_clusters": 2, "threshold": 2.0, "branching_factor": 50},
    {"n_clusters": 2, "threshold": 3.0, "branching_factor": 75},
    {"threshold": 1.5, "branching_factor": 300, "n_clusters": 3}
]