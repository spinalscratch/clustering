# OPTICS
optics_params = [
    {"min_samples": 12, "xi": 0.005, "min_cluster_size": 0.05},
    {"min_samples": 8, "xi": 0.03, "min_cluster_size": 0.05},
    {"min_samples": 10, "xi": 0.02, "min_cluster_size": 0.05},
    {"min_samples": 5, "xi": 0.007, "min_cluster_size": 0.05},
    {"min_samples": 8, "xi": 0.02, "min_cluster_size": 0.05},
    {"min_samples": 3, "xi": 0.05, "min_cluster_size": 0.02},
    {"min_samples": 4, "xi": 0.03, "min_cluster_size": 0.05},
    {"min_samples": 5, "xi": 0.03, "min_cluster_size": 0.05},
    {"min_samples": 6, "xi": 0.02, "min_cluster_size": 0.05},
    {"min_samples": 5, "xi": 0.002, "min_cluster_size": 0.02},
    {"min_samples": 11, "xi": 0.02, "min_cluster_size": 0.037},
    {"min_samples": 5, "xi": 0.01, "min_cluster_size": 0.02},
    {"min_samples": 5, "xi": 0.1, "min_cluster_size": 0.02},
    {"min_samples": 5, "xi": 0.05, "min_cluster_size": 0.02},
    {"min_samples": 10, "xi": 0.05, "min_cluster_size": 0.02},
    {"min_samples": 6, "xi": 0.15, "min_cluster_size": 0.029},
    {"min_samples": 5, "xi": 0.02, "min_cluster_size": 0.02},
    {"min_samples": 10, "xi": 0.01, "min_cluster_size": 0.02},
    {"min_samples": 5, "xi": 0.15, "min_cluster_size": 0.02},
    {"min_samples": 27, "xi": 0.016, "min_cluster_size": 0.175},
    {"min_samples": 10, "xi": 0.01, "min_cluster_size": 0.05}
]

# MeanShift
meanshift_params = [
    {"bandwidth": 1.95},
    {"bandwidth": 2.0},
    {"bandwidth": 2.05},
    {"bandwidth": 2.15},
    {"bandwidth": 2.2},
    {"bandwidth": 3.6}
]

# BIRCH
birch_params = [
    {"threshold": 3.1, "branching_factor": 280, "n_clusters": 2}
]