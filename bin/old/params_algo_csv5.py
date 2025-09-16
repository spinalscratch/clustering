import random

# OPTICS
optics_params = []

min_samples_grid = [5, 10, 20, 30, 50]
xi_grid = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15]
min_cluster_size_grid = [0.02, 0.05, 0.1, 0.15, 0.2]

# combinazioni a griglia
for ms in min_samples_grid:
    for xi in xi_grid:
        for mcs in min_cluster_size_grid:
            optics_params.append({
                "min_samples": ms,
                "xi": xi,
                "min_cluster_size": mcs
            })

# sampling casuale
random.seed(42)
n_random = 700
for _ in range(n_random):
    optics_params.append({
        "min_samples": random.choice(range(5, 51)),
        "xi": round(random.uniform(0.002, 0.2), 3),
        "min_cluster_size": round(random.uniform(0.02, 0.2), 3)
    })

random.shuffle(optics_params)
optics_params = optics_params[:1000]

# MeanShift
meanshift_params = []

bandwidth_grid = [1, 2, 5, 10, 20]
for bw in bandwidth_grid:
    meanshift_params.append({"bandwidth": bw})

n_random = 100
for _ in range(n_random):
    meanshift_params.append({
        "bandwidth": round(random.uniform(2.5, 6), 2)
    })

random.shuffle(meanshift_params)
meanshift_params = meanshift_params[:200]

# BIRCH
birch_params = []

threshold_grid = [0.1, 0.5, 1.0, 2.0, 3.0]
branching_factor_grid = [20, 50, 100, 200, 300]
n_clusters_grid = [None, 2, 3, 5]

for th in threshold_grid:
    for bf in branching_factor_grid:
        for nc in n_clusters_grid:
            birch_params.append({
                "threshold": th,
                "branching_factor": bf,
                "n_clusters": nc
            })

n_random = 200
for _ in range(n_random):
    birch_params.append({
        "threshold": round(random.uniform(2, 5.0), 2),
        "branching_factor": random.choice(range(20, 400)),
        "n_clusters": random.choice([None, 2, 3, 4, 5, 6])
    })

random.shuffle(birch_params)
birch_params = birch_params[:500]