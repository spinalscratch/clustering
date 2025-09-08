import random

optics_params = []

min_samples_grid = [5, 10, 20, 30, 50]
xi_grid = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15]
min_cluster_size_grid = [0.02, 0.05, 0.1, 0.15, 0.2]

# combinazioni con triplo for
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


meanshift_params = [
    {'bandwidth': 5.25}
]

birch_params = [
    {'threshold': 3.25, 'branching_factor': 310, 'n_clusters': None}
]