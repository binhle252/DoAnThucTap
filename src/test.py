import numpy as np

graph = np.load("graph_arrays.npz")

print(graph["x"].shape)

raw = np.load("initial_node_stats.npy")

print(raw.shape)