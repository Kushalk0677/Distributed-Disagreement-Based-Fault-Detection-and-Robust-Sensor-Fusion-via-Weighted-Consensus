"""
network.py — sensor network topology builder

Supports:
  - Random Geometric Graph (RGG) topology
  - Metropolis-Hastings mixing weights (for consensus)
  - Random and clustered fault placement
  - Connectivity verification and repair
"""

import numpy as np
from collections import deque


def build_random_geometric_graph(N: int, radius: float, seed: int = 42) -> dict:
    """
    Place N sensors uniformly in [0,1]^2.
    Two sensors are neighbors if Euclidean distance <= radius.
    Ensures connectivity by increasing radius if needed.
    """
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, 1, size=(N, 2))

    r = radius
    for _ in range(10):                        # auto-increase radius until connected
        adj = np.zeros((N, N), dtype=bool)
        neighbors = [[] for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(positions[i] - positions[j]) <= r:
                    adj[i, j] = adj[j, i] = True
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        if _is_connected(neighbors, N):
            break
        r *= 1.15

    degrees = np.array([len(nb) for nb in neighbors])
    return dict(positions=positions, neighbors=neighbors,
                adj_matrix=adj, degrees=degrees, radius=r)


def build_metropolis_weights(graph: dict) -> np.ndarray:
    """
    Compute (N, N) Metropolis-Hastings mixing matrix for consensus.
    Guarantees convergence to the unweighted average.
    P_ij = 1 / (1 + max(d_i, d_j))  for j in N_i
    P_ii = 1 - sum_{j in N_i} P_ij
    """
    N       = len(graph["neighbors"])
    degrees = graph["degrees"]
    P       = np.zeros((N, N))
    for i in range(N):
        for j in graph["neighbors"][i]:
            P[i, j] = 1.0 / (1 + max(int(degrees[i]), int(degrees[j])))
        P[i, i] = 1.0 - P[i].sum()
    return P


def assign_faults_random(N: int, fault_fraction: float,
                          fault_type: str, rng: np.random.Generator):
    """Uniformly random fault assignment."""
    n_faulty   = max(1, int(N * fault_fraction))
    faulty_idx = rng.choice(N, size=n_faulty, replace=False)
    mask       = np.zeros(N, dtype=bool)
    mask[faulty_idx] = True
    types = ["none"] * N
    for i in faulty_idx:
        types[i] = fault_type
    return mask, types


def assign_faults_clustered(positions: np.ndarray, fault_fraction: float,
                             fault_type: str, rng: np.random.Generator):
    """
    Place faults in a geographic cluster (a random disk).
    This creates neighborhoods where >50% of sensors may be faulty,
    the hard case for local-median-based methods.
    """
    N        = len(positions)
    n_faulty = max(1, int(N * fault_fraction))
    center   = rng.uniform(0.2, 0.8, size=2)

    dists   = np.linalg.norm(positions - center, axis=1)
    ordered = np.argsort(dists)
    faulty_idx = ordered[:n_faulty]

    mask  = np.zeros(N, dtype=bool)
    mask[faulty_idx] = True
    types = ["none"] * N
    for i in faulty_idx:
        types[i] = fault_type
    return mask, types


def _is_connected(neighbors, N):
    visited, q = set(), deque([0])
    while q:
        v = q.popleft()
        if v in visited:
            continue
        visited.add(v)
        q.extend(neighbors[v])
    return len(visited) == N
