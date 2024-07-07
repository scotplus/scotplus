import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix


def get_barycentre(Xt, pi_samp, device=torch.device("cpu")):
    """
    Purpose:

    Calculate the projection of samples in X_s (not inputted) into the feature
    space of X_t, using the sample coupling matrix (P).

    Parameters:

    Xt: target data of size ny x dy.
    pi_samp: optimal plan of size nx x ny.

    Returns:

    Barycentre of size nx x dy
    """
    
    if isinstance(Xt, np.ndarray):
        Xt = torch.from_numpy(Xt).to(device)
    elif isinstance(Xt, pd.DataFrame):
        Xt = torch.from_numpy(Xt.to_numpy()).to(device)
    barycentre = pi_samp @ Xt / pi_samp.sum(1).reshape(-1, 1)

    return barycentre


def calc_frac_idx(ground, comparison):
    """
    Purpose:

    Calculate the fraction of samples closer than the true match (FOSCTTM)
    metric in one direction. Requires co-assayed data.

    Parameters:

    ground: the ground domain to compare.
    comparison: a comparison matrix with the same sample set as ground.

    Returns:

    fracs: for each sample $x$ in ground, the fraction of samples in comparison
    closer than $x$'s true match.
    xs: for each sample $x$ in ground, the list of samples in comparison closer
    than $x$'s true match.
    """
    
    fracs = []
    x = []
    nsamp = ground.shape[0]
    rank = 0
    for row_idx in range(nsamp):
        euc_dist = np.sqrt(
            np.sum(np.square(np.subtract(ground[row_idx, :], comparison)), axis=1)
        )
        true_nbr = euc_dist[row_idx]
        sort_euc_dist = sorted(euc_dist)
        rank = sort_euc_dist.index(true_nbr)
        frac = float(rank) / (nsamp - 1)

        fracs.append(frac)
        x.append(row_idx + 1)

    return fracs, x


def FOSCTTM(original, projected):
    """
    Purpose:

    Outputs the average of both FOSCTTM directions, as per calc_frac_idx.
    Requires co-assayed data.

    Parameters:

    original: original anchor space of size nx x dx.
    projected: an approximation of the anchor space, also of size nx x dx.

    Returns:

    For each sample $x$ shared by original and projected, the average fraction
    of samples closer than the true match in both directions.
    """
    if isinstance(original, pd.DataFrame):
        original = original.to_numpy()
    elif isinstance(original, torch.Tensor):
        original = original.numpy()
    if isinstance(projected, pd.DataFrame):
        projected = projected.to_numpy()
    elif isinstance(projected, torch.Tensor):
        projected = projected.numpy()
    
    fracs1, _ = calc_frac_idx(original, projected)
    fracs2, _ = calc_frac_idx(projected, original)
    fracs = []
    for i in range(len(fracs1)):
        fracs.append((fracs1[i] + fracs2[i]) / 2)
    return fracs


def compute_graph_distances(
    data, n_neighbors=5, mode="connectivity", metric="correlation"
):
    """
    Purpose:

    Compute a similarity matrix for the samples in data. Construct a $k$
    nearest neighbors graph according to metric with edge weights assigned
    according to mode, and extract intra-domain distances with Dijkstra's
    algorithm.

    Parameters:

    data: tabular dataset over which to compute a similarity matrix. Shape
    nx x nx.
    n_neighbors: $k$, for constructing the $k$-nn graph.
    mode: the metric for determining edge weights in the $k$-nn graph.
    metric: the metric for determining nearest neighbors when constructing the
    graph.

    Returns:

    An intra-domain similarity matrix of shape nx x nx.
    """
    
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    elif isinstance(data, torch.Tensor):
        data = data.numpy()

    graph = kneighbors_graph(
        data, n_neighbors=n_neighbors, mode=mode, metric=metric, include_self=True
    )
    shortestPath = dijkstra(
        csgraph=csr_matrix(graph, dtype=data.dtype), directed=False, return_predecessors=False
    )
    shortestPath = shortestPath.astype(data.dtype)
    max_dist = np.nanmax(shortestPath[shortestPath != np.inf])
    shortestPath[shortestPath > max_dist] = max_dist

    return np.asarray(shortestPath / shortestPath.max())
