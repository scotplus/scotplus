import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix

def get_barycentre(Xt, pi_samp):
    """
    Calculate the barycentre by the following formula: diag(1 / P1_{n_2}) P Xt
    (need to be typed in latex).

    Parameters
    ----------
    Xt: target data of size ny x dy.
    pi_samp: optimal plan of size nx x ny.

    Returns
    ----------
    Barycentre of size nx x dy
    """

    barycentre = pi_samp @ Xt / pi_samp.sum(1).reshape(-1, 1)

    return barycentre

def calc_frac_idx(x1_mat, x2_mat):
    """
    Returns fraction closer than true match for each sample (as an array)
    """
    fracs = []
    x = []
    nsamp = x1_mat.shape[0]
    rank=0
    for row_idx in range(nsamp):
        euc_dist = np.sqrt(np.sum(np.square(np.subtract(x1_mat[row_idx,:], x2_mat)), axis=1))
        true_nbr = euc_dist[row_idx]
        sort_euc_dist = sorted(euc_dist)
        rank =sort_euc_dist.index(true_nbr)
        frac = float(rank)/(nsamp -1)

        fracs.append(frac)
        x.append(row_idx+1)

    return fracs,x

def calc_domainAveraged_FOSCTTM(x1_mat, x2_mat):
    """
    Outputs average FOSCTTM measure (averaged over both domains)
    Get the fraction matched for all data points in both directions
    Averages the fractions in both directions for each data point
    """
    fracs1,xs = calc_frac_idx(x1_mat, x2_mat)
    fracs2,xs = calc_frac_idx(x2_mat, x1_mat)
    fracs = []
    for i in range(len(fracs1)):
        fracs.append((fracs1[i]+fracs2[i])/2)  
    return fracs

def knn_dist(X, k, mode= "connectivity", metric="correlation"):
    assert (mode in ["connectivity", "distance"]), "Norm argument has to be either one of 'connectivity', or 'distance'. "
    if mode == "connectivity":
        include_self = True
    else:
        include_self = False
    X_graph = kneighbors_graph(X, k, mode=mode, metric=metric, include_self=include_self)
    
    X_spath = dijkstra(csgraph= csr_matrix(X_graph), directed=False, return_predecessors=False)
    X_max = np.nanmax(X_spath[X_spath != np.inf])
    X_spath[X_spath > X_max] = X_max

    # Finally, normalize the distance matrix:
    Dx = X_spath/X_spath.max()

    return Dx

def compute_graph_distances(data, n_neighbors=5, mode="distance", metric="correlation"):
        graph=kneighbors_graph(data, n_neighbors=n_neighbors, mode=mode, metric=metric, include_self=True)
        shortestPath=dijkstra(csgraph= csr_matrix(graph), directed=False, return_predecessors=False)
        max_dist=np.nanmax(shortestPath[shortestPath != np.inf])
        shortestPath[shortestPath > max_dist] = max_dist

        return np.asarray(shortestPath)