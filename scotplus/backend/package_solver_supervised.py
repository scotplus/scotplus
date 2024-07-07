import torch
import numpy as np
import pandas as pd
from functools import partial

# formatted with black


def approx_kl(p, q):
    """
    Purpose:

    Calculates KL without mass shifts, for the purposes of calculating full KL
    computations as well as adjustments in the local cost matrix as seen in the
    functions below.

    Parameters:

    p - an n-D probability measure
    q - an n-D probability measure

    Math:

    Return $p \log{\frac{p}{q}} = \langle \log{\frac{p}{q}}, p \rangle$
    Note: 0 log 0 = 0.
    """
    return (
        torch.nan_to_num(p * p.log(), nan=0.0, posinf=0.0, neginf=0.0).sum()
        - (p * q.log()).sum()
    )


def kl(p, q):
    """
    Purpose:

    Calculate KL divergence in the most general case, used only in global
    cost calculation where additional constants matter.

    Parameters:

    p - an n-D probability measure
    q - an n-D probability measure

    Math:

    Return  $KL(p|q) = p \log{\frac{p}{q}} - m(p) + m(q)$
    Note: m(p) is the total mass of a given measure p
    """
    return approx_kl(p, q) - p.sum() + q.sum()


def quad_kl(mu, nu, alpha, beta):
    """
    Purpose:

    Calculate the KL divergence between two product measures (i.e., two n+1-D
    measures defined by four marginal n-D measures).

    Parameters:

    mu: an n-D probability measure
    nu: an n-D probability measure
    alpha: an n-D probability measure with the same size as mu
    beta: an n-D probability measure with the same size as nu

    Math:

    Return $KL(\mu \otimes \nu | \alpha \otimes \beta) = m(\nu)KL(\mu | \alpha)
    + m(\mu)KL(\nu | \beta) + (m(\mu) - m(\alpha))(m(\nu) - m(\beta))$
    Note: m(p) is the total mass of a given measure p
    """
    m_mu = mu.sum()
    m_nu = nu.sum()
    m_alpha = alpha.sum()
    m_beta = beta.sum()
    const = (m_mu - m_alpha) * (m_nu - m_beta)

    return m_nu * kl(mu, alpha) + m_mu * kl(nu, beta) + const


def uot_ent(cost, init_duals, tuple_log_p, params, n_iters, tol, eval_freq):
    """
    Purpose:

    Solve entropic UOT using Sinkhorn algorithm. Allow rho_x and/or rho_y to be
    infinity but epsilon must be strictly positive.

    Parameters:

    cost - 2-D cost matrix C
    init_duals - initialization vectors for the dual formulation of UOT
    tuple_log_p - log supports for coupling matrix to be optimized, $\log\
    {\mu_x}, \log{\mu_y}, \log{\mu_x \otimes \mu_y}$
    params - $\epsilon, \rho_x, \rho_y$
    n_iters - number of iterations for sinkhorn convergence
    tol - early stopping tolerance for sinkhorn convergence
    eval_freq - number of iterations to check for early stopping

    Math:

    Return \pi that approximately minimizes $\langle C, \pi \rangle + \rho_x KL
    (\pi_{\#1} | \mu_x) + \rho_y KL(\pi_{\#2} | \mu_y) + \epsilon KL(\pi | 
    \mu_x \otimes \mu_y$
    """

    eps, rho1, rho2 = params
    log_a, log_b, ab = tuple_log_p
    f, g = init_duals

    tau1 = 1 if torch.isinf(rho1) else rho1 / (rho1 + eps)
    tau2 = 1 if torch.isinf(rho2) else rho2 / (rho2 + eps)

    for idx in range(n_iters):
        f_prev = f.detach().clone()
        if rho2 == 0:  # semi-relaxed
            g = torch.zeros_like(g)
        else:
            g = -tau2 * ((f + log_a)[:, None] - cost / eps).logsumexp(dim=0)

        if rho1 == 0:  # semi-relaxed
            f = torch.zeros_like(f)
        else:
            f = -tau1 * ((g + log_b)[None, :] - cost / eps).logsumexp(dim=1)

        if (idx % eval_freq == 0) and (f - f_prev).abs().max().item() < tol:
            break

    pi = ab * (f[:, None] + g[None, :] - cost / eps).exp()

    return (f, g), pi


def get_local_cost(
    pi, prod_simplif_mats, measures, supervision, hyperparams, entropic_mode
):
    """
    Purpose:

    Calculate cost of the UOT, which can work for both COOT-like and GW-like
    sub-optimizations. For example, we might have X = Dx, Y = Dy, pi = P' for
    optimizing P in GW-like context, or we might have X = X.T, Y = Y.T, pi = P
    for optimizing Q in COOT-like context (omitting two other present
    permutations, i.e. optimizing P' in GW-like context and optimizing P in
    COOT-like context). We'll mostly work with P, P', Q for math notation in
    other functions; using pi here as a generic coupling matrix.

    Parameters:

    pi - a given coupling matrix $\pi$
    prod_simplif_mats - tabular datasets and their squares, formatted as $X^2,
    Y^2, X, Y$
    measures - supports for $\pi, \mu_x, \mu_y, \mu_x \otimes \mu_y$
    hyperparams - $\epsilon, \rho_x, \rho_y$
    supervision - $\beta$, $D$
    entropic_mode - flags whether or not to regularize $\pi$ jointly with the
    coupling matrix this cost will be used to optimize

    Math:

    Return $(X^{\odot 2} \pi_{#1} + Y^{\odot 2} pi_{#2} - 2X \pi Y^T) +
        \beta D +
        \rho_x \langle \log{\frac{\pi_{\#1}}{\mu_x}, \pi_{\#1} \rangle +
        \rho_y \langle \log{\frac{\pi_{\#2}}{\mu_y}, \pi_{\#2} \rangle +
        \epsilon \langle \log{\frac{\pi}{\mu_x \otimes \mu_y}}, \pi \rangle$
    Note: for more detail, see the supplementary material in Baker, et. al. for
    how we isolate these terms.
    """

    eps, rho1, rho2 = hyperparams
    beta, D = supervision
    X_sqr, Y_sqr, X, Y = prod_simplif_mats
    support1, support2, support_prod = measures

    # $X^{\odot 2} \pi_{#1} + Y^{\odot 2} pi_{#2} - 2X \pi Y^T + \beta D$
    pi1, pi2 = pi.sum(1), pi.sum(0)
    A = X_sqr @ pi1
    B = Y_sqr @ pi2
    cost = A[:, None] + B[None, :] - 2 * X @ pi @ Y.T + beta * D

    if entropic_mode == "joint":
        # if ent. reg. is independent, other coupling matrix entropic KL term
        # drops out as a constant for \textit{local} cost
        # $\epsilon \langle \log{\frac{\pi}{\mu_x \otimes \mu_y}}, \pi \rangle$
        cost = cost + eps * approx_kl(pi, support_prod)
    if not (rho1 == float("inf")):
        # $\rho_x \langle \log{\frac{\pi_{\#1}}{\mu_x}, \pi_{\#1} \rangle$
        cost = cost + rho1 * approx_kl(pi1, support1)
    if not (rho2 == float("inf")):
        # $\rho_y \langle \log{\frac{\pi_{\#2}}{\mu_y}, \pi_{\#2} \rangle$
        cost = cost + rho2 * approx_kl(pi2, support2)

    return cost


def get_full_cost(
    pi, alpha, hyperparams, supervision, entropic_mode, prod_simplif_mats, measures
):
    """
    Purpose:

    Calculate the full UAGW cost for display after each block iteration.

    Parameters:

    pi - tuple of three coupling matrices P, P', Q
    alpha - $\alpha$ hyperparameter that trades UGW and UCOOT loss
    hyperparams - two tuples containing $\eps^{gw}, rho_x^{gw}, rho_y^{gw}$ and
    $\eps^{coot}, rho_x^{coot}, rho_y^{coot}$
    supervision - two tuples containing $\beta_s, D_s$ and $\beta_f, D_f$
    entropic_mode - flags whether or not to regularize P jointly with Q and P'
    prod_simplif_mats - two tuples containing $D_x^{\odot 2}, D_y^{\odot 2},
    D_x, D_y$ and $X^{\odot 2}, Y^{\odot 2}, X, Y$
    measures - two tuples of support measures, $\mu_{sx}, \mu_{sy}, \mu_{sx}
    \otimes \mu_{sy}$ and $\mu_{fx}, \mu_{fy}, \mu_{fx} \otimes \mu_{fy}$

    Math:

    Return full UAGW loss, as noted in our supplementary materials. In short
    form: $\alpha*[\langle |Dx - Dy|^2, P \otimes P' \rangle + reg. + relax.] +
    (1 - \alpha)*[\langle |X - Y|^2, P \otimes Q \rangle + reg. + relax.]$
    """

    # $P, P', Q$
    pi_samp, pi_samp_prime, pi_feat = pi

    # $\epsilon^{gw}, \rho_x^{gw}, \rho_y^{gw}; \epsilon^{coot}, \rho_x^{coot},
    # \rho_y^{coot}$
    (eps_gw, rho1_gw, rho2_gw), (eps_coot, rho1_coot, rho2_coot) = hyperparams
    (beta_gw, D_gw), (beta_coot, D_coot) = supervision

    # $D_x^{\odot 2}, D_y^{\odot 2}, D_x, D_y$; $X^{\odot 2}, Y^{\odot 2}, X, Y$
    (Dx_sqr, Dy_sqr, Dx, Dy), (X_sqr, Y_sqr, X, Y) = prod_simplif_mats

    # $\mu_{sx}, \mu_{sy}, \mu_{sx} \otimes \mu_{sy}$; $\mu_{fx}, \mu_{fy}, \mu_{fx} \otimes \mu_{fy}$
    (support1_samp, support2_samp, support_prod_samp), (
        support1_feat,
        support2_feat,
        support_prod_feat,
    ) = measures

    # $P_{\#1}, P_{\#2}$
    pi1_samp, pi2_samp = pi_samp.sum(1), pi_samp.sum(0)

    gw_cost, ent_gw_cost, coot_cost, ent_coot_cost = 0, 0, 0, 0

    if alpha > 0:
        # $P'_{\#1}, P'_{\#2}$
        pi1_samp_prime, pi2_samp_prime = pi_samp_prime.sum(1), pi_samp_prime.sum(0)

        # $\langle |D_x - D_y|^2, P \otimes P' \rangle$
        # using A/a and B/b as generic substitutes in the eqn $(a - b)^2 = a^2
        # + b^2 - 2ab$
        Da_sqr = (Dx_sqr @ pi1_samp_prime).dot(pi1_samp)
        Db_sqr = (Dy_sqr @ pi2_samp_prime).dot(pi2_samp)
        Dab = (Dx @ pi_samp_prime @ Dy.T) * pi_samp
        gw_cost = Da_sqr + Db_sqr - 2 * Dab.sum()

        gw_cost = gw_cost + beta_gw * (D_gw * pi_samp).sum()
        gw_cost = gw_cost + beta_gw * (D_gw * pi_samp_prime).sum()

        if rho1_gw != float("inf"):
            # $rho_x^{gw}KL(P_{\#1} \otimes P'_{\#1} | \mu_{sx} \otimes \mu_{sx})$
            gw_cost = gw_cost + rho1_gw * quad_kl(
                pi1_samp, pi1_samp_prime, support1_samp, support1_samp
            )
        if rho2_gw != float("inf"):
            # $rho_y^{gw}KL(P_{\#2} \otimes P'_{\#2} | \mu_{sy} \otimes \mu_{sy})$
            gw_cost = gw_cost + rho2_gw * quad_kl(
                pi2_samp, pi2_samp_prime, support2_samp, support2_samp
            )

        ent_gw_cost = gw_cost.detach().clone()
        if entropic_mode == "joint":
            # cost under joint ent. reg.
            # $\eps^{gw}*KL(P \otimes P' | \mu_{sx} \otimes \mu_{sx} \otimes
            # \mu_{sy} \otimes \mu_{sy})$
            ent_gw_cost += eps_gw * quad_kl(
                pi_samp, pi_samp_prime, support_prod_samp, support_prod_samp
            )
        elif entropic_mode == "independent":
            # cost under indep. ent. reg.
            # $\eps^{gw}(KL(P | \mu_{sx} \otimes \mu_{sy}) + KL(P' | \mu_{sx}
            # \otimes \mu_{sy}))$
            ent_gw_cost += eps_gw * (
                kl(pi_samp, support_prod_samp) + kl(pi_samp_prime, support_prod_samp)
            )

    if alpha < 1:
        # $Q_{\#1}, Q_{\#2}$
        pi1_feat, pi2_feat = pi_feat.sum(1), pi_feat.sum(0)

        # $\langle |X - Y|^2, P \otimes Q \rangle$
        A_sqr = (X_sqr @ pi1_feat).dot(pi1_samp)
        B_sqr = (Y_sqr @ pi2_feat).dot(pi2_samp)
        AB = (X @ pi_feat @ Y.T) * pi_samp
        coot_cost = A_sqr + B_sqr - 2 * AB.sum()

        coot_cost = coot_cost + beta_coot * (D_coot * pi_feat).sum()
        coot_cost = coot_cost + beta_coot * (D_gw * pi_samp).sum()

        if rho1_coot != float("inf"):
            # $rho_x^{coot}KL(P_{\#1} \otimes Q_{\#1} | \mu_{sx} \otimes \mu_{fx})$
            coot_cost = coot_cost + rho1_coot * quad_kl(
                pi1_samp, pi1_feat, support1_samp, support1_feat
            )
        if rho2_coot != float("inf"):
            # $rho_y^{coot}KL(P_{\#2} \otimes Q_{\#2} | \mu_{sy} \otimes \mu_{fy})$
            coot_cost = coot_cost + rho2_coot * quad_kl(
                pi2_samp, pi2_feat, support2_samp, support2_feat
            )

        ent_coot_cost = coot_cost.detach().clone()
        if entropic_mode == "joint":
            # cost under joint ent. reg.
            # $\eps^{coot}*KL(P \otimes Q | \mu_{sx} \otimes \mu_{fx} \otimes
            # \mu_{sy} \otimes \mu_{fy})$
            ent_coot_cost += eps_coot * quad_kl(
                pi_samp, pi_feat, support_prod_samp, support_prod_feat
            )
        elif entropic_mode == "independent":
            # cost under indep. ent. reg.
            # $\eps^{coot}(KL(P | \mu_{sx} \otimes \mu_{sy}) + KL(Q | \mu_{fx}
            # \otimes \mu_{fy}))$
            ent_coot_cost += eps_coot * (
                kl(pi_samp, support_prod_samp) + kl(pi_feat, support_prod_feat)
            )

    # $\alpha*[\langle |Dx - Dy|^2, P \otimes P' \rangle + relax.] +
    # (1 - \alpha)*[\langle |X - Y|^2, P \otimes Q \rangle + relax.]$
    cost = alpha * gw_cost + (1 - alpha) * coot_cost

    # $\alpha*[\langle |Dx - Dy|^2, P \otimes P' \rangle + reg. + relax.] +
    # (1 - \alpha)*[\langle |X - Y|^2, P \otimes Q \rangle + reg. + relax.]$
    ent_cost = alpha * ent_gw_cost + (1 - alpha) * ent_coot_cost

    return cost.item(), ent_cost.item()


def solver(
    X=None,
    Y=None,
    Dx=None,
    Dy=None,
    px=(None, None),
    py=(None, None),
    eps=(1e-3, 1e-3),
    rho=(float("inf"), float("inf"), float("inf"), float("inf")),
    entropic_mode="independent",
    eps_mode="true",
    alpha=1,
    beta=(0, 0),
    D=(0, 0),
    init_pi=(None, None, None),
    init_duals=(None, None, None),
    verbose=True,
    log=False,
    eval_bcd=1,
    eval_uot=1,
    tol_bcd=1e-7,
    nits_bcd=15,
    tol_uot=1e-5,
    nits_uot=int(1e5),
    tol_gw=1e-7,
    nits_gw=10,
    early_stopping_tol=1e-5,
    device=torch.device("cpu"),
):
    """
    Purpose:

    Solve for P and Q that minimize UAGW loss, given $X, Y, D_x, D_y$ and regularization/relaxation hyperparameters.

    Parameters:

    X: matrix of size nx x dx. First input data.
    Y: matrix of size ny x dy. Second input data.
    Dx: sample distance matrix of size nx x nx.
    Dy: sample distance matrix of size ny x ny.

    px: tuple of 2 vectors of length (nx, dx). Measures assigned on rows and
    columns of X; uniform by default.
    py: tuple of 2 vectors of length (ny, dy). Measures assigned on rows and
    columns of Y; uniform distributions by default.

    eps: scalar or tuple of scalars.
        $\epsilon_{gw}$ and $\epsilon_{coot}$ from our supplementary materials.
        In the independent case, $\epsilon_{gw}$ is used to regularize both $P$
        and $P'$.
    rho: scalar or tuple of scalars.
        When rho is a scalar, we set
        $\rho_x^{gw} = \rho_y^{gw} = \rho_x^{coot} = \rho_y^{coot} = rho$.
        When rho is a tuple of two scalars, we set
        $\rho_x^{gw} = \rho_y^{gw} = rho[0]$ and $\rho_x^{coot} =
        \rho_y^{coot} = rho[1]$.
        When rho is a tuple of four scalars, we set
        $\rho_x^{gw} = \rho_y^{gw} = rho[0]$ and $\rho_x^{coot} =
        \rho_y^{coot} = rho[1]$.
        Setting rho = float('inf') recovers balanced formulations.

    entropic_mode: string
        entropic_mode = "joint": jointly regularize P, P' and
        P, Q.
        entropic_mode = "independent": independently regularize P, P', and Q.
    eps_mode: string
        eps_mode = "true": correct regularization according to calculations in
        our supplementary material.
        eps_mode = "fast": use uniform epsilon for all inner UOT tasks; i.e.,
        ignore the fact that the eps used in optimizing P' should be
        $\alpha \epsilon_{gw}$ and instead just use $\epsilon_{gw}$. We do not
        recommend using this keyword, in general, but feel free to experiment.

    alpha: scalar between 0 and 1, tradeoff parameter between COOT and GW loss.
        $\alpha = 0$ recovers UCOOT.
        $\alpha = 1$ recovers UGW.

    beta: scalar or tuple of nonnegative scalars.
        Used to magnify or decrease supervision as per D below. beta[0]
        modifies sample supervision while beta[1] modifies feature supervision.
        In other words, $\beta_s = beta[0]$ and $\beta_f = beta[1]$
    D: tuple of matrices of size (nx x ny) and (dx x dy), used for linear
    sample and feature supervision
        A smaller value at $D_{ij}$ encourages transport from $X_i$ to $Y_j$.
        $D_s = D[0]$, $D_f = D[1]$.

    init_pi: tuple of matrices of size nx x ny and dx x dy if not None.
        Initialization of sample and feature couplings.
    init_duals: tuple of tuple of vectors of size (nx,ny) and (dx, dy) if not
    None.
        Initialization of sample and feature dual vectos if using Sinkhorn
        algorithm.

    verbose: if True then print the recorded cost at each BCD iteration.
    log: if True then record cost and entropic cost in individual lists.

    eval_bcd: multiplier of iteration at which the cost is calculated. For
    example, if eval_bcd = 10, then the full cost is calculated at iteration
    10, 20, 30, etc...
    eval_uot: multiplier of iteration at which the old and new duals are
    compared in the Sinkhorn algorithm.

    tol_bcd: tolerance of BCD scheme (based on changes in the coupling matrices,
    not total cost)
    nits_bcd: number of BCD iterations.

    tol_uot: tolerance of Sinkhorn algorithm.
    nits_uot: number of Sinkhorn iterations.

    tol_gw: tolerance of GW block.
    nits_gw: number of GW iterations.

    early_stopping_tol: threshold for early stopping according to UAGW cost
    differences.

    device: pytorch device to use, inferred if not provided.

    Returns:

    pi_samp: matrix of size nx x ny. Sample matrix.
    pi_samp_prime: matrix of size nx x ny. Sample GW matrix, better to use
    pi_samp.
    pi_feat: matrix of size dx x dy. Feature matrix.
    log_cost: if log is True, return a list of costs (without taking into
    account the regularization term).
    log_ent_cost: if log is True, return a list of entropic costs.

    Format is [pi_samp, pi_samp_prime, pi_feat] if log is False and otherwise
    [(pi_samp, pi_samp_prime, pi_feat), log_cost, log_ent_cost]
    """

    # syncing object types and dtypes
    dtype = None
    if device is None:
        device = torch.device("cpu")
    if device == torch.device("cpu"):
        for data in [X, Y, Dx, Dy]:
            if data is None:
                continue
            if isinstance(data, torch.Tensor):
                device = data.device

    new_data = []

    # not an ideal design here, but need to populate all unique vars with
    # correct type
    for data in (
        [X, Y, Dx, Dy]
        + [x for x in init_pi]
        + [x for x in init_duals]
        + [x for x in D]
    ):
        if data is None:
            new_data.append(None)
        elif isinstance(data, int):
            new_data.append(data)
            continue
        elif isinstance(data, np.ndarray):
            new_data.append(torch.from_numpy(data).to(device))
            continue
        elif isinstance(data, pd.DataFrame):
            new_data.append(torch.from_numpy(data.to_numpy()).to(device))
            continue
        elif isinstance(data, torch.Tensor):
            new_data.append(data)
        else:
            raise ValueError(
                f"Our current interface exclusively supports numpy, pandas, and torch array/tensor objects, but yours was type {data.type()}."
            )
    init_pi, init_duals = [[None] * 3] * 2
    D = [None] * 2
    (
        X,
        Y,
        Dx,
        Dy,
        init_pi[0],
        init_pi[1],
        init_pi[2],
        init_duals[0],
        init_duals[1],
        init_duals[2],
        D[0],
        D[1],
    ) = new_data

    dtype = None
    for data in [X, Y, Dx, Dy] + [x for x in init_pi] + [x for x in init_duals] + [x for x in D]:
        if data is None or isinstance(data, int):
            continue
        if dtype is not None:
            if not (dtype == data.dtype):
                raise ValueError("Tabular datasets all must have same dtype.")
        dtype = data.dtype

    # support dimension setup
    if (Dx is None or Dy is None) and (X is None or Y is None):
        raise ValueError("Must have at least one pair of tabular datasets.")
    elif Dx is None or Dy is None:
        nx, dx = X.shape
        ny, dy = Y.shape
        if not (alpha == 0):
            print("Changing alpha to 0, no GW data.")
        alpha = 0
    elif X is None or Y is None:
        nx = Dx.shape[0]
        ny = Dy.shape[0]
        if not (alpha == 1):
            print("Changing alpha to 1, no COOT data.")
        alpha = 1
    else:
        nx, dx = X.shape
        ny, dy = Y.shape

    # epsilon setup
    if isinstance(eps, float) or isinstance(eps, int):
        # under indep. regularization, eps[0] used for both sample matrices
        # \epsilon_{gw}, \epsilon_{coot}
        eps = (eps, eps)
    if not isinstance(eps, tuple):
        raise ValueError("Epsilon must be either a scalar or a tuple of scalars.")
    if len(eps) == 1:
        eps = (eps[0], eps[0])
    if not (len(eps) == 2):
        raise ValueError(
            "Must be at most two epsilon parameters (epsilon for GW and epsilon for COOT, or for P, P' and for Q)."
        )

    # rho setup
    if isinstance(rho, float) or isinstance(rho, int):
        # \rho^x_{gw}, rho^y_{gw}, rho^x_{coot}, rho^y_{coot}
        rho = (rho, rho, rho, rho)
    if not isinstance(rho, tuple):
        raise ValueError("Rho must be either a scalar or a tuple of scalars.")
    if len(rho) == 2:
        rho = (rho[0], rho[0], rho[1], rho[1])
    if not (len(rho) == 4):
        raise ValueError(
            "Must be at most four rho parameters (two for the GW marginals and two for the COOT marginals)."
        )

    # alpha setup
    if not (isinstance(alpha, float) or isinstance(alpha, int)):
        raise ValueError("Alpha must be a scalar.")
    if not (0 <= alpha and 1 >= alpha):
        raise ValueError("Alpha must be between zero and one.")

    # \mu_{sx}, \mu_{fx}
    px_samp, px_feat = px

    # \mu_{sy}, \mu_{fy}
    py_samp, py_feat = py

    # populating each of the four measures
    if px_samp is None:
        px_samp = torch.ones(nx).to(device).to(dtype) / nx
    if alpha < 1 and px_feat is None:
        px_feat = torch.ones(dx).to(device).to(dtype) / dx
    if py_samp is None:
        py_samp = torch.ones(ny).to(device).to(dtype) / ny
    if alpha < 1 and py_feat is None:
        py_feat = torch.ones(dy).to(device).to(dtype) / dy

    # \mu_{sx} \otimes \mu_{sy}
    pxy_samp = px_samp[:, None] * py_samp[None, :]

    # \mu_{fx} \otimes \mu_{fy}
    pxy_feat = None if alpha == 1 else px_feat[:, None] * py_feat[None, :]

    # for cost calculations
    sample_measures = (px_samp, py_samp, pxy_samp)
    feature_measures = (px_feat, py_feat, pxy_feat)

    # for UOT sinkhorn iteration
    sample_log_measures = (px_samp.log(), py_samp.log(), pxy_samp)
    feature_log_measures = (
        (None, None, None) if alpha == 1 else (px_feat.log(), py_feat.log(), pxy_feat)
    )

    # local cost precomputations
    # will be used to compute X^{\otimes 2} @ \pi_{#1} + Y^{\otimes 2} @ \pi_{#2} + 2X\piY^T
    empty_mats = (None, None, None, None)
    samp_lc_gw, samp_prime_lc_gw, samp_lc_coot, feat_lc_coot = (
        empty_mats,
        empty_mats,
        empty_mats,
        empty_mats,
    )
    if alpha < 1:
        X_sqr = X**2
        Y_sqr = Y**2
        feat_lc_coot = (X_sqr, Y_sqr, X, Y)
        samp_lc_coot = (X_sqr.T, Y_sqr.T, X.T, Y.T)

    if alpha > 0:
        Dx_sqr = Dx**2
        Dy_sqr = Dy**2
        samp_prime_lc_gw = (Dx_sqr, Dy_sqr, Dx, Dy)
        samp_lc_gw = (Dx_sqr.T, Dy_sqr.T, Dx.T, Dy.T)

    # initialize couplings
    pi_samp, pi_samp_prime, pi_feat = init_pi
    if pi_samp is None:
        pi_samp = pxy_samp  # size nx x ny
    if pi_samp_prime is None:
        pi_samp_prime = pxy_samp  # size nx x ny
    if alpha < 1 and pi_feat is None:
        pi_feat = pxy_feat  # size dx x dy

    # initialize duals for UOT iteration
    duals_samp, duals_samp_prime, duals_feat = init_duals
    if duals_samp is None:
        duals_samp = (
            torch.zeros_like(px_samp),
            torch.zeros_like(py_samp),
        )  # shape nx, ny
    if duals_samp_prime is None:
        duals_samp_prime = (
            torch.zeros_like(px_samp),
            torch.zeros_like(py_samp),
        )  # shape nx, ny
    if alpha < 1 and duals_feat is None:
        duals_feat = (
            torch.zeros_like(px_feat),
            torch.zeros_like(py_feat),
        )  # shape dx, dy

    # packing hyperparameters
    gw_params = (eps[0], rho[0], rho[1])
    sample_supervision = (beta[0], D[0])
    coot_params = (eps[1], rho[2], rho[3])
    feature_supervision = (beta[1], D[1])

    # partial UOT solver functions with convergence hyperparameters (for
    # simpler representation below)
    gw_sinkhorn = partial(uot_ent, n_iters=nits_uot, tol=tol_uot, eval_freq=eval_uot)
    coot_sinkhorn = partial(uot_ent, n_iters=nits_uot, tol=tol_uot, eval_freq=eval_uot)

    # partial local cost computations with static hyperparameters (for simpler
    # representation below)
    gw_local = partial(
        get_local_cost, hyperparams=gw_params, entropic_mode=entropic_mode
    )
    coot_local = partial(
        get_local_cost, hyperparams=coot_params, entropic_mode=entropic_mode
    )

    # total cost computations with static hyperparameters (for simpler
    # representation below)
    total_cost = partial(
        get_full_cost,
        alpha=alpha,
        hyperparams=(gw_params, coot_params),
        supervision=(sample_supervision, feature_supervision),
        entropic_mode=entropic_mode,
        prod_simplif_mats=(samp_prime_lc_gw, feat_lc_coot),
        measures=(sample_measures, feature_measures),
    )

    # cost logs
    log_cost = []
    log_ent_cost = [float("inf")]

    for i in range(nits_bcd):
        # (X**2 @ Q_{#1} + Y**2 @ Q_{#2} - 2X @ Q @ Y.T) +
        # \rho^x_{coot} \langle \log{\frac{Q_{#1}}{\mu_{fx}}, Q_{#1} \rangle +
        # \rho^y_{coot} \langle \log{\frac{Q_{#2}}{\mu_{fy}}, Q_{#2} \rangle +
        # \epsilon_{coot} \langle \log{\frac{Q}{\mu_{fx} \otimes \mu_{fy}}}, Q \rangle

        uot_coot = (
            0
            if alpha == 1
            else coot_local(pi_feat, feat_lc_coot, feature_measures, sample_supervision)
        )

        # if GW component, run multiple GW minimizations before COOT component
        if alpha > 0:
            for j in range(nits_gw):
                pi_samp_prev = pi_samp.detach().clone()

                # (Dx**2 @ P'_{#1} + Dy**2 @ P'_{#2} - 2Dx @ P' @ Dy.T) +
                # \rho^x_{gw} \langle \log{\frac{P'_{#1}}{\mu_{sx}}, P'_{#1} \rangle +
                # \rho^y_{gw} \langle \log{\frac{P'_{#2}}{\mu_{sy}}, P'_{#2} \rangle +
                # \epsilon_{gw} \langle \log{\frac{P'}{\mu_{sx} \otimes \mu_{sy}}}, P' \rangle
                uot_gw = gw_local(
                    pi_samp_prime, samp_prime_lc_gw, sample_measures, sample_supervision
                )

                # \alpha [GW cost] + (1 - \alpha) [COOT cost]
                full_uot_cost = alpha * uot_gw + (1 - alpha) * uot_coot

                # for joint reg., we find
                # \alpha m(P') \epsilon_{gw} + (1 - \alpha) m(Q) \epsilon_{coot}
                # for indep. reg., we find
                # \alpha \epsilon_{gw} + (1 - \alpha) \epsilon_{coot}
                feat_mass = 1 if alpha == 1 else pi_feat.sum()
                new_eps = (
                    alpha * pi_samp_prime.sum() * eps[0]
                    + (1 - alpha) * feat_mass * eps[1]
                    if entropic_mode == "joint"
                    else eps[0]
                )

                # \alpha m(P') rho^x_{gw}
                new_rho1 = alpha * pi_samp_prime.sum() * rho[0]

                # \alpha m(P') rho^y_{gw}
                new_rho2 = alpha * pi_samp_prime.sum() * rho[1]

                # if statement protects against inf*0 or None calculations
                if not (alpha == 1):
                    # (1 - \alpha) m(Q) rho^x_{coot}
                    new_rho1 += (1 - alpha) * feat_mass * rho[2]

                    # (1 - \alpha) m(Q) rho^y_{coot}
                    new_rho2 += (1 - alpha) * feat_mass * rho[3]

                # updates P to minimize UAGW loss w.r.t. constant P', Q
                duals_samp, pi_samp = gw_sinkhorn(
                    full_uot_cost,
                    duals_samp,
                    sample_log_measures,
                    (new_eps, new_rho1, new_rho2),
                )

                pi_samp = (pi_samp_prime.sum() / pi_samp.sum()).sqrt() * pi_samp

                # (Dx.T**2 @ P_{#1} + Dy.T**2 @ P_{#2} - 2Dx.T @ P @ Dy) +
                # \rho^x_{gw} \langle \log{\frac{P_{#1}}{\mu_{sx}}, P_{#1} \rangle +
                # \rho^y_{gw} \langle \log{\frac{P_{#2}}{\mu_{sy}}, P_{#2} \rangle +
                # \epsilon_{gw} \langle \log{\frac{P}{\mu_{sx} \otimes \mu_{sy}}}, P \rangle
                uot_gw = gw_local(
                    pi_samp, samp_lc_gw, sample_measures, sample_supervision
                )

                # \alpha [GW cost]
                full_uot_cost = alpha * uot_gw

                # \alpha m(P) \epsilon_{gw} for joint
                # \alpha \epsilon_{gw} for indep.
                new_eps = (
                    alpha * pi_samp.sum() * eps[0]
                    if entropic_mode == "joint"
                    else alpha * eps[0]
                )

                # \alpha m(P) \rho^x_{gw}
                new_rho1 = alpha * pi_samp.sum() * rho[0]

                # \alpha m(P) \rho^y_{gw}
                new_rho2 = alpha * pi_samp.sum() * rho[1]

                if not (eps_mode == "true"):
                    new_eps /= alpha
                    new_rho1 /= alpha
                    new_rho2 /= alpha

                # find P' to minimize UAGW loss w.r.t. constant P, Q
                duals_samp_prime, pi_samp_prime = gw_sinkhorn(
                    full_uot_cost,
                    duals_samp_prime,
                    sample_log_measures,
                    (new_eps, new_rho1, new_rho2),
                )
                pi_samp_prime = (
                    pi_samp.sum() / pi_samp_prime.sum()
                ).sqrt() * pi_samp_prime

                # early stopping for GW loop
                err = (pi_samp - pi_samp_prev).abs().sum().item()
                if err < tol_gw:
                    break

                if nits_gw > 1 and verbose:
                    print(f"GW Iteration {j + 1}", end="\x1b[1K\r")

        if alpha < 1:
            pi_samp_prev = pi_samp.detach().clone()

            # (X**2 @ Q_{#1} + Y**2 @ Q_{#2} - 2X @ Q @ Y.T) +
            # \rho^x_{coot} \langle \log{\frac{Q_{#1}}{\mu_{fx}}, Q_{#1} \rangle +
            # \rho^y_{coot} \langle \log{\frac{Q_{#2}}{\mu_{fy}}, Q_{#2} \rangle +
            # \epsilon_{coot} \langle \log{\frac{Q}{\mu_{fx} \otimes \mu_{fy}}}, Q \rangle
            uot_coot = coot_local(
                pi_feat, feat_lc_coot, feature_measures, sample_supervision
            )

            # (Dx**2 @ P'_{#1} + Dy**2 @ P'_{#2} - 2Dx @ P' @ Dy.T) +
            # \rho^x_{gw} \langle \log{\frac{P'_{#1}}{\mu_{sx}}, P'_{#1} \rangle +
            # \rho^y_{gw} \langle \log{\frac{P'_{#2}}{\mu_{sy}}, P'_{#2} \rangle +
            # \epsilon_{gw} \langle \log{\frac{P'}{\mu_{sx} \otimes \mu_{sy}}}, P' \rangle
            uot_gw = (
                0
                if alpha == 0
                else gw_local(
                    pi_samp_prime, samp_prime_lc_gw, sample_measures, sample_supervision
                )
            )

            # \alpha [GW cost] + (1 - \alpha) [COOT cost]
            full_uot_cost = alpha * uot_gw + (1 - alpha) * uot_coot

            # for joint reg., we find
            # \alpha m(P') \epsilon_{gw} + (1 - \alpha) m(Q) \epsilon_{coot}
            # for indep. reg., we find
            # \alpha \epsilon_{gw} + (1 - \alpha) \epsilon_{coot}
            new_eps = (
                alpha * pi_samp_prime.sum() * eps[0]
                + (1 - alpha) * pi_feat.sum() * eps[1]
                if entropic_mode == "joint"
                else eps[0]
            )

            # (1 - \alpha) m(Q) rho^x_{coot}
            new_rho1 = (1 - alpha) * pi_feat.sum() * rho[2]

            # (1 - \alpha) m(Q) rho^y_{coot}
            new_rho2 = (1 - alpha) * pi_feat.sum() * rho[3]

            if not (alpha == 0):
                # \alpha m(P') rho^x_{gw}
                new_rho1 += alpha * pi_samp_prime.sum() * rho[0]

                # \alpha m(P') rho^y_{gw}
                new_rho2 += alpha * pi_samp_prime.sum() * rho[1]

            # solve for P w.r.t. constant P', Q
            duals_samp, pi_samp = gw_sinkhorn(
                full_uot_cost,
                duals_samp,
                sample_log_measures,
                (new_eps, new_rho1, new_rho2),
            )

            pi_samp = (pi_feat.sum() / pi_samp.sum()).sqrt() * pi_samp

            # (X.T**2 @ P_{#1} + Y.T**2 @ P_{#2} - 2X.T @ P @ Y) +
            # \rho^x_{coot} \langle \log{\frac{P_{#1}}{\mu_{sx}}, P_{#1} \rangle +
            # \rho^y_{coot} \langle \log{\frac{P_{#2}}{\mu_{sy}}, P_{#2} \rangle +
            # \epsilon_{coot} \langle \log{\frac{P}{\mu_{sx} \otimes \mu_{sy}}}, P \rangle
            uot_coot = coot_local(
                pi_samp, samp_lc_coot, sample_measures, feature_supervision
            )

            # (1 - alpha) [COOT cost]
            full_uot_cost = (1 - alpha) * uot_coot

            # for joint reg., we find
            # (1 - \alpha) m(P) \epsilon_{coot}
            # for indep. reg., we find
            # (1 - \alpha) \epsilon_{coot}
            new_eps = (
                (1 - alpha) * pi_samp.sum() * eps[1]
                if entropic_mode == "joint"
                else (1 - alpha) * eps[1]
            )

            # (1 - alpha) rho^x_{coot}
            new_rho1 = (1 - alpha) * pi_samp.sum() * rho[2]

            # (1 - alpha) rho^y_{coot}
            new_rho2 = (1 - alpha) * pi_samp.sum() * rho[3]

            if not(eps_mode == "true"):
                new_eps /= 1 - alpha
                new_rho1 /= 1 - alpha
                new_rho2 /= 1 - alpha

            duals_feat, pi_feat = coot_sinkhorn(
                full_uot_cost,
                duals_feat,
                feature_log_measures,
                (new_eps, new_rho1, new_rho2),
            )

            pi_feat = (pi_samp.sum() / pi_feat.sum()).sqrt() * pi_feat

        if i % eval_bcd == 0:
            # Update error
            err = (pi_samp - pi_samp_prev).abs().sum().item()

            # log and print global cost
            cost, ent_cost = total_cost((pi_samp, pi_samp_prime, pi_feat))
            log_cost.append(cost)
            log_ent_cost.append(ent_cost)

            if (
                err < tol_bcd
                or abs(log_ent_cost[-2] - log_ent_cost[-1]) < early_stopping_tol
            ):
                if verbose:
                    print(f"BCD Iteration {i+1} - Loss: {cost:.6f}, {ent_cost:.6f}")
                break

            if verbose:
                print(f"BCD Iteration {i+1} - Loss: {cost:.6f}, {ent_cost:.6f}")
    if log:
        return (pi_samp, pi_samp_prime, pi_feat), log_cost[:], log_ent_cost[1:]
    return pi_samp, pi_samp_prime, pi_feat
