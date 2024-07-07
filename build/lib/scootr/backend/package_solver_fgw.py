from fugw.solvers.dense import FUGWSolver
import torch
import ot

def approx_kl(p, q):
    """
    Calculate p * log (p/q). By convention: 0 log 0 = 0
    """

    return torch.nan_to_num(p * p.log(), nan=0.0, posinf=0.0, neginf=0.0).sum() - (p * q.log()).sum()


def kl(p, q):
    """
    Calculate KL divergence in the most general case:
    KL = p * log (p/q) - mass(p) + mass(q)
    """

    return approx_kl(p, q) - p.sum() + q.sum()


def quad_kl(mu, nu, alpha, beta):
    """
    Calculate the KL divergence between two product measures:
    KL(mu \otimes nu, alpha \otimes beta) =
    m_mu * KL(nu, beta) + m_nu * KL(mu, alpha) + (m_mu - m_alpha) * (m_nu - m_beta)

    Parameters
    ----------
    mu: vector or matrix
    nu: vector or matrix
    alpha: vector or matrix with the same size as mu
    beta: vector or matrix with the same size as nu

    Returns
    ----------
    KL divergence between two product measures
    """

    m_mu = mu.sum()
    m_nu = nu.sum()
    m_alpha = alpha.sum()
    m_beta = beta.sum()
    const = (m_mu - m_alpha) * (m_nu - m_beta)

    return m_nu * kl(mu, alpha) + m_mu * kl(nu, beta) + const

def uot_ent(cost, init_duals, tuple_log_p, params, n_iters, tol, eval_freq):
    """
    Solve entropic UOT using Sinkhorn algorithm.
    Allow rho1 and/or rho2 to be infinity but epsilon must be strictly positive.
    """

    rho1, rho2, eps = params
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

def get_local_cost(data, pi, tuple_p, hyperparams, entropic_mode):
    """
    Calculate cost of the UOT.
    cost = (X**2 * P_#1 + Y**2 * P_#2 - 2 * X * P * Y.T) +
            rho1 * approx_kl(P_#1 | a) + rho2 * approx_kl(P_#2 | b) +
            eps * approx_kl(P | a \otimes b)
    """

    rho, eps = hyperparams
    rho1, rho2, _, _ = rho
    a, b, ab = tuple_p
    X_sqr, Y_sqr, X, Y = data

    pi1, pi2 = pi.sum(1), pi.sum(0)
    A = X_sqr @ pi1
    B = Y_sqr @ pi2
    cost = A[:, None] + B[None, :] - 2 * X @ pi @ Y.T

    if rho1 != float("inf") and rho1 != 0:
        cost = cost + rho1 * approx_kl(pi1, a)
    if rho2 != float("inf") and rho2 != 0:
        cost = cost + rho2 * approx_kl(pi2, b)
    if entropic_mode == "joint":
        cost = cost + eps[0] * approx_kl(pi, ab)

    return cost

def solver(
    X,
    Y,
    Dx=None,
    Dy=None,
    px=(None, None),
    py=(None, None),
    rho=(float("inf"), float("inf"), float("inf"), float("inf")),
    uot_mode="entropic",
    fugw_mode = "entropic",
    eps=(1e-2, 1e-2),
    entropic_mode="independent",
    alpha=1,
    init_pi=(None, None),
    init_duals=(None, None),
    log=False,
    verbose=False,
    early_stopping_tol=1e-6,
    eval_bcd=1,
    eval_uot=1,
    tol_bcd=1e-7,
    nits_bcd=100,
    tol_uot=1e-7,
    nits_uot=int(1e5)
):
    """
    Parameters
    ----------
    X: matrix of size nx x dx. First input data.
    Y: matrix of size ny x dy. Second input data.
    D: matrix of size nx x ny. Sample matrix, in case of fused GW
    px: tuple of 2 vectors of length (nx, dx). Measures assigned on rows and columns of X.
        Uniform distributions by default.
    py: tuple of 2 vectors of length (ny, dy). Measures assigned on rows and columns of Y.
        Uniform distributions by default.
    rho: tuple of 4 relaxation marginal-relaxation parameters for FUGW and UOT.
    uot_mode: string or tuple of strings. Uot modes for each update of couplings
        uot_mode = "entropic": use Sinkhorn algorithm in each BCD iteration.
        uot_mode = "mm": use maximisation-minimisation algorithm in each BCD iteration.
    eps: scalar or tuple of scalars.
        Regularisation parameters for COOT under joint reg., GW and COOT under indep. reg.
    entropic_mode:
        entropic_mode = "joint": use UGW-like regularisation.
        entropic_mode = "independent": use COOT-like regularisation.
    alpha: scaler or tuple of nonnegative scalars.
        Interpolation parameter for fused UGW w.r.t the sample and feature couplings.
    D: tuple of matrices of size (nx x ny) and (dx x dy). The linear terms in UOT.
        By default, set to None.
    init_pi: tuple of matrices of size nx x ny and dx x dy if not None.
        Initialisation of sample and feature couplings.
    init_duals: tuple of tuple of vectors of size (nx,ny) and (dx, dy) if not None.
        Initialisation of sample and feature dual vectos if using Sinkhorn algorithm.
    log: True if the cost is recorded, False otherwise.
    verbose: if True then print the recorded cost.
    early_stopping_tol: threshold for the early stopping.
    eval_bcd: multiplier of iteration at which the cost is calculated. For example, if eval_bcd = 10, then the
        cost is calculated at iteration 10, 20, 30, etc...
    eval_bcd: multiplier of iteration at which the old and new duals are compared in the Sinkhorn
        algorithm.
    tol_bcd: tolerance of BCD scheme.
    nits_bcd: number of BCD iterations.
    tol_uot: tolerance of Sinkhorn or MM algorithm.
    nits_uot: number of Sinkhorn or MM iterations.

    Returns
    ----------
    pi_samp: matrix of size nx x ny. Sample matrix.
    pi_feat: matrix of size dx x dy. Feature matrix.
    dual_samp: tuple of vectors of size (nx, ny). Pair of dual vectors when using Sinkhorn algorithm
        to estimate the sample coupling. Used in case of faster solver.
        If use MM algorithm then dual_samp = None.
    dual_feat: tuple of vectors of size (dx, dy). Pair of dual vectors when using Sinkhorn algorithm
        to estimate the feature coupling. Used in case of faster solver.
        If use MM algorithm then dual_feat = None.
    log_cost: if log is True, return a list of cost (without taking into account the regularisation term).
    log_ent_cost: if log is True, return a list of entropic cost.
    """

    nx, dx = X.shape
    ny, dy = Y.shape
    device, dtype = X.device, X.dtype

    if (Dx is None or Dy is None) and (X is None or Y is None):
        raise ValueError(
            "Must have at least one pair of tabular data.")
    elif (Dx is None or Dy is None):
        print("Changing alpha to 0, no GW data.")
        alpha = 0
    elif (X is None or Y is None):
        print("Changing alpha to 1, no COOT data.")
        alpha = 1

    # hyper-parameters
    if isinstance(eps, float) or isinstance(eps, int):
        # under indep. regularization, eps[0] used for both sample matrices
        eps = (eps, eps)
    if not isinstance(eps, tuple):
        raise ValueError(
            "Epsilon must be either a scalar or a tuple of scalars.")
    if not(len(eps) == 2):
        raise ValueError(
            "Must be two epsilon parameters (epsilon for GW and epsilon for COOT).")

    if isinstance(rho, float) or isinstance(rho, int):
        rho = (rho, rho, rho, rho)
    if not isinstance(rho, tuple):
        raise ValueError(
            "Rho must be either a scalar or a tuple of scalars.")
    if len(rho) == 2:
        rho = (rho[0], rho[0], rho[1], rho[1])

    if not isinstance(alpha, float) or isinstance(alpha, int):
        raise ValueError(
            "Alpha must be a scalar.")
    
    px_samp, px_feat = px
    py_samp, py_feat = py

    if px_samp is None:
        px_samp = torch.ones(nx).to(device).to(dtype) / nx
    if px_feat is None:
        px_feat = torch.ones(dx).to(device).to(dtype) / dx
    if py_samp is None:
        py_samp = torch.ones(ny).to(device).to(dtype) / ny
    if py_feat is None:
        py_feat = torch.ones(dy).to(device).to(dtype) / dy
    pxy_samp = px_samp[:, None] * py_samp[None, :]
    pxy_feat = px_feat[:, None] * py_feat[None, :]

    tuple_pxy_samp = (px_samp, py_samp, pxy_samp)
    tuple_pxy_feat = (px_feat, py_feat, pxy_feat)
    tuple_log_pxy_samp = (px_samp.log(), py_samp.log(), pxy_samp)
    tuple_log_pxy_feat = (px_feat.log(), py_feat.log(), pxy_feat)

    X_sqr = X ** 2
    Y_sqr = Y ** 2
    feat_lc = (X_sqr, Y_sqr, X, Y)
    samp_lc = (X_sqr.T, Y_sqr.T, X.T, Y.T)

    # initialise coupling and dual vectors
    pi_samp, pi_feat = init_pi
    if pi_samp is None:
        pi_samp = pxy_samp  # size nx x ny
    if pi_feat is None:
        pi_feat = pxy_feat  # size dx x dy

    if "entropic" in uot_mode:

        duals_samp, duals_feat = init_duals
        if uot_mode == "entropic" and duals_samp is None:
            duals_samp = (torch.zeros_like(px_samp),
                          torch.zeros_like(py_samp))  # shape nx, ny
        if uot_mode == "entropic" and duals_feat is None:
            duals_feat = (torch.zeros_like(px_feat),
                          torch.zeros_like(py_feat))  # shape dx, dy

    # initialise log
    log_cost = []
    log_ent_cost = [float("inf")]
    err = tol_bcd + 1e-3


    # set up hyperparams, etc.
    # calculate UOT cost matrices
    # calculate hyperparams given regularization mode

    for i in range(10):
        # (X^2 * Q_#1 + Y^2 * Q_#2 - 2 * X * Q * Y^T) –> rho=inf for now
        uot_part = get_local_cost(feat_lc, pi_feat, tuple_pxy_feat, hyperparams=(rho, eps[0]), entropic_mode="independent")
        # note Dx, Dy are intra-domain distance matrices for FGW
        pi_samp = ot.gromov.entropic_fused_gromov_wasserstein(M=uot_part, C1=Dx, C2=Dy, alpha=alpha, epsilon=eps[0])

        # (X^T^2 * P_#1 + Y^T^2 * P_#2 - 2 * X^T * P * Y)
        uot_part = (1-alpha)*get_local_cost(samp_lc, pi_samp, tuple_pxy_samp, hyperparams=(rho, eps), entropic_mode="independent")
        duals_feat, pi_feat = uot_ent(uot_part, duals_feat, tuple_log_pxy_feat, (torch.Tensor([rho[2]]), torch.Tensor([rho[3]]), eps[1]), n_iters=nits_uot, tol=tol_uot, eval_freq=eval_uot)

        print(f'Iteration {i}')


    
    # BCD
    # for n iters
        # If GW term, do FUGW here; if no GW term, do UOT (coot case)
        # If COOT term, do UOT here
    return pi_samp, pi_feat