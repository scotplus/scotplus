import torch
# require torch >= 1.9
from functools import partial


def approx_kl(p : torch.Tensor, q : torch.Tensor):
    """
    Computes approximate KL divergence between two measures. In particular, calculates p * log (p/q). By convention, 0 log 0 = 0.

    Parameters
    ----------
    p: measure of dimension n
    q: measure of dimension n

    Returns
    -------
    The approximate KL divergence between p and q
    """

    return torch.nan_to_num(p * p.log(), nan=0.0, posinf=0.0, neginf=0.0).sum() - (p * q.log()).sum()


def kl(p, q):
    """
    Calculates KL divergence in the most general case between two measures. In particular, calculates KL = p * log (p/q) - mass(p) + mass(q).

    Parameters
    ----------
    p: measure of dimension n
    q: measure of dimension n

    Returns
    -------
    The approximate KL divergence between p and q
    """

    return approx_kl(p, q) - p.sum() + q.sum()


def quad_kl(mu, nu, supervision_coef, beta):
    """
    Calculate the KL divergence between two product measures:
    KL(mu \otimes nu, supervision_coef \otimes beta) =
    m_mu * KL(nu, beta) + m_nu * KL(mu, supervision_coef) + (m_mu - m_supervision_coef) * (m_nu - m_beta)

    Parameters
    ----------
    mu: vector or matrix
    nu: vector or matrix
    supervision_coef: vector or matrix with the same size as mu
    beta: vector or matrix with the same size as nu

    Returns
    ----------
    KL divergence between two product measures
    """

    m_mu = mu.sum()
    m_nu = nu.sum()
    m_supervision_coef = supervision_coef.sum()
    m_beta = beta.sum()
    const = (m_mu - m_supervision_coef) * (m_nu - m_beta)

    return m_nu * kl(mu, supervision_coef) + m_mu * kl(nu, beta) + const


def uot_ent(cost, init_duals, tuple_log_p, params, n_iters, tol, eval_freq):
    """
    Solve entropic UOT using Sinkhorn algorithm. Allow rho1 and/or rho2 to be infinity but epsilon must be strictly positive.

    Parameters
    ----------
    cost: UOT cost matrix
    init_duals: initialized duals for Sinkhorn iteration
    tuple_log_p: for the unfrozen coupling matrix, the log of both its initial support measures as well as the product of its initial support measures
    params: specific hyperparameters (marginal relaxation and entropic regularization) for this iteration round
    n_iters: maximum number of Sinkhorn iterations
    tol: tolerance for difference in duals between iterations; if the difference is below tol, return
    eval_freq: the number of iterations before another check against tol is run 

    Returns
    ----------
    The resulting duals and final unfrozen coupling matrix
    """

    # unpacking parameters
    rho1, rho2, eps = params
    log_a, log_b, ab = tuple_log_p
    f, g = init_duals

    tau1 = 1 if torch.isinf(rho1) else rho1 / (rho1 + eps)
    tau2 = 1 if torch.isinf(rho2) else rho2 / (rho2 + eps)

    # Sinkhorn iterations
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

        # convergence check
        if (idx % eval_freq == 0) and (f - f_prev).abs().max().item() < tol:
            break

    # computing pi from duals
    pi = ab * (f[:, None] + g[None, :] - cost / eps).exp()

    return (f, g), pi

# will delete or fix later
def uot_mm(cost, init_pi, tuple_p, params, n_iters, tol, eval_freq):
    """
    Solve (entropic) UOT using the max-min algorithm. Allow epsilon to be 0 but rho1 and rho2 can't be infinity. Note that if the parameters are small so that numerically, the exponential of negative cost will contain zeros and this serves as sparsification of the optimal plan. If the parameters are large, then the resulting optimal plan is more dense than the one obtained from Sinkhorn algo. But the parameters should not be too small, otherwise the kernel will contain too many zeros and consequently, the optimal plan will contain NaN (because the Kronecker sum of two marginals will eventually contain zeros, and divided by zero will result in undesirable result).
    """

    a, b, _ = tuple_p
    rho1, rho2, eps = params
    sum_param = rho1 + rho2 + eps
    tau1, tau2, rho_r = rho1 / sum_param, rho2 / sum_param, eps / sum_param
    K = a[:, None]**(tau1 + rho_r) * b[None, :]**(tau2 +
                                                  rho_r) * (- cost / sum_param).exp()

    m1, m2, pi = init_pi.sum(1), init_pi.sum(0), init_pi

    for idx in range(n_iters):
        m1_old, m2_old = m1.detach().clone(), m2.detach().clone()
        pi = pi**(tau1 + tau2) / (m1[:, None]**tau1 * m2[None, :]**tau2) * K
        m1, m2 = pi.sum(1), pi.sum(0)
        if (idx % eval_freq == 0) and \
                max((m1 - m1_old).abs().max(), (m2 - m2_old).abs().max()) < tol:
            break

    return None, pi


def get_local_cost(data, pi, tuple_p, hyperparams, entropic_mode):
    """
    Calculate the cost matrix to be used in a given sub-UOT problem, without the fast tensor multiplication trick outlined below in quick_sql_tensor_mult_mats.

    Parameters
    ----------
    data: a tuple including the element-wise squares of X and Y, as well as X, Y, and the supervision matrix
    pi: the frozen coupling matrix used in the calculation of the sub-UOT cost matrix
    tuple_p: the individual and product support measures of the frozen coupling matrix
    hyperparams: hyperparameters (marginal relaxation and entropic regularization) relevant to computing the local cost matrix for the unfrozen coupling matrix
    entropic_mode: a flag indicating whether entropic regularization should be done using the frozen coupling matrix and its support product or independently on each marginal of the frozen matrix

    Returns
    ----------

    A cost matrix that identifies the cost of matching a row index i with a column index j for the unfrozen coupling matrix (NOT pi in this function), for use in a following OT optimization step within a BCD iteration.
    """

    # unpacking parameters
    rho, eps = hyperparams
    rho1, rho2, _, _, _, _ = rho
    a, b, ab = tuple_p
    X_sqr, Y_sqr, X, Y, D, supervision_coef = data
    pi1, pi2 = pi.sum(1), pi.sum(0)
    A = X_sqr @ pi1
    B = Y_sqr @ pi2

    # computing cost matrix of frozen UOT problem
    cost = A[:, None] + B[None, :] - 2 * X @ pi @ Y.T + supervision_coef * D

    if rho1 != float("inf") and rho1 != 0:
        cost = cost + rho1 * approx_kl(pi1, a)
    if rho2 != float("inf") and rho2 != 0:
        cost = cost + rho2 * approx_kl(pi2, b)
    if entropic_mode == "joint":
        cost = cost + eps[0] * approx_kl(pi, ab)

    return cost

def get_local_cost_fast(same_pi, opp_pi, data, tuple_p, tensmult, alpha, hyperparams, entropic_mode):
    """
    Calculates cost of AGW, COOT, and GW without marginal relaxation.
    cost = alpha * (Dx - Dy)^2 \otimes P + (1 - alpha) ((X - y)^2 \otimes P_opp) + rho1 * approx_kl(P_#1 | a) + rho2 * approx_kl(P_#2 | b) + eps * approx_kl(P | a \otimes b).

    Balanced parallel to get_local_cost.

    Parameters
    ----------

    same_pi: the coupling matrix that will soon undergo optimization via minimization of the inner product with the cost matrix returned by this function
    opp_pi: the other coupling matrix, which is frozen for the current step within a BCD iteration
    data: includes the relevant D and supervision coefficient for supervising same_pi
    tuple_p: the desired measures and product measure associated with same_pi
    tensmult: a dictionary containing constants used for faster calculation of the tensor products in the first part of the AGW cost described above
    alpha: the hyperparameter which trades off GW and COOT distance in the AGW cost above
    hyperparams: rho and epsilon (marginal relaxation and entropic regularization), as explained in our solver below
    entropic_mode: a flag for whether pi_samp and pi_feat are jointly entropically regularized or not

    Returns
    -------

    A cost matrix that identifies the cost of matching a row index i with a column index j for same_pi, for use in a following OT optimization step within a BCD iteration.
    """

    # unpacking parameters
    _, eps = hyperparams
    if not(tuple_p == None):
        a, b, ab = tuple_p
    _, _, _, _, D, supervision_coef = data

    # computing composite cost matrix
    if alpha == 0:
        cost = (tensmult['const']['coot'] - tensmult['h']['coot1'] @ opp_pi @ tensmult['h']['coot2'].T)
    elif alpha == 1:
        cost = (tensmult['const']['gw'] - tensmult['h']['gw1'] @ same_pi @ tensmult['h']['gw2'].T)
    else:
        cost = alpha * (tensmult['const']['gw'] - tensmult['h']['gw1'] @ same_pi @ tensmult['h']['gw2'].T) + (1 - alpha) * ((tensmult['const']['coot'] - tensmult['h']['coot1'] @ opp_pi @ tensmult['h']['coot2'].T))

    # additional terms in frozen UOT cost matrix
    if (supervision_coef > 0):
        cost += supervision_coef * D

    if entropic_mode == "joint":
        cost = cost + eps[0] * approx_kl(opp_pi, ab)

    return cost


def get_cost(pi_samp, pi_feat, data, data_T, tuple_pxy_samp, tuple_pxy_feat, hyperparams, entropic_mode):
    """
    Calculates complete cost of the UGW/UCOOT objective functions. We specifically use this function for computing unbalanced costs.

    Parameters
    ----------

    pi_samp: sample coupling matrix
    pi_feat: feature coupling matrix
    data: includes X, Y, their element-wise squares, the sample supervision matrix, and the sample supervision coefficient
    data_T: includes the feature supervision matrix and the feature supervision coefficient
    tuple_pxy_samp: the desired measures and product measure associated with the sample coupling matrix
    tuple_pxy_feat: the desired measures and product measure associated with the feature coupling matrix
    hyperparams: rho and epsilon (marginal relaxation and entropic regularization), as explained in our solver below
    entropic_mode: a flag for whether pi_samp and pi_feat are jointly entropically regularized or not

    Returns
    -------

    The total current optimal transport cost in terms of the fused OT objective function, with and without entropic regularization costs. Only used for displaying the cost-so-far during optimization. 
    """

    rho, eps = hyperparams
    eps_samp, eps_feat = eps
    rho1, rho2, rho1_samp, rho2_samp, rho1_feat, rho2_feat = rho
    px_samp, py_samp, pxy_samp = tuple_pxy_samp
    px_feat, py_feat, pxy_feat = tuple_pxy_feat
    X_sqr, Y_sqr, X, Y, D_samp, supervision_coef_samp = data
    _, _, _, _, D_feat, supervision_coef_feat = data_T

    pi1_samp, pi2_samp = pi_samp.sum(1), pi_samp.sum(0)
    pi1_feat, pi2_feat = pi_feat.sum(1), pi_feat.sum(0)

    # UGW part
    A_sqr = (X_sqr @ pi1_feat).dot(pi1_samp)
    B_sqr = (Y_sqr @ pi2_feat).dot(pi2_samp)
    AB = (X @ pi_feat @ Y.T) * pi_samp
    cost = A_sqr + B_sqr - 2 * AB.sum()

    if rho1 != float("inf") and rho1 != 0:
        cost = cost + rho1 * quad_kl(pi1_samp, pi1_feat, px_samp, px_feat)
    if rho2 != float("inf") and rho2 != 0:
        cost = cost + rho2 * quad_kl(pi2_samp, pi2_feat, py_samp, py_feat)

    # UOT part
    if supervision_coef_samp != 0:
        uot_cost_samp = (D_samp * pi_samp).sum()
        if rho1_samp != float("inf") and rho1_samp != 0:
            uot_cost_samp = uot_cost_samp + rho1_samp * kl(pi1_samp, px_samp)
        if rho2_samp != float("inf") and rho2_samp != 0:
            uot_cost_samp = uot_cost_samp + rho2_samp * kl(pi2_samp, py_samp)

        cost = cost + supervision_coef_samp * uot_cost_samp

    if supervision_coef_feat != 0:
        uot_cost_feat = (D_feat * pi_feat).sum()
        if rho1_feat != float("inf") and rho1_feat != 0:
            uot_cost_feat = uot_cost_feat + rho1_feat * kl(pi1_feat, px_feat)
        if rho2_feat != float("inf") and rho2_feat != 0:
            uot_cost_feat = uot_cost_feat + rho2_feat * kl(pi2_feat, py_feat)

        cost = cost + supervision_coef_feat * uot_cost_feat

    # Entropic part
    ent_cost = cost
    if entropic_mode == "joint" and eps_samp != 0:
        ent_cost = ent_cost + eps_samp * \
            quad_kl(pi_samp, pi_feat, pxy_samp, pxy_feat)
    elif entropic_mode == "independent":
        if eps_samp != 0:
            ent_cost = ent_cost + eps_samp * kl(pi_samp, pxy_samp)
        if eps_feat != 0:
            ent_cost = ent_cost + eps_feat * kl(pi_feat, pxy_feat)

    return cost.item(), ent_cost.item()

def get_cost_fast(pi_samp, pi_feat, data, data_T, tuple_pxy_samp, tuple_pxy_feat, hyperparams, entropic_mode, tensmult, alpha):
    """
    Calculates complete cost of the GW/COOT/AGW objective functions. We specifically use this function for computing balanced costs.

    Parameters
    ----------

    pi_samp: sample coupling matrix
    pi_feat: feature coupling matrix
    data: includes X, Y, their element-wise squares, the sample supervision matrix, and the sample supervision coefficient
    data_T: includes the feature supervision matrix and the feature supervision coefficient
    tuple_pxy_samp: the desired measures and product measure associated with the sample coupling matrix
    tuple_pxy_feat: the desired measures and product measure associated with the feature coupling matrix
    hyperparams: rho and epsilon (marginal relaxation and entropic regularization), as explained in our solver below
    entropic_mode: a flag for whether pi_samp and pi_feat are jointly entropically regularized or not
    tensmult: a dictionary containing constants used for faster calculation of the tensor products in the first part of the AGW cost described above
    alpha: the hyperparameter which trades off GW and COOT distance in AGW cost

    Returns
    -------

    The total current optimal transport cost in terms of the fused OT objective function, with and without entropic regularization costs. Only used for displaying the cost-so-far during optimization. 
    """

    # unpacking parameters
    rho, eps = hyperparams
    eps_samp, eps_feat = eps
    rho1, rho2, rho1_samp, rho2_samp, rho1_feat, rho2_feat = rho
    if not(tuple_pxy_samp == None):
        px_samp, py_samp, pxy_samp = tuple_pxy_samp
    if not(tuple_pxy_feat == None):
        px_feat, py_feat, pxy_feat = tuple_pxy_feat
    tensmult_samp = tensmult
    alpha = alpha
    _, _, _, _, D_samp, supervision_samp = data
    _, _, _, _, D_feat, supervision_feat = data_T

    pi1_samp, pi2_samp = None, None
    pi1_feat, pi2_feat = None, None

    cost = 0
    # independent GW cost
    if (alpha == 1):
        cost_mat = (tensmult_samp['const']['gw'] - tensmult_samp['h']['gw1'] @ pi_samp @ tensmult_samp['h']['gw2'].T)
        cost += sum(torch.einsum('ij, ij -> i', cost_mat, pi_samp))
        pi1_samp, pi2_samp = pi_samp.sum(1), pi_samp.sum(0)
        kl_pairs = ((pi1_samp, px_samp), (pi2_samp, py_samp))
        rho_main = (rho1, rho2)

        if rho_main[0] != float("inf") and rho_main[0] != 0:
            cost = cost + rho_main[0] * quad_kl(kl_pairs[0][0], kl_pairs[1][0], kl_pairs[0][1], kl_pairs[1][1])

    # cost for non-independent GW (at least some COOT in the balance)
    else:
        if alpha == 0:
            cost_mat = (tensmult_samp['const']['coot'] - tensmult_samp['h']['coot1'] @ pi_feat @ tensmult_samp['h']['coot2'].T)
        else:
            cost_mat = alpha * (tensmult_samp['const']['gw'] - tensmult_samp['h']['gw1'] @ pi_samp @ tensmult_samp['h']['gw2'].T) + (1 - alpha) * ((tensmult_samp['const']['coot'] - tensmult_samp['h']['coot1'] @ pi_feat @ tensmult_samp['h']['coot2'].T))
        cost += sum(torch.einsum('ij, ij -> i', cost_mat, pi_samp))
        pi1_samp, pi2_samp = pi_samp.sum(1), pi_samp.sum(0)
        pi1_feat, pi2_feat = pi_feat.sum(1), pi_feat.sum(0)
        if rho1 != float("inf") and rho1 != 0:
                cost = cost + rho1 * quad_kl(pi1_samp, pi2_samp, px_samp, py_samp)
        if rho2 != float("inf") and rho2 != 0:
                cost = cost + rho2 * quad_kl(pi1_feat, pi2_feat, px_feat, py_feat)

    if supervision_samp != 0:
        uot_cost_samp = (D_samp * pi_samp).sum()
        if rho1_samp != float("inf") and rho1_samp != 0:
            uot_cost_samp = uot_cost_samp + rho1_samp * kl(pi1_samp, px_samp)
        if rho2_samp != float("inf") and rho2_samp != 0:
            uot_cost_samp = uot_cost_samp + rho2_samp * kl(pi2_samp, py_samp)

        cost = cost + supervision_samp * uot_cost_samp

    if supervision_feat != 0 and alpha < 1:
        uot_cost_feat = (D_feat * pi_feat).sum()
        if rho1_feat != float("inf") and rho1_feat != 0:
            uot_cost_feat = uot_cost_feat + rho1_feat * kl(pi1_feat, px_feat)
        if rho2_feat != float("inf") and rho2_feat != 0:
            uot_cost_feat = uot_cost_feat + rho2_feat * kl(pi2_feat, py_feat)

        cost = cost + supervision_feat * uot_cost_feat

    # Entropic cost
    ent_cost = cost
    if entropic_mode == "joint" and eps_samp != 0 and alpha != 1:
        ent_cost = ent_cost + eps_samp * quad_kl(pi_samp, pi_feat, pxy_samp, pxy_feat)
    elif entropic_mode == "independent":
        if eps_samp != 0:
            ent_cost = ent_cost + eps_samp * kl(pi_samp, pxy_samp)
        if eps_feat != 0 and alpha != 1:
            ent_cost = ent_cost + eps_feat * kl(pi_feat, pxy_feat)

    return cost.item(), ent_cost.item()

def quick_sql_tensor_mult_mats(X1, X2, v1, v2, device, dtype):
    """
    Sets up quick tensor multiplication matrices, as per Peyre et al. 2016, to solve tensor multiplication of the form (X1 - X2)^2 \otimes pi (for getting OT cost matrices). Note that if we chose to no longer use l2 penalty (square loss, or sql as in the function name) for the development of our cost tensor, this trick would no longer apply and we would lose the runtime bonus.

    Parameters
    ----------
    X1: the first matrix in the in the cost expression above
    X2: the second matrix in the expression above
    v1: the desired support measure of pi, on the row axis
    v2: the desired support measure of pi, on the column axis
    device, dtype: torch parameters, given user setup

    Returns
    -------
    Matrices necessary to do fast tensor multiplication in local and total cost calculations.

    """

    # functions decomposed to relate to Peyre et al. 2016
    def f1(a):
        return (a ** 2)
    
    def f2(b):
        return (b ** 2)
    
    def h1(a):
        return a
    
    def h2(b):
        return 2 * b
    
    constC1 = (f1(X1) @ v1.reshape(-1, 1) @ torch.ones(f1(X2).shape[0]).to(device).to(dtype).reshape(1, -1))
    constC2 = (torch.ones(f1(X1).shape[0]).to(device).to(dtype).reshape(-1, 1) @ v2.reshape(1, -1) @ f2(X2).T)
    
    constC = constC1 + constC2
    hX1 = h1(X1)
    hX2 = h2(X2)
    
    return constC, hX1, hX2


def final_solver(
    X = None,
    Y = None,
    Dx = None,
    Dy = None,
    eps=(1e-2, 1e-2),
    rho=(float("inf"), float("inf"), 0, 0, 0, 0),
    alpha=0,
    supervision_coef=(1, 1),
    D=(None, None),
    px=(None, None),
    py=(None, None),
    uot_mode=("entropic", "entropic"),
    entropic_mode="joint",
    init_pi=(None, None),
    init_duals=(None, None),
    log=False,
    verbose=False,
    early_stopping_tol=1e-6,
    eval_bcd=10,
    eval_uot=1,
    tol_bcd=1e-7,
    nits_bcd=100,
    tol_uot=1e-7,
    nits_uot=500
):
    """
    A generalized solver for computing optimal AGW sample and feature coupling matrices. See our theory document for a better understanding of what this really means.

    Parameters
    ----------
    X: matrix of size nx x dx. First input data.
    Y: matrix of size ny x dy. Second input data.
    Dx: matrix of size nx x nx. Similarity matrix on first input data samples.
    Dy: matrix of size ny x ny. Similarity matrix on second input data samples.
    eps: scalar or tuple of scalars, used for entropic regularization of each coupling matrix.
    rho: tuple of six marginal relaxation coefficients for each coupling matrix:
        rho[0]: marginal relaxation of X sample/feature supports
        rho[1]: marginal relaxation of Y sample/feature supports
        rho[2]: independent marginal relaxation of X sample support, for supervision
        rho[3]: independent marginal relaxation of Y sample support, for supervision
        rho[4]: independent marginal relaxation of X feature support, for supervision
        rho[5]: independent marginal relaxation of Y feature support, for supervision
    alpha: AGW hyperparamter (range: 0-1) that trades UCOOT cost with UGW cost in each inner step within a BCD iteration. The higher alpha is, the more UGW is considered.
    supervision_coef: a tuple of scalars indicating how much to consider supervision on sample and feature coupling matrices.
    D: a tuple of matrices indicating supervision preferences for samples and features, respectively. The lower a value is in a given cell ij in D, the more mass will be moved from i to j in pi.
    px: desired sample and feature support measures for X
    py: desired sample and feature support measures for Y
    uot_mode: a string or tuple of strings, determining the algorithm used to solve OT optimizations within each BCD iteration. If a tuple is provided, the first value will be applied to sample optimization, and the second will be applied to feature optimization.
        uot_mode = "entropic": use Sinkhorn algorithm in each BCD iteration.
        uot_mode = "mm": use maximisation-minimisation algorithm in each BCD iteration.
        cg/emd to come?
    entropic_mode: a flag indicating whether entropic regularization will be joint or independent. If joint, entropic regularization will be applied to pi_samp and pi_feat simultaneously during optimization. If independent, it will be applied independently.
    init_pi: an initialization option for pi_samp and pi_feat.
    init_duals: an initialization option for duals, which will be used during Sinkhorn optimization. Only relevant if uot_mode='entropic'.
    log: a flag for whether a log of AGW and entropic AGW cost will be returned
    verbose: a flag for whether a cost will be printed every "eval_bcd"th BCD iteration.
    early_stopping_tol: early stopping tolerance, in terms of AGW cost change iteration to iteration.
    eval_bcd: how often (in terms of BCD iterations) the AGW cost and pi_samp convergence will be computed.
    eval_uot: how often (in terms of algorithm iterations) the pi convergence will be computed during BCD's inner OT optimization problems.
    tol_bcd: if the absolute sum of differences between pi_samp at iteration i - 1 and pi_samp at iteration i is less than tol_bcd, the process will exit and return the current coupling matrices.
    nits_bcd: the maximum number of BCD iterations before this function returns.
    tol_uot: if the absolute sum or maximum (OT algorithm dependent) difference of elements in a selected optimization byproduct is less than tol_uot, the OT algorithm in use will return.
    nits_uot: the maximum number of iterations for the OT algorithms used within a BCD iterations. A good value for this parameter varies greatly by algorithm, so we recommend modifying tol_uot to exit earlier, rather than nits_uot. 

    Returns
    -------
    pi_samp: matrix of size nx x ny. Sample coupling matrix.
    pi_feat: matrix of size dx x dy. Feature coupling matrix (or, another sample coupling matrix for UGW/GW).
    dual_samp: tuple of vectors of size (nx, ny). Pair of dual vectors when using Sinkhorn algorithm.
        to estimate the sample coupling. Used in case of faster solver.
        If use MM algorithm then dual_samp = None.
    dual_feat: tuple of vectors of size (dx, dy). Pair of dual vectors when using Sinkhorn algorithm.
        to estimate the feature coupling. Used in case of faster solver.
        If use MM algorithm then dual_feat = None.
    log_cost: if log is True, return a list of cost (without taking into account the regularisation term).
    log_ent_cost: if log is True, return a list of entropic cost.
    """

    # parsing hyperparameters
    if isinstance(eps, float) or isinstance(eps, int):
        eps = (eps, eps)
    if not isinstance(eps, tuple):
        raise ValueError(
            "Epsilon must be either a scalar or a tuple of scalars.")
    
    if entropic_mode == "joint":
        eps = (eps[0], eps[0])

    if isinstance(supervision_coef, float) or isinstance(supervision_coef, int):
        supervision_coef = (supervision_coef, supervision_coef)
    if not isinstance(supervision_coef, tuple):
        raise ValueError(
            "supervision_coef must be either a scalar or a tuple of scalars.")

    if isinstance(uot_mode, str):
        uot_mode = (uot_mode, uot_mode)
    if not isinstance(uot_mode, tuple):
        raise ValueError(
            "uot_mode must be either a string or a tuple of strings.")
    
    if not(isinstance(alpha, float) or isinstance(alpha, int)) or (0 > alpha or alpha > 1):
        raise ValueError(
            "alpha must be a scalar between 0 and 1")
    
    # depending on present matrices, populate necessary parameters such as data shapes, forced alphas (i.e. constrained to GW or COOT by absence of X/Y or Dx/Dy)
    exists_feat = True
    if not(X is None) and not(Y is None):
        nx, dx = X.shape
        ny, dy = Y.shape
        device, dtype = X.device, X.dtype
        if (Dx is None) or (Dy is None):
            alpha = 0
        else:
            if not(Dx.shape[0] == Dx.shape[1]) or not(Dx.shape[0] == nx):
                raise ValueError(
                "Dx matrix (similarity matrix on X samples) must be squared with nx rows and cols.")
            if not(Dy.shape[0] == Dy.shape[1]) or not(Dy.shape[0] == ny):
                raise ValueError(
                "Dy matrix (similarity matrix on y samples) must be squared with ny rows and cols.")
            if alpha == 1:
                X = Dx
                Y = Dy
                exists_feat = False
    elif not(Dx is None) and not(Dy is None):
        if not(Dx.shape[0] == Dx.shape[1]):
            raise ValueError(
            "Dx matrix (similarity matrix on X samples) must be squared.")
        if not(Dy.shape[0] == Dy.shape[1]):
            raise ValueError(
            "Dy matrix (similarity matrix on y samples) must be squared.")
        nx, dx = Dx.shape[0], Dx.shape[0]
        ny, dy = Dy.shape[0], Dy.shape[0]
        device, dtype = Dx.device, Dx.dtype
        alpha = 1
        X = Dx
        Y = Dy
        exists_feat = False

    # unpacking parsed hyperparameters
    rho1, rho2, rho1_samp, rho2_samp, rho1_feat, rho2_feat = rho
    eps_samp, eps_feat = eps
    uot_mode_samp, uot_mode_feat = uot_mode

    # verifying necessary epsilon/rho relationships
    if eps_samp == 0 and torch.isinf(torch.Tensor([rho1, rho2, rho1_samp, rho2_samp])).sum() > 0:
        raise ValueError("Invalid values for epsilon and rho of sample coupling. \
                        Cannot contain zero in epsilon AND infinity in rho at the same time.")
    else:
        if eps_samp == 0:
            uot_mode_samp = "mm"
        if torch.isinf(torch.Tensor([rho1, rho2, rho1_samp, rho2_samp])).sum() > 0:
            uot_mode_samp = "entropic"

    if eps_feat == 0 and torch.isinf(torch.Tensor([rho1, rho2, rho1_feat, rho2_feat])).sum() > 0:
        raise ValueError("Invalid values for epsilon and rho of feature coupling. \
                        Cannot contain zero in epsilon AND infinity in rho at the same time.")
    else:
        if eps_feat == 0:
            uot_mode_feat = "mm"
        if torch.isinf(torch.Tensor([rho1, rho2, rho1_feat, rho2_feat])).sum() > 0:
            uot_mode_feat = "entropic"
    uot_mode = (uot_mode_samp, uot_mode_feat)

    # measures on rows and columns
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

    # product mesaures (Kronecker of independent supports)    
    pxy_samp = px_samp[:, None] * py_samp[None, :]
    pxy_feat = px_feat[:, None] * py_feat[None, :]

    # zipping coupling matrix supports for easy passing to cost calculations
    tuple_pxy_samp = (px_samp, py_samp, pxy_samp)
    tuple_log_pxy_samp = (px_samp.log(), py_samp.log(), pxy_samp)

    tuple_pxy_feat = (px_feat, py_feat, pxy_feat)
    tuple_log_pxy_feat = (px_feat.log(), py_feat.log(), pxy_feat)

    # constant data variables for supervision
    supervision_coef_samp, supervision_coef_feat = supervision_coef
    D_samp, D_feat = D
    if D_samp is None or supervision_coef_samp == 0:
        D_samp, supervision_coef_samp = 0, 0
    if D_feat is None or supervision_coef_feat == 0:
        D_feat, supervision_coef_feat = 0, 0

    # populating constants for fast tensor multiplication (balanced cases)
    tensmult_samp = None
    tensmult_feat = None
    agw = (rho[0] == float('inf') and rho[1] == float('inf'))
    if agw:
        tensmult_samp = {'const' : {}, 'h' : {}}
        tensmult_feat = {'const' : {}, 'h' : {}}

        if not(alpha == 0):
            tensmult_samp['const']['gw'], tensmult_samp['h']['gw1'], tensmult_samp['h']['gw2'] = quick_sql_tensor_mult_mats(Dx, Dy, px_samp, py_samp, device, dtype)
        if exists_feat:
            tensmult_samp['const']['coot'], tensmult_samp['h']['coot1'], tensmult_samp['h']['coot2'] = quick_sql_tensor_mult_mats(X, Y, px_feat, py_feat, device, dtype)
            tensmult_feat['const']['coot'], tensmult_feat['h']['coot1'], tensmult_feat['h']['coot2'] = quick_sql_tensor_mult_mats(X.T, Y.T, px_samp, py_samp, device, dtype)

    # populating constants for unbalanced cost calculations and supervision
    X_sqr = X ** 2
    Y_sqr = Y ** 2
    data = (X_sqr, Y_sqr, X, Y, D_samp, supervision_coef_samp)
    data_T = (X_sqr.T, Y_sqr.T, X.T, Y.T, D_feat, supervision_coef_feat)

    # initializing coupling and dual vectors
    pi_samp, pi_feat = init_pi
    if pi_samp is None:
        pi_samp = pxy_samp  # size nx x ny
    if pi_feat is None:
        pi_feat = pxy_feat  # size dx x dy
    else:
        pi_feat = None

    # simplifying local cost and optimization functions, given information in the broader call
    if "entropic" in uot_mode:
        self_uot_ent = partial(uot_ent, n_iters=nits_uot,
                               tol=tol_uot, eval_freq=eval_uot)

        duals_samp, duals_feat = init_duals
        if uot_mode_samp == "entropic" and duals_samp is None:
            duals_samp = (torch.zeros_like(px_samp),
                          torch.zeros_like(py_samp))  # shape nx, ny
        if uot_mode_feat == "entropic" and duals_feat is None:
            duals_feat = (torch.zeros_like(px_feat),
                          torch.zeros_like(py_feat))  # shape dx, dy

    elif "mm" in uot_mode:
        self_uot_mm = partial(uot_mm, n_iters=nits_uot,
                              tol=tol_uot, eval_freq=eval_uot)

    hyperparams = (rho, eps)
    self_get_local_cost = partial(
        get_local_cost, hyperparams=hyperparams, entropic_mode=entropic_mode)
    self_get_cost = partial(get_cost, data=data, data_T=data_T, tuple_pxy_samp=tuple_pxy_samp,
                            tuple_pxy_feat=tuple_pxy_feat, hyperparams=hyperparams, entropic_mode=entropic_mode)
    if agw:
        self_get_local_cost = partial(
        get_local_cost_fast, hyperparams=hyperparams, entropic_mode=entropic_mode)
        self_get_cost = partial(get_cost_fast, data=data, data_T=data_T, tuple_pxy_samp=tuple_pxy_samp, tuple_pxy_feat=tuple_pxy_feat, hyperparams=hyperparams, entropic_mode=entropic_mode, tensmult=tensmult_samp, alpha=alpha)

    # initializing log
    log_cost = []
    log_ent_cost = [float("inf")]
    err = tol_bcd + 1e-3

    # beginning of BCD (actual optimization)
    for idx in range(nits_bcd):
        pi_samp_prev = pi_samp.detach().clone()

        if exists_feat or not(agw):
        # Update pi_feat (feature coupling)
            mass = pi_samp.sum()
            new_rho1 = rho1 * mass + supervision_coef_feat * rho1_feat
            new_rho2 = rho2 * mass + supervision_coef_feat * rho2_feat
            new_eps = mass * eps_feat if entropic_mode == "joint" else eps_feat
            # use fast vs. slow cost calculation
            if agw:
                uot_cost = self_get_local_cost(pi_feat, pi_samp, data_T, tuple_pxy_samp, tensmult_feat, alpha=0)
            else:
                uot_cost = self_get_local_cost(
                    data_T, pi_samp, tuple_pxy_samp)  # size dx x dy
            uot_params = (new_rho1, new_rho2, new_eps)

            if uot_mode_feat == "entropic":
                duals_feat, pi_feat = self_uot_ent(
                    uot_cost, duals_feat, tuple_log_pxy_feat, uot_params)
            elif uot_mode_feat == "mm":
                duals_feat, pi_feat = self_uot_mm(
                    uot_cost, pi_feat, tuple_pxy_feat, uot_params)
            pi_feat = (mass / pi_feat.sum()).sqrt() * pi_feat  # shape dx x dy

        # Update pi (sample coupling)
        mass = torch.Tensor([1])
        if not(pi_feat == None):
            mass = pi_feat.sum()
        new_rho1 = rho1 * mass + supervision_coef_samp * rho1_samp
        new_rho2 = rho2 * mass + supervision_coef_samp * rho2_samp
        new_eps = mass * eps_samp if entropic_mode == "joint" else eps_samp
        # use fast vs. slow cost calculation
        if agw:
            uot_cost = self_get_local_cost(pi_samp, pi_feat, data, tuple_pxy_feat, tensmult_samp, alpha)
        else:
            uot_cost = self_get_local_cost(data, pi_feat, tuple_pxy_feat)  # size nx x ny
        uot_params = (new_rho1, new_rho2, new_eps)

        if uot_mode_samp == "entropic":
            duals_samp, pi_samp = self_uot_ent(
                uot_cost, duals_samp, tuple_log_pxy_samp, uot_params)
        elif uot_mode_samp == "mm":
            duals_samp, pi_samp = self_uot_mm(
                uot_cost, pi_samp, tuple_pxy_samp, uot_params)
        pi_samp = (mass / pi_samp.sum()).sqrt() * pi_samp  # shape nx x ny

        # check for convergence
        if idx % eval_bcd == 0:
            # Update error
            err = (pi_samp - pi_samp_prev).abs().sum().item()
            cost, ent_cost = self_get_cost(pi_samp, pi_feat)
            log_cost.append(cost)
            log_ent_cost.append(ent_cost)

            if err < tol_bcd or abs(log_ent_cost[-2] - log_ent_cost[-1]) < early_stopping_tol:
                break

            if verbose:
                print("Cost at iteration {}: {}".format(idx+1, cost))

    if pi_samp.isnan().any() or (exists_feat and pi_feat.isnan().any()):
        print("There is NaN in coupling")

    if log:
        return (pi_samp, pi_feat), (duals_samp, duals_feat), log_cost, log_ent_cost[1:]
    else:
        return (pi_samp, pi_feat), (duals_samp, duals_feat)
