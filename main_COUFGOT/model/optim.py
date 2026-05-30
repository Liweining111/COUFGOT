import torch
from torch.nn import KLDivLoss
from torch.autograd import Variable
# require torch >= 1.9
from functools import partial
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from COUFGOT.main_COUFGOT.model.filter import apply_filter,compute_covariance_matrix
import warnings
warnings.filterwarnings('ignore', message='.*not a leaf Tensor.*')
# COUFGOT main program implementation

def approx_kl(p, q):
    r"""
    Calculate p * log (p/q). By convention: 0 log 0 = 0
    """
    p_safe = torch.clamp(p, min=1e-12)  # Avoid log(0)
    return torch.nan_to_num(p_safe * torch.log(p_safe / (q + 1e-12)), nan=0.0, posinf=0.0, neginf=0.0).sum()

def kl(p, q):# Function name and variable name repetition causes overwrite; KL divergence with mass difference correction
    r"""
    Calculate KL divergence in the most general case:
    KL = p * log (p/q) - mass(p) + mass(q)
    """

    return approx_kl(p, q) - p.sum() + q.sum()


def quad_kl(mu, nu, alpha, beta):# μ, ν from one multi-omics dataset; α, β from another multi-omics dataset
    r"""This metric can measure the consistency of distributions between two multi-modal datasets.
    For one dataset, two parameters represent sample distribution and feature distribution respectively,
    plus marginal quality deviation included in the penalty term.
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
    r"""Also in log domain, Sinkhorn: alternately update dual variables (f, g)
    cost: cost matrix (torch tensor, usually local cost), shape (n_x, n_y).
    init_duals: initial duals (f, g), vectors of length nx and ny respectively.
    tuple_log_p: log marginals (log_a, log_b, ab), where ab is a[:,None]*b[None,:].
    params: triple (rho1, rho2, eps), inf=strict constraint, 0=no constraint.
    n_iters, tol, eval_freq: number of iterations, convergence tolerance, evaluation frequency.
    Solve entropic UOT using Sinkhorn algorithm (iterative algorithm solving optimal transport through row-column alternating normalization).
    Allow rho1 and/or rho2 to be infinity but epsilon must be strictly positive.
    """

    rho1, rho2, eps = params
    log_a, log_b, ab = tuple_log_p
    f, g = init_duals
    if f is None or g is None:
        f, g = torch.ones_like(log_a), torch.ones_like(log_b)
    tau1 = rho1 / (rho1 + eps)
    tau2 = rho2 / (rho2 + eps)
    #print('cost:', cost)
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

    pi = ab * (f[:, None] + g[None, :] - cost / eps).exp()# Already inverse log

    return (f, g), pi

def uot_mm(cost, init_pi, tuple_p, params, n_iters, tol, eval_freq):
    r"""Max-Min: directly update transport plan π and marginals (m1, m2)
    cost: cost matrix or local cost (in current implementation cost is not directly exponentiated except when constructing K).
    init_pi: initial transport plan pi.
    tuple_p: marginals a, b, _ (where _ is outer product).
    params: (rho1, rho2, eps).
    eval_freq: evaluation frequency.
    Solve (entropic) UOT using the max-min algorithm.
    Allow epsilon to be 0 but rho1 and rho2 can't be infinity.
    Note that if the parameters are small so that numerically, the exponential of
    negative cost will contain zeros and this serves as sparsification of the optimal plan.
    If the parameters are large, then the resulting optimal plan is more dense than the one
    obtained from Sinkhorn algo.
    But the parameters should not be too small, otherwise the kernel will contain too many zeros
    and consequently, the optimal plan will contain NaN (because the Kronecker sum of two marginals
    will eventually contain zeros, and divided by zero will result in undesirable result).
    """

    a, b, _ = tuple_p
    rho1, rho2, eps = params
    sum_param = rho1 + rho2 + eps
    tau1, tau2, rho_r = rho1 / sum_param, rho2 / sum_param, eps / sum_param  # Parameter normalization
    K = a[:, None] ** (tau1 + rho_r) * b[None, :] ** (tau2 + rho_r) * (- cost / sum_param).exp()

    m1, m2, pi = init_pi.sum(1), init_pi.sum(0), init_pi

    for idx in range(n_iters):
        m1_old, m2_old = m1.detach().clone(), m2.detach().clone()
        pi = pi ** (tau1 + tau2) / (m1[:, None] ** tau1 * m2[None, :] ** tau2) * K
        m1, m2 = pi.sum(1), pi.sum(0)
        if (idx % eval_freq == 0) and \
                max((m1 - m1_old).abs().max(), (m2 - m2_old).abs().max()) < tol:
            break

    return None, pi

def compute_ufgot_distance(gL1, gL2, gl1, gl2, pi_samp, pi_feat, filter_type='identity',device='cpu'):
    r"""
    Calculate UFGOT distance (based on paper formulas 2-9)
    """
    # Calculate trace terms
    trace_term = (torch.trace(gL1 @ gL1) + torch.trace(gL2 @ gL2) +
                  torch.trace(gl1 @ gl1) + torch.trace(gl2 @ gl2))

    # Calculate interaction terms
    interaction_term_sample = -2 * torch.trace(gL1 @ pi_samp @ gL2 @ pi_samp.T)
    interaction_term_feat = -2 * torch.trace(gl1 @ pi_feat @ gl2 @ pi_feat.T)

    return trace_term + interaction_term_sample + interaction_term_feat
def get_cost_ufgot(pi_samp, pi_feat, data, data_T, tuple_pxy_samp, tuple_pxy_feat, hyperparams, entropic_mode, filter_type='identity',device='cpu'):
    r"""
    Calculate CO-UFGOT cost using filtered covariance matrices
    """
    rho, eps = hyperparams
    eps_samp, eps_feat = eps
    rho1, rho2, rho1_samp, rho2_samp, rho1_feat, rho2_feat = rho
    px_samp, py_samp, pxy_samp = tuple_pxy_samp
    px_feat, py_feat, pxy_feat = tuple_pxy_feat
    gL1, gL2, X, Y, D_samp, alpha_samp = data
    gl1, gl2, _, _, D_feat, alpha_feat = data_T

    pi1_samp, pi2_samp = pi_samp.sum(1), pi_samp.sum(0)  # Row sum and column sum
    pi1_feat, pi2_feat = pi_feat.sum(1), pi_feat.sum(0)

    # UFGOT distance term
    ufgot_distance = compute_ufgot_distance(gL1, gL2, gl1, gl2,pi_samp, pi_feat, filter_type, device)
    cost = ufgot_distance
    #print('UFGOT distance term i.e., four traces:', cost)
    # Marginal distribution penalty terms
    if rho1 != float("inf") and rho1 != 0:
        cost = cost + rho1 * quad_kl(pi1_samp, pi1_feat, px_samp, px_feat)
    if rho2 != float("inf") and rho2 != 0:
        cost = cost + rho2 * quad_kl(pi2_samp, pi2_feat, py_samp, py_feat)
    #print('Marginal terms:', cost-ufgot_distance)
    # UOT part
    if alpha_samp != 0:
        uot_cost_samp = (D_samp * pi_samp).sum()
        if rho1_samp != float("inf") and rho1_samp != 0:
            uot_cost_samp = uot_cost_samp + rho1_samp * kl(pi1_samp, px_samp)
        if rho2_samp != float("inf") and rho2_samp != 0:
            uot_cost_samp = uot_cost_samp + rho2_samp * kl(pi2_samp, py_samp)
        cost = cost + alpha_samp * uot_cost_samp
        print('UOT_sample：',alpha_samp * uot_cost_samp)
    if alpha_feat != 0:
        uot_cost_feat = (D_feat * pi_feat).sum()
        if rho1_feat != float("inf") and rho1_feat != 0:
            uot_cost_feat = uot_cost_feat + rho1_feat * kl(pi1_feat, px_feat)
        if rho2_feat != float("inf") and rho2_feat != 0:
            uot_cost_feat = uot_cost_feat + rho2_feat * kl(pi2_feat, py_feat)
        cost = cost + alpha_feat * uot_cost_feat
        #print('UOT_feature：', alpha_feat * uot_cost_feat)
    # Entropic part
    ent_cost = cost
    if entropic_mode == "joint" and eps_samp != 0:
        ent_cost = ent_cost + eps_samp * quad_kl(pi_samp, pi_feat, pxy_samp, pxy_feat)
    elif entropic_mode == "independent":
        if eps_samp != 0:
            ent_cost = ent_cost + eps_samp * kl(pi_samp, pxy_samp)
        if eps_feat != 0:
            ent_cost = ent_cost + eps_feat * kl(pi_feat, pxy_feat)
    #print('Entropic regularization part:',ent_cost - cost)

    return cost.item(), ent_cost.item()


def get_local_cost_ufgot(data, pi, tuple_p, hyperparams, entropic_mode):
    r"""
    Calculate cost of the UFGOT.
    cost = (g(L1)**2 @ P_#1 + g(L2)**2 @ P_#2 - 2 * g(L1) @ P @ g(L2).T) +
            rho1 * approx_kl(P_#1 | a) + rho2 * approx_kl(P_#2 | b) +
            eps * kl(P | a \otimes b)
    """

    rho, eps = hyperparams
    rho1, rho2, _, _, _, _ = rho
    a, b, ab = tuple_p
    g1, g2, X, Y, D, alpha = data
    g1_sqr = g1**2
    g2_sqr = g2**2
    n1, d1 = X.shape
    n2, d2 = Y.shape
    pi1, pi2 = pi.sum(1), pi.sum(0)

    A = (g1_sqr @ pi1)[:, None]
    B = (g2_sqr @ pi2)[None, :]
    cross = - 2 * g1 @ pi @ g2
    FGGW = A + B + cross
    #print('FGGW:',FGGW)
    cost = alpha * D + FGGW
    #print('Custom penalty matrix:',alpha * D)
    if rho1 != float("inf") and rho1 != 0:
        cost = cost + rho1 * approx_kl(pi1, a)
        # print('Adding row soft constraint:',rho1 * approx_kl(pi1, a))
    if rho2 != float("inf") and rho2 != 0:
        cost = cost + rho2 * approx_kl(pi2, b)
        # print('Adding column soft constraint:',rho2 * approx_kl(pi2, b))
    if entropic_mode == "joint" and eps[0] > 0:
        cost = cost + eps[0] * kl(pi, ab)
        # print('Adding regularization constraint:',eps[0] * kl(pi, ab))
    return cost


def solver_ufgot(
        X,
        Y,
        gL1=None,
        gL2=None,
        gl1=None,
        gl2=None,
        px=(None, None),  # Marginals, default to uniform distribution
        py=(None, None),
        rho=(float("inf"), float("inf"), 0, 0, 0, 0),
        uot_mode=("entropic", "entropic"),  # "entropic" (Sinkhorn) or "mm" (max-min/fixed-point)
        eps=(1e-2, 1e-2),  # Entropic regularization strength (sample/feature)
        entropic_mode="joint",
        alpha=(1, 1),  # Fusion coefficients (linear term weights for sample/feature)
        D=(None, None),
        init_pi=(None, None),  # Initial transport matrices
        init_duals=(None, None),  # Initial dual variables
        log=False,
        verbose=True,
        early_stopping_tol=1e-6,
        eval_bcd=10,
        eval_uot=1,
        tol_bcd=1e-7,
        nits_bcd=100,
        tol_uot=1e-7,
        nits_uot=500,
        horizonal=False,
        vertical=False,
        filter='identity',
        device='cpu'):
    # Horizontal, vertical solver
    # Initialization
    nx, dx = X.shape
    ny, dy = Y.shape
    device, dtype = X.device, X.dtype
    # hyper-parameters
    if isinstance(eps, float) or isinstance(eps, int):
        eps = (eps, eps)
    if not isinstance(eps, tuple):
        raise ValueError("Epsilon must be either a scalar or a tuple of scalars.")
    # if use joint penalisation for couplings, then only use the first value epsilon.
    if entropic_mode == "joint":
        eps = (eps[0], eps[0])

    if isinstance(alpha, float) or isinstance(alpha, int):
        alpha = (alpha, alpha)
    if not isinstance(alpha, tuple):
        raise ValueError("Alpha must be either a scalar or a tuple of scalars.")

    if isinstance(uot_mode, str):
        uot_mode = (uot_mode, uot_mode)
    if not isinstance(uot_mode, tuple):
        raise ValueError("uot_mode must be either a string or a tuple of strings.")

    # some constants
    rho1, rho2, rho1_samp, rho2_samp, rho1_feat, rho2_feat = rho
    eps_samp, eps_feat = eps
    uot_mode_samp, uot_mode_feat = uot_mode
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

    if px_samp is None:  # Column vector, uniform distribution
        px_samp = torch.ones(nx).to(device).to(dtype) / nx
    if px_feat is None:
        px_feat = torch.ones(dx).to(device).to(dtype) / dx
    if py_samp is None:
        py_samp = torch.ones(ny).to(device).to(dtype) / ny
    if py_feat is None:
        py_feat = torch.ones(dy).to(device).to(dtype) / dy
    pxy_samp = px_samp[:, None] * py_samp[None, :]  # pxy_samp[i, j] = px_samp[i] * py_samp[j]
    pxy_feat = px_feat[:, None] * py_feat[None, :]  # pxy_feat[i, j] = px_feat[i] * py_feat[j]

    tuple_pxy_samp = (px_samp, py_samp, pxy_samp)
    tuple_pxy_feat = (px_feat, py_feat, pxy_feat)
    tuple_log_pxy_samp = (px_samp.log(), py_samp.log(), pxy_samp)
    tuple_log_pxy_feat = (px_feat.log(), py_feat.log(), pxy_feat)

    # constant data variables
    alpha_samp, alpha_feat = alpha
    D_samp, D_feat = D
    if D_samp is None or alpha_samp == 0:
        D_samp, alpha_samp = 0, 0
    if D_feat is None or alpha_feat == 0:
        D_feat, alpha_feat = 0, 0

    data = (gL1, gL2, X, Y, D_samp, alpha_samp)
    data_T = (gl1, gl2, X.T, Y.T, D_feat, alpha_feat)

    # initialise coupling and dual vectors
    pi_samp, pi_feat = init_pi
    if pi_samp is None:
        pi_samp = pxy_samp  # size nx x ny, initial values are all the same
    if pi_feat is None:
        pi_feat = pxy_feat  # size dx x dy
        # print(pi_feat.shape)

    P = [pi_samp, pi_feat]
    if "entropic" in uot_mode:
        self_uot_ent = partial(uot_ent, n_iters=nits_uot, tol=tol_uot, eval_freq=eval_uot)
        duals_samp, duals_feat = init_duals
        if uot_mode_samp == "entropic" and duals_samp is None:
            duals_samp = (torch.zeros_like(px_samp), torch.zeros_like(py_samp))  # shape nx, ny
        if uot_mode_feat == "entropic" and duals_feat is None:
            duals_feat = (torch.zeros_like(px_feat), torch.zeros_like(py_feat))  # shape dx, dy
    elif "mm" in uot_mode:
        self_uot_mm = partial(uot_mm, n_iters=nits_uot, tol=tol_uot, eval_freq=eval_uot)

    hyperparams = (rho, eps)
    self_get_local_cost = partial(get_local_cost_ufgot, hyperparams=hyperparams, entropic_mode=entropic_mode)

    for idx in range(nits_bcd):
        P0_prev = P[0].detach().clone()
        P1_prev = P[1].detach().clone()
        if horizonal:
            mass = P[1].sum()
            # Update pi_feat (feature coupling)
            uot_cost = self_get_local_cost(data_T, P[1], tuple_pxy_feat)  # size dx x dy
            new_rho1 = rho1 * mass + alpha_feat * rho1_feat
            new_rho2 = rho2 * mass + alpha_feat * rho2_feat
            new_eps = mass * eps_feat if entropic_mode == "joint" else eps_feat  # remains unchanged
            uot_params = (new_rho1, new_rho2, new_eps)

            if uot_mode_feat == "entropic":  # Sinkhorn algorithm
                duals_feat, P[1] = self_uot_ent(uot_cost, duals_feat, tuple_log_pxy_feat, uot_params)
            elif uot_mode_feat == "mm":  # Min-Max algorithm
                duals_feat, P[1] = self_uot_mm(uot_cost, P[1], tuple_pxy_feat, uot_params)
            P[1] = (mass / P[1].sum()).sqrt() * P[1]

        if vertical:
            mass = P[0].sum()
            # Update pi_samp (sample coupling)
            uot_cost = self_get_local_cost(data, P[0], tuple_pxy_samp)  # size nx x ny
            new_rho1 = rho1 * mass + alpha_samp * rho1_samp
            new_rho2 = rho2 * mass + alpha_samp * rho2_samp
            new_eps = mass * eps_samp if entropic_mode == "joint" else eps_samp
            uot_params = (new_rho1, new_rho2, new_eps)

            if uot_mode_samp == "entropic":
                duals_samp, P[0] = self_uot_ent(uot_cost, duals_samp, tuple_log_pxy_samp, uot_params)
            elif uot_mode_samp == "mm":
                duals_samp, P[0] = self_uot_mm(uot_cost, P[0], tuple_pxy_samp, uot_params)
            P[0] = (mass / P[0].sum()).sqrt() * P[0]

        #print(idx)
        if vertical:
            err_samp = (P[0] - P0_prev).abs().max().item()
            #print('Check if p_samp is updated:', err_samp)
            if np.isnan(err_samp):
                print("NaN detected in err or err_feat, breaking loop.")
                break
            if P[0].isnan().any():
                print("There is NaN in coupling")
            if err_samp < tol_bcd:
                break
        if horizonal:
            err_feat = (P[1] - P1_prev).abs().max().item()
            #print('Check if p_feature is updated:', err_feat)
            if np.isnan(err_feat):
                print("NaN detected in err or err_feat, breaking loop.")
                break
            if P[1].isnan().any():
                print("There is NaN in coupling")
            if err_feat < tol_bcd:
                break

    if horizonal:
        return P[1]
    if vertical:
        return P[0]

def solver_coufgot(
        X,
        Y,
        gL1=None,
        gL2=None,
        gl1=None,
        gl2=None,
        px=(None, None),  # Marginals, default to uniform distribution
        py=(None, None),
        rho=(float("inf"), float("inf"), 0, 0, 0, 0),
        uot_mode=("entropic", "entropic"),  # "entropic" (Sinkhorn) or "mm" (max-min/fixed-point)
        eps=(1e-2, 1e-2),  # Entropic regularization strength (sample/feature)
        entropic_mode="joint",
        alpha=(1, 1),  # Fusion coefficients (linear term weights for sample/feature)
        D=(None, None),
        init_pi=(None, None),  # Initial transport matrices
        init_duals=(None, None),  # Initial dual variables
        log=False,
        verbose=True,
        early_stopping_tol=1e-6,
        eval_bcd=10,
        eval_uot=1,
        tol_bcd=1e-7,
        nits_bcd=100,
        tol_uot=1e-7,
        nits_uot=500,
        filter='identity',
        device='cpu'
):
    r"""Coordinate both samples and features, diagonal and mosaic alignment, updated version of solver
    Parameters
    ----------
    X: matrix of size n x dx. First input data.
    Y: matrix of size n x dy. Second input data.
    D: matrix of size nx x ny. Sample matrix, in case of fused GW
    px: tuple of 2 vectors of length (n, dx). Measures assigned on rows and columns of X.
        Uniform distributions by default.
    py: tuple of 2 vectors of length (n, dy). Measures assigned on rows and columns of Y.
        Uniform distributions by default.
    rho: tuple of 6 relaxation marginal-relaxation parameters for UGW and UOT.
    uot_mode: string or tuple of strings. Uot modes for each update of couplings
        uot_mode = "entropic": use Sinkhorn algorithm in each BCD iteration.
        uot_mode = "mm": use maximisation-minimisation algorithm in each BCD iteration.
    eps: scalar or tuple of scalars.
        Regularisation parameters for entropic approximation of sample and feature couplings.
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
    x_alig: matrix of size n x dy. alig matrix.
    """
    nx, dx = X.shape
    ny, dy = Y.shape
    device, dtype = X.device, X.dtype
    # hyper-parameters
    if isinstance(eps, float) or isinstance(eps, int):
        eps = (eps, eps)
    if not isinstance(eps, tuple):
        raise ValueError("Epsilon must be either a scalar or a tuple of scalars.")
    # if use joint penalisation for couplings, then only use the first value epsilon.
    if entropic_mode == "joint":
        eps = (eps[0], eps[0])

    if isinstance(alpha, float) or isinstance(alpha, int):
        alpha = (alpha, alpha)
    if not isinstance(alpha, tuple):
        raise ValueError("Alpha must be either a scalar or a tuple of scalars.")

    if isinstance(uot_mode, str):
        uot_mode = (uot_mode, uot_mode)
    if not isinstance(uot_mode, tuple):
        raise ValueError("uot_mode must be either a string or a tuple of strings.")

    # some constants
    rho1, rho2, rho1_samp, rho2_samp, rho1_feat, rho2_feat = rho
    eps_samp, eps_feat = eps
    uot_mode_samp, uot_mode_feat = uot_mode
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

    if px_samp is None:  # Column vector, uniform distribution
        px_samp = torch.ones(nx).to(device).to(dtype) / nx
    if px_feat is None:
        px_feat = torch.ones(dx).to(device).to(dtype) / dx
    if py_samp is None:
        py_samp = torch.ones(ny).to(device).to(dtype) / ny
    if py_feat is None:
        py_feat = torch.ones(dy).to(device).to(dtype) / dy
    pxy_samp = px_samp[:, None] * py_samp[None, :]  # pxy_samp[i, j] = px_samp[i] * py_samp[j]
    pxy_feat = px_feat[:, None] * py_feat[None, :]  # pxy_feat[i, j] = px_feat[i] * py_feat[j]

    tuple_pxy_samp = (px_samp, py_samp, pxy_samp)
    tuple_pxy_feat = (px_feat, py_feat, pxy_feat)
    tuple_log_pxy_samp = (px_samp.log(), py_samp.log(), pxy_samp)
    tuple_log_pxy_feat = (px_feat.log(), py_feat.log(), pxy_feat)

    # constant data variables
    alpha_samp, alpha_feat = alpha
    D_samp, D_feat = D
    if D_samp is None or alpha_samp == 0:
        D_samp, alpha_samp = 0, 0
    if D_feat is None or alpha_feat == 0:
        D_feat, alpha_feat = 0, 0

    data = (gL1, gL2, X, Y, D_samp, alpha_samp)
    data_T = (gl1, gl2, X.T, Y.T, D_feat, alpha_feat)

    # initialise coupling and dual vectors
    pi_samp, pi_feat = init_pi
    if pi_samp is None:
        pi_samp = pxy_samp  # size nx x ny, initial values are all the same
    if pi_feat is None:
        pi_feat = pxy_feat  # size dx x dy
        #print(pi_feat.shape)

    P = [pi_samp, pi_feat]
    if "entropic" in uot_mode:
        self_uot_ent = partial(uot_ent, n_iters=nits_uot, tol=tol_uot, eval_freq=eval_uot)
        duals_samp, duals_feat = init_duals
        if uot_mode_samp == "entropic" and duals_samp is None:
            duals_samp = (torch.zeros_like(px_samp), torch.zeros_like(py_samp))  # shape nx, ny
        if uot_mode_feat == "entropic" and duals_feat is None:
            duals_feat = (torch.zeros_like(px_feat), torch.zeros_like(py_feat))  # shape dx, dy

    elif "mm" in uot_mode:
        self_uot_mm = partial(uot_mm, n_iters=nits_uot, tol=tol_uot, eval_freq=eval_uot)

    hyperparams = (rho, eps)
    self_get_local_cost = partial(get_local_cost_ufgot, hyperparams=hyperparams, entropic_mode=entropic_mode)
    self_get_cost = partial(get_cost_ufgot, data=data, data_T=data_T, tuple_pxy_samp=tuple_pxy_samp,tuple_pxy_feat=tuple_pxy_feat, \
                            hyperparams=hyperparams, entropic_mode=entropic_mode,filter_type=filter,device=device)

    # initialise log
    log_cost = []
    log_ent_cost = [float("inf")]
    err = tol_bcd + 1e-3

    for idx in range(nits_bcd):
        P0_prev = P[0].detach().clone()
        P1_prev = P[1].detach().clone()

        # Update pi_feat (feature coupling)
        mass = P[1].sum()
        new_rho1 = rho1 * mass + alpha_feat * rho1_feat
        new_rho2 = rho2 * mass + alpha_feat * rho2_feat
        new_eps = mass * eps_feat if entropic_mode == "joint" else eps_feat  # remains unchanged
        uot_cost = self_get_local_cost(data_T, P[1], tuple_pxy_feat)  # size dx x dy
        uot_params = (new_rho1, new_rho2, new_eps)
        #print("feature uot_params update =", float(new_rho1), float(new_rho2), float(new_eps))
        if uot_mode_feat == "entropic":  # Sinkhorn algorithm
            duals_feat, P[1] = self_uot_ent(uot_cost, duals_feat, tuple_log_pxy_feat, uot_params)
        elif uot_mode_feat == "mm":  # Min-Max algorithm
            duals_feat, P[1] = self_uot_mm(uot_cost, P[1], tuple_pxy_feat, uot_params)
        P[1] = (mass / P[1].sum()).sqrt() * P[1]  # shape dx x dy

        # Update pi_samp (sample coupling)
        mass = P[0].sum()
        new_rho1 = rho1 * mass + alpha_samp * rho1_samp
        new_rho2 = rho2 * mass + alpha_samp * rho2_samp
        new_eps = mass * eps_feat if entropic_mode == "joint" else eps_samp
        uot_cost = self_get_local_cost(data, P[0], tuple_pxy_samp)  # size nx x ny
        uot_params = (new_rho1, new_rho2, new_eps)
        #print("sample uot_params update =", float(new_rho1), float(new_rho2), float(new_eps))

        if uot_mode_samp == "entropic":
            duals_samp, P[0] = self_uot_ent(uot_cost, duals_samp, tuple_log_pxy_samp, uot_params)
        elif uot_mode_samp == "mm":
            duals_samp, P[0] = self_uot_mm(uot_cost, P[0], tuple_pxy_samp, uot_params)
        P[0] = (mass / P[0].sum()).sqrt() * P[0]  # shape nx x ny

        #print(idx)
        err_samp = (P[0] - P0_prev).abs().max().item()
        err_feat = (P[1] - P1_prev).abs().max().item()
        #print('Check if P[0] is updated:', err_samp)
        #print('Check if P[1] is updated:', err_feat)

        if idx % eval_bcd == 0:
            # Update error
            err_samp = (P[0] - P0_prev).abs().max().item()  # Sample transport matrix error
            err_feat = (P[1] - P1_prev).abs().max().item()  # Feature transport matrix error
            cost, ent_cost = self_get_cost(P[0], P[1],filter_type=filter)
            log_cost.append(cost)
            log_ent_cost.append(ent_cost)

            if verbose:
                print("Cost at iteration {}: {}".format(idx + 1, cost))

            if np.isnan(err_samp) or np.isnan(err_feat):
                print("NaN detected in err or err_feat, breaking loop.")
                break

            if (err_samp < tol_bcd and err_feat < tol_bcd) and abs(log_ent_cost[-2] - log_ent_cost[-1]) < early_stopping_tol:
                break

    if P[0].isnan().any() or P[1].isnan().any():
        print("There is NaN in coupling")

    if log:
        return (P[0], P[1]), (duals_samp, duals_feat), log_cost, log_ent_cost[1:]
    else:
        return (P[0], P[1]), (duals_samp, duals_feat)