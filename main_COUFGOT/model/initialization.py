from COUFGOT.main_COUFGOT.model.optim import solver_coufgot,solver_ufgot
import torch
import numpy as np


# Provide interface for various alignment calls
class MegaWass:
    def __init__(self, nits_bcd=500, tol_bcd=1e-6, eval_bcd=5, nits_uot=100, tol_uot=1e-6, eval_uot=1, filter='identity'):
        r"""
        Initialization
        """
        self.nits_bcd = nits_bcd
        self.tol_bcd = tol_bcd
        self.eval_bcd = eval_bcd

        self.nits_uot = nits_uot
        self.tol_uot = tol_uot
        self.eval_uot = eval_uot
        self.filter = filter

    def solver_megawass_coufgot(
        self,
        X,
        Y,
        gL1=None,
        gL2=None,
        gl1=None,
        gl2=None,
        px=(None, None),
        py=(None, None),
        rho=(float("inf"), float("inf"), 0, 0, 0, 0),
        uot_mode=("entropic", "entropic"),
        eps=(1e-2, 1e-2),
        entropic_mode="joint",
        alpha=(1, 1),
        D=(None, None),
        init_pi=(None, None),
        init_duals=(None, None),
        log=False,
        verbose=False,
        early_stopping_tol=1e-6,
        filter_type='identity',
        device='cpu'
    ):
        r"""COUFGOT version corresponding to the paper
        Parameters for mode:
        - Ent-LB-UGW: alpha = 1, mode = "joint", rho1 != infty, rho2 != infty. No need to care about rho1_samp and rho2_samp.
        - EGW: alpha = 1, mode = "independent", rho1 = rho2 = infty. No need to care about rho1_samp and rho2_samp.
        - Ent-FGW: 0 < alpha < 1, D != None, mode = "independent", rho1 = rho2 = infty (so rho1_samp = rho2_samp = infty)
        - Ent-semi-relaxed GW: alpha = 1, mode = "independent", (rho1 = 0, rho2 = infty), or (rho1 = infty, rho2 = 0).
        No need to care about rho1_samp and rho2_samp.
        - Ent-semi-relaxed FGW: 0 < alpha < 1, mode = "independent", (rho1 = rho1_samp = 0, rho2 = rho2_samp = infty),
        or (rho1 = rho1_samp = infty, rho2 = rho2_samp = 0).
        - Ent-UOT: alpha = 0, mode = "independent", D != None, rho1 != infty, rho2 != infty, rho1_samp != infty, rho2_samp != infty.

        Parameters
        ----------
        X: matrix of size nx x dx
        Y: matrix of size ny x dy
        D: matrix of size nx x ny. Sample matrix, in case of fused GW
        px: tuple of 2 vectors of length (nx, dx). Measures assigned on rows and columns of X.
        py: tuple of 2 vectors of length (ny, dy). Measures assigned on rows and columns of Y.
        rho: tuple of 4 relaxation parameters for UGW and UOT.
        eps: regularisation parameter for entropic approximation.
        alpha: between 0 and 1. Interpolation parameter for fused UGW.
        entropic_mode:
            entropic_mode="joint": use UGW-like regularisation term
            entropic_mode = "independent": use COOT-like regularisation
        init_n: matrix of size nx x ny if not None. Initialisation matrix for sample coupling.
        log: True if the loss is recorded, False otherwise.
        verbose: if True then print the recorded loss.
        eval_bcd: The multiplier of iteration at which the loss is calculated. For example, if eval_bcd = 10, then the
                    loss is calculated at iteration 10, 20, 30, etc...

        Returns
        ----------
        pi_samp: matrix of size nx x ny. Sample matrix.
        pi_feat: matrix of size dx x dy. Feature matrix.
        log_cost: if log is True, return a list of loss (without taking into account the regularisation term).
        log_ent_cost: if log is True, return a list of entropic loss.
        :param filter_type:
        """
        rho1, rho2 = rho
        rho = (rho1, rho2, rho1, rho2, rho1, rho2)
        return solver_coufgot(X, Y, gL1, gL2, gl1, gl2, px, py, rho, uot_mode, eps, entropic_mode, alpha, D, init_pi, \
                    init_duals, log, verbose, early_stopping_tol, eval_bcd=self.eval_bcd, \
                    eval_uot=self.eval_uot, tol_bcd=self.tol_bcd, nits_bcd=self.nits_bcd, \
                    tol_uot=self.tol_uot, nits_uot=self.nits_uot,filter=filter_type,device=device)


    def solver_megawass_ufgot(
        self,
        X,
        Y,
        gL1=None,
        gL2=None,
        gl1=None,
        gl2=None,
        px=(None, None),
        py=(None, None),
        rho=(float("inf"), float("inf"), 0, 0, 0, 0),
        uot_mode=("entropic", "entropic"),
        eps=(1e-2, 1e-2),
        entropic_mode="joint",
        alpha=(1, 1),
        D=(None, None),
        init_pi=(None, None),
        init_duals=(None, None),
        log=False,
        verbose=False,
        early_stopping_tol=1e-6,
        horizonal=False,
        vertical=False,
        filter_type='identity',
        device='cpu'
    ):
        r"""Call single-coupling UFGOT
        Parameters for mode:
        - Ent-LB-UGW: alpha = 1, mode = "joint", rho1 != infty, rho2 != infty. No need to care about rho1_samp and rho2_samp.
        - EGW: alpha = 1, mode = "independent", rho1 = rho2 = infty. No need to care about rho1_samp and rho2_samp.
        - Ent-FGW: 0 < alpha < 1, D != None, mode = "independent", rho1 = rho2 = infty (so rho1_samp = rho2_samp = infty)
        - Ent-semi-relaxed GW: alpha = 1, mode = "independent", (rho1 = 0, rho2 = infty), or (rho1 = infty, rho2 = 0).
        No need to care about rho1_samp and rho2_samp.
        - Ent-semi-relaxed FGW: 0 < alpha < 1, mode = "independent", (rho1 = rho1_samp = 0, rho2 = rho2_samp = infty),
        or (rho1 = rho1_samp = infty, rho2 = rho2_samp = 0).
        - Ent-UOT: alpha = 0, mode = "independent", D != None, rho1 != infty, rho2 != infty, rho1_samp != infty, rho2_samp != infty.

        Parameters
        ----------
        X: matrix of size nx x dx
        Y: matrix of size ny x dy
        D: matrix of size nx x ny. Sample matrix, in case of fused GW
        px: tuple of 2 vectors of length (nx, dx). Measures assigned on rows and columns of X.
        py: tuple of 2 vectors of length (ny, dy). Measures assigned on rows and columns of Y.
        rho: tuple of 4 relaxation parameters for UGW and UOT.
        eps: regularisation parameter for entropic approximation.
        alpha: between 0 and 1. Interpolation parameter for fused UGW.
        entropic_mode:
            entropic_mode="joint": use UGW-like regularisation term
            entropic_mode = "independent": use COOT-like regularisation
        init_n: matrix of size nx x ny if not None. Initialisation matrix for sample coupling.
        log: True if the loss is recorded, False otherwise.
        verbose: if True then print the recorded loss.
        eval_bcd: The multiplier of iteration at which the loss is calculated. For example, if eval_bcd = 10, then the
                    loss is calculated at iteration 10, 20, 30, etc...

        Returns
        ----------
        pi_samp: matrix of size nx x ny. Sample matrix.
        pi_feat: matrix of size dx x dy. Feature matrix.
        log_cost: if log is True, return a list of loss (without taking into account the regularisation term).
        log_ent_cost: if log is True, return a list of entropic loss.
        """
        rho1, rho2 = rho
        rho = (rho1, rho2, rho1, rho2, rho1, rho2)
        return solver_ufgot(X, Y, gL1, gL2, gl1, gl2, px, py, rho, uot_mode, eps, entropic_mode, alpha, D, init_pi, \
                              init_duals, log, verbose, early_stopping_tol, eval_bcd=self.eval_bcd, \
                              eval_uot=self.eval_uot, tol_bcd=self.tol_bcd, nits_bcd=self.nits_bcd, \
                              tol_uot=self.tol_uot, nits_uot=self.nits_uot, horizonal=horizonal, vertical=vertical, \
                              filter=filter_type, device=device)