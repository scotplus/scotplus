from scootr.backend.package_solver_supervised import solver

class SinkhornSolver:
    def __init__(self, nits_bcd=15, tol_bcd=1e-7, eval_bcd=1, nits_uot=int(1e5), tol_uot=1e-5, eval_uot=1):
        """
        Init
        """

        self.nits_bcd = nits_bcd
        self.tol_bcd = tol_bcd
        self.eval_bcd = eval_bcd

        self.nits_uot = nits_uot
        self.tol_uot = tol_uot
        self.eval_uot = eval_uot

    def gw(self, Dx, Dy, eps=1e-3, beta=(0,0), D=(0,0), px=None, py=None, init_pi=None, init_duals=None, log=False, verbose=True, early_stopping_tol=1e-6):
        return solver(X=None, Y=None, Dx=Dx, Dy=Dy, eps=eps, alpha=1, px=(px, None), py=(py, None), entropic_mode="independent", beta=beta, D=D, init_pi=(init_pi, init_pi, None), init_duals=(init_duals, init_duals, None), log=log, verbose=verbose, early_stopping_tol=early_stopping_tol, eval_bcd=self.eval_bcd, eval_uot=self.eval_uot, tol_uot=self.tol_uot, nits_uot=self.nits_uot, nits_gw=1, nits_bcd=self.nits_bcd, tol_bcd=self.tol_bcd)


    def ugw(self, Dx, Dy, eps=1e-3, rho=(1, 1), beta=(0,0), D=(0,0), entropic_mode="joint", px=None, py=None, init_pi=None, init_duals=None, log=False, verbose=True, early_stopping_tol=1e-6):
        if isinstance(rho, int) or isinstance(rho, float):
            rho = (rho, rho, None, None)
        if isinstance(rho, tuple) and len(rho) == 2:
            rho = (rho[0], rho[1], None, None)
        return solver(X=None, Y=None, Dx=Dx, Dy=Dy, eps=eps, rho=rho, alpha=1, beta=beta, D=D, px=(px, None), py=(py, None), entropic_mode=entropic_mode, init_pi=(init_pi, init_pi, None), init_duals=(init_duals, init_duals, None), log=log, verbose=verbose, early_stopping_tol=early_stopping_tol, eval_bcd=self.eval_bcd, eval_uot=self.eval_uot, tol_bcd=self.tol_bcd, nits_bcd=self.nits_bcd, tol_uot=self.tol_uot, nits_uot=self.nits_uot, nits_gw=1)
    
    def coot(self, X, Y, eps=(1e-3, 1e-3), beta=(0,0), D=(0,0), px=(None, None), py=(None, None), init_pi=(None, None), init_duals=(None, None), log=False, verbose=True, early_stopping_tol=1e-6):
        return solver(X=X, Y=Y, Dx=None, Dy=None, eps=eps, alpha=0, beta=beta, D=D, px=px, py=py, entropic_mode="independent", init_pi=(init_pi[0], init_pi[0], init_pi[1]), init_duals=(init_duals[0], init_duals[0], init_duals[1]), log=log, verbose=verbose, early_stopping_tol=early_stopping_tol, eval_bcd=self.eval_bcd, eval_uot=self.eval_uot, tol_bcd=self.tol_bcd, nits_bcd=self.nits_bcd, tol_uot=self.tol_uot, nits_uot=self.nits_uot)
    
    def ucoot(self, X, Y, eps=(1e-3, 1e-3), rho=(1, 1), beta=(0,0), D=(0,0), entropic_mode="joint", px=(None, None), py=(None, None), init_pi=(None, None), init_duals=(None, None), log=False, verbose=True, early_stopping_tol=1e-6):

        if isinstance(rho, int) or isinstance(rho, float):
            rho = (None, None, rho, rho)
        if isinstance(rho, tuple) and len(rho) == 2:
            rho = (None, None, rho[0], rho[1])

        return solver(X=X, Y=Y, Dx=None, Dy=None, eps=eps, rho=rho, alpha=0, beta=beta, D=D, px=px, py=py, entropic_mode=entropic_mode, init_pi=(init_pi[0], init_pi[0], init_pi[1]), init_duals=(init_duals[0], init_duals[0], init_duals[1]), log=log, verbose=verbose, early_stopping_tol=early_stopping_tol, eval_bcd=self.eval_bcd, eval_uot=self.eval_uot, tol_bcd=self.tol_bcd, nits_bcd=self.nits_bcd, tol_uot=self.tol_uot, nits_uot=self.nits_uot)
    
    def agw(self, X, Y, Dx, Dy, eps=(1e-3, 1e-3), alpha=0.5, beta=(0,0), D=(0,0), entropic_mode="independent", px=(None, None), py=(None, None), init_pi=(None, None), init_duals=(None, None), log=False, verbose=True, early_stopping_tol=1e-6, nits_gw=10, tol_gw=1e-7):
        return solver(X=X, Y=Y, Dx=Dx, Dy=Dy, eps=eps, alpha=alpha, beta=beta, D=D, px=px, py=py, entropic_mode=entropic_mode, init_pi=(init_pi[0], init_pi[0], init_pi[1]), init_duals=(init_duals[0], init_duals[0], init_duals[1]), log=log, verbose=verbose, early_stopping_tol=early_stopping_tol, eval_bcd=self.eval_bcd, eval_uot=self.eval_uot, tol_bcd=self.tol_bcd, nits_bcd=self.nits_bcd, tol_uot=self.tol_uot, nits_uot=self.nits_uot, nits_gw=nits_gw, tol_gw=tol_gw)