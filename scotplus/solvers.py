from scotplus.backend.package_solver_supervised import solver
import torch

# formatted with black


class SinkhornSolver:
    """
    Solver class that uses Sinkhorn scaling for the backend UOT algorithm and
    runs based on iteratively solving for P, P' and Q in isolation.
    """

    def __init__(
        self,
        nits_bcd=15,
        tol_bcd=1e-7,
        eval_bcd=1,
        nits_uot=int(1e5),
        tol_uot=1e-5,
        eval_uot=1,
        device=torch.device("cpu"),
    ):
        """
        Purpose:

        Initializer for the SinkhornSolver class, which contains all partials of
        UAGW that correspond to meaningful OT formulations.

        Parameters:

        nits_bcd: number of complete BCD iterations (GW and COOT, in most
        general case).
        tol_bcd: tolerance of each BCD iteration.
        eval_bcd: number of iterations between each BCD cost/tolerance
        calculation.
        nits_uot: number of Sinkhorn iterations per OT problem.
        tol_uot: tolerance of Sinkhorn algorithm for each OT problem.
        eval_uot: number of iterations between each Sinkhorn cost/tolerance
        calculation.
        device: torch device to use for calculations.

        Returns:

        None
        """

        self.nits_bcd = nits_bcd
        self.tol_bcd = tol_bcd
        self.eval_bcd = eval_bcd

        self.nits_uot = nits_uot
        self.tol_uot = tol_uot
        self.eval_uot = eval_uot
        self.device = device

    def gw(
        self,
        Dx,
        Dy,
        eps=1e-3,
        beta=0,
        D=0,
        entropic_mode="independent",
        px=None,
        py=None,
        init_pi=None,
        init_duals=None,
        log=False,
        verbose=True,
        early_stopping_tol=1e-6,
        device=None,
    ):
        """
        Purpose:

        Compute GW solution; i.e. compute, P, P' to minimize $\langle |D_x -
        D_y|^2, P \otimes P' \\rangle$ + reg. + supervision. Forces
        $\\alpha = 1$ and all $\\rho = \infty$.

        Parameters:

        Dx: sample intra-domain distance matrix for domain 1; nx x nx.
        Dy: sample intra-domain distance matrix for domain 2; ny x ny.
        eps: regularization parameter $\epsilon_{gw}$.
        beta: supervision coefficient.
        D: supervision cost matrix for OT with P, P' to add to cost. Shape nx x
        ny
        entropic_mode: regularize P and P' jointly or independently; "joint"
        for joint and "independent" for independent.
        px: support measure for samples of domain 1; shape nx.
        py: support measure for samples of domain 2; shape ny.
        init_pi: initialization for P and P'; shape nx x ny.
        init_duals: initialization for duals of P and P' (for use in Sinkhorn).
        log: output log of costs and entropic costs in addition to P, P', Q.
        verbose: print cost every BCD iteration.
        early_stopping_tol: tolerance checked at every eval_bcd iteration,
        measured by change in GW cost.
        device: torch device to use, if different from object-level decision.

        Returns:

        P, P', Q (only look at P and P' in this case) that minimize GW loss, in
        addition to logs if specified.
        """
        return solver(
            X=None,
            Y=None,
            Dx=Dx,
            Dy=Dy,
            eps=eps,
            alpha=1,
            px=(px, None),
            py=(py, None),
            entropic_mode=entropic_mode,
            beta=(beta, 0),
            D=(D, 0),
            init_pi=(init_pi, init_pi, None),
            init_duals=(init_duals, init_duals, None),
            log=log,
            verbose=verbose,
            early_stopping_tol=early_stopping_tol,
            eval_bcd=self.eval_bcd,
            eval_uot=self.eval_uot,
            tol_uot=self.tol_uot,
            nits_uot=self.nits_uot,
            nits_gw=1,
            nits_bcd=self.nits_bcd,
            tol_bcd=self.tol_bcd,
            device=device if device is not None else self.device
        )

    def ugw(
        self,
        Dx,
        Dy,
        eps=1e-3,
        rho=(1, 1),
        beta=0,
        D=0,
        entropic_mode="joint",
        px=None,
        py=None,
        init_pi=None,
        init_duals=None,
        log=False,
        verbose=True,
        early_stopping_tol=1e-6,
        device=None,
    ):
        """
        Purpose:

        Compute UGW solution; i.e. compute, P, P' to minimize $\langle |D_x -
        D_y|^2, P \otimes P' \rangle$ + reg. + relax. + supervision. Forces
        $\alpha = 1$.

        Parameters:

        Dx: sample intra-domain distance matrix for domain 1; nx x nx.
        Dy: sample intra-domain distance matrix for domain 2; ny x ny.
        eps: regularization parameter $\epsilon_{gw}$.
        rho: relaxation parameters $rho_1^{gw}$ and $rho_2^{gw}$.
        beta: supervision coefficient.
        D: supervision cost matrix for OT with P, P' to add to cost. Shape nx x
        ny
        entropic_mode: regularize P and P' jointly or independently; "joint"
        for joint and "independent" for independent.
        px: support measure for samples of domain 1; shape nx.
        py: support measure for samples of domain 2; shape ny.
        init_pi: initialization for P and P'; shape nx x ny.
        init_duals: initialization for duals of P and P' (for use in Sinkhorn).
        log: output log of costs and entropic costs in addition to P, P', Q.
        verbose: print cost every BCD iteration.
        early_stopping_tol: tolerance checked at every eval_bcd iteration,
        measured by change in UGW cost.
        device: torch device to use, if different from object-level decision.

        Returns:

        P, P', Q (only look at P and P' in this case) that minimize UGW loss,
        in addition to logs if specified.
        """
        if isinstance(rho, int) or isinstance(rho, float):
            rho = (rho, rho, None, None)
        if isinstance(rho, tuple) and len(rho) == 2:
            rho = (rho[0], rho[1], None, None)
        return solver(
            X=None,
            Y=None,
            Dx=Dx,
            Dy=Dy,
            eps=eps,
            rho=rho,
            alpha=1,
            beta=(beta, 0),
            D=(D, 0),
            px=(px, None),
            py=(py, None),
            entropic_mode=entropic_mode,
            init_pi=(init_pi, init_pi, None),
            init_duals=(init_duals, init_duals, None),
            log=log,
            verbose=verbose,
            early_stopping_tol=early_stopping_tol,
            eval_bcd=self.eval_bcd,
            eval_uot=self.eval_uot,
            tol_bcd=self.tol_bcd,
            nits_bcd=self.nits_bcd,
            tol_uot=self.tol_uot,
            nits_uot=self.nits_uot,
            nits_gw=1,
            device=device if device is not None else self.device
        )

    def coot(
        self,
        X,
        Y,
        eps=(1e-3, 1e-3),
        beta=(0, 0),
        D=(0, 0),
        entropic_mode="independent",
        px=(None, None),
        py=(None, None),
        init_pi=(None, None),
        init_duals=(None, None),
        log=False,
        verbose=True,
        early_stopping_tol=1e-6,
        device=None,
    ):
        """
        Purpose:

        Compute COOT solution; i.e. compute, P, Q to minimize $\langle |X -
        Y|^2, P \otimes Q \rangle$ + reg. + supervision. Forces
        $\alpha = 0$ and all $\rho = \infty$.

        Parameters:

        X: tabular dataset for domain 1; nx x dx.
        Y: tabular dataset for domain 2; ny x dy.
        eps: regularization parameter $\epsilon_{coot}$ or indep.
        regularization for P and Q.
        beta: supervision coefficients.
        D: supervision cost matrices for OT with P, Q respectively to add to
        cost. Shapes nx x ny and dx x dy respectively.
        entropic_mode: regularize P and Q jointly or independently; "joint"
        for joint and "independent" for independent.
        px: support measures for samples and features of domain 1; tuples of
        shapes nx and dx.
        py: support measures for samples and features of domain 2; tuples of
        shapes ny and dy.
        init_pi: initialization for P and Q; shapes nx x ny and dx x dy.
        init_duals: initialization for duals of P and Q (for use in Sinkhorn).
        log: output log of costs and entropic costs in addition to P, P', Q.
        verbose: print cost every BCD iteration.
        early_stopping_tol: tolerance checked at every eval_bcd iteration,
        measured by change in COOT cost.
        device: torch device to use, if different from object-level decision.

        Returns:

        P, P', Q (only look at P and Q in this case) that minimize COOT loss,
        in addition to logs if specified.
        """
        return solver(
            X=X,
            Y=Y,
            Dx=None,
            Dy=None,
            eps=eps,
            alpha=0,
            beta=beta,
            D=D,
            px=px,
            py=py,
            entropic_mode=entropic_mode,
            init_pi=(init_pi[0], init_pi[0], init_pi[1]),
            init_duals=(init_duals[0], init_duals[0], init_duals[1]),
            log=log,
            verbose=verbose,
            early_stopping_tol=early_stopping_tol,
            eval_bcd=self.eval_bcd,
            eval_uot=self.eval_uot,
            tol_bcd=self.tol_bcd,
            nits_bcd=self.nits_bcd,
            tol_uot=self.tol_uot,
            nits_uot=self.nits_uot,
            device=device if device is not None else self.device
        )

    def ucoot(
        self,
        X,
        Y,
        eps=(1e-3, 1e-3),
        rho=(1, 1),
        beta=(0, 0),
        D=(0, 0),
        entropic_mode="joint",
        px=(None, None),
        py=(None, None),
        init_pi=(None, None),
        init_duals=(None, None),
        log=False,
        verbose=True,
        early_stopping_tol=1e-6,
        device=None,
    ):
        """
        Purpose:

        Compute UCOOT solution; i.e. compute, P, Q to minimize $\langle |X -
        Y|^2, P \otimes Q \rangle$ + reg. relax. + supervision. Forces
        $\alpha = 0$.

        Parameters:

        X: tabular dataset for domain 1; nx x dx.
        Y: tabular dataset for domain 2; ny x dy.
        eps: regularization parameter $\epsilon_{coot}$ or indep.
        regularization for P and Q.
        rho: relaxation parameters $rho_1^{coot}$ and $rho_2^{coot}$
        beta: supervision coefficients.
        D: supervision cost matrices for OT with P, Q respectively to add to
        cost. Shapes nx x ny and dx x dy respectively.
        entropic_mode: regularize P and Q jointly or independently; "joint"
        for joint and "independent" for independent.
        px: support measures for samples and features of domain 1; tuples of
        shapes nx and dx.
        py: support measures for samples and features of domain 2; tuples of
        shapes ny and dy.
        init_pi: initialization for P and Q; shapes nx x ny and dx x dy.
        init_duals: initialization for duals of P and Q (for use in Sinkhorn).
        log: output log of costs and entropic costs in addition to P, P', Q.
        verbose: print cost every BCD iteration.
        early_stopping_tol: tolerance checked at every eval_bcd iteration,
        measured by change in COOT cost.
        device: torch device to use, if different from object-level decision.

        Returns:

        P, P', Q (only look at P and Q in this case) that minimize UCOOT loss,
        in addition to logs if specified.
        """
        if isinstance(rho, int) or isinstance(rho, float):
            rho = (None, None, rho, rho)
        if isinstance(rho, tuple) and len(rho) == 2:
            rho = (None, None, rho[0], rho[1])

        return solver(
            X=X,
            Y=Y,
            Dx=None,
            Dy=None,
            eps=eps,
            rho=rho,
            alpha=0,
            beta=beta,
            D=D,
            px=px,
            py=py,
            entropic_mode=entropic_mode,
            init_pi=(init_pi[0], init_pi[0], init_pi[1]),
            init_duals=(init_duals[0], init_duals[0], init_duals[1]),
            log=log,
            verbose=verbose,
            early_stopping_tol=early_stopping_tol,
            eval_bcd=self.eval_bcd,
            eval_uot=self.eval_uot,
            tol_bcd=self.tol_bcd,
            nits_bcd=self.nits_bcd,
            tol_uot=self.tol_uot,
            nits_uot=self.nits_uot,
            device=device if device is not None else self.device
        )

    def agw(
        self,
        X,
        Y,
        Dx,
        Dy,
        eps=(1e-3, 1e-3),
        alpha=0.5,
        beta=(0, 0),
        D=(0, 0),
        entropic_mode="joint",
        px=(None, None),
        py=(None, None),
        init_pi=(None, None),
        init_duals=(None, None),
        log=False,
        verbose=True,
        early_stopping_tol=1e-6,
        nits_gw=10,
        tol_gw=1e-7,
        device=None,
    ):
        """
        Purpose:

        Compute AGW solution; i.e. compute, P, P', Q to minimize $\alpha
        \langle |X - Y|^2, P \otimes Q \rangle + (1 - \alpha) \langle |D_x -
        D_y|^2, P \otimes P' \rangle$ + reg. + supervision. Forces all $\rho =
        \infty$.

        Parameters:

        X: tabular dataset for domain 1; nx x dx.
        Y: tabular dataset for domain 2; ny x dy.
        Dx: sample intra-domain distance matrix for domain 1; nx x nx.
        Dy: sample intra-domain distance matrix for domain 2; ny x ny.
        eps: regularization parameters $\epsilon_{gw}$ and $\epsilon_{coot}$ or
        indep. regularization for P, P', and Q (P and P' given same
        coefficient).
        alpha: tradeoff parameter for GW and COOT loss. 1 recovers GW, 0
        recovers COOT.
        beta: supervision coefficients.
        D: supervision cost matrices for OT with P, P', Q respectively to add
        to cost. Shapes nx x ny and dx x dy respectively.
        entropic_mode: regularize P, P' and Q jointly or independently; "joint"
        for joint and "independent" for independent.
        px: support measures for samples and features of domain 1; tuples of
        shapes nx and dx.
        py: support measures for samples and features of domain 2; tuples of
        shapes ny and dy.
        init_pi: initialization for P, P' and Q; shapes nx x ny and dx x dy.
        init_duals: initialization for duals of P, P' and Q (for use in
        Sinkhorn).
        log: output log of costs and entropic costs in addition to P, P', Q.
        verbose: print cost every BCD iteration.
        early_stopping_tol: tolerance checked at every eval_bcd iteration,
        measured by change in COOT cost.
        nits_gw: number of GW iterations per COOT iteration in BCD.
        tol_gw: tolerance of GW iterations (i.e., tolerance to end GW
        iterations before reaching nits_gw).
        device: torch device to use, if different from object-level decision.

        Returns:

        P, P', Q that minimize AGW loss, in addition to logs if specified.
        """
        return solver(
            X=X,
            Y=Y,
            Dx=Dx,
            Dy=Dy,
            eps=eps,
            alpha=alpha,
            beta=beta,
            D=D,
            px=px,
            py=py,
            entropic_mode=entropic_mode,
            init_pi=(init_pi[0], init_pi[0], init_pi[1]),
            init_duals=(init_duals[0], init_duals[0], init_duals[1]),
            log=log,
            verbose=verbose,
            early_stopping_tol=early_stopping_tol,
            eval_bcd=self.eval_bcd,
            eval_uot=self.eval_uot,
            tol_bcd=self.tol_bcd,
            nits_bcd=self.nits_bcd,
            tol_uot=self.tol_uot,
            nits_uot=self.nits_uot,
            nits_gw=nits_gw,
            tol_gw=tol_gw,
            device=device if device is not None else self.device
        )

    def uagw(
        self,
        X,
        Y,
        Dx,
        Dy,
        eps=(1e-3, 1e-3),
        rho=(1, 1, 1, 1),
        alpha=0.5,
        beta=(0, 0),
        D=(0, 0),
        entropic_mode="joint",
        px=(None, None),
        py=(None, None),
        init_pi=(None, None),
        init_duals=(None, None),
        log=False,
        verbose=True,
        early_stopping_tol=1e-6,
        nits_gw=10,
        tol_gw=1e-7,
        device=None,
    ):
        """
        Purpose:

        Compute UAGW solution; i.e. compute, P, P', Q to minimize $\alpha
        \langle |X - Y|^2, P \otimes Q \rangle + (1 - \alpha) \langle |D_x -
        D_y|^2, P \otimes P' \rangle$ + reg. relax. + supervision.

        Parameters:

        X: tabular dataset for domain 1; nx x dx.
        Y: tabular dataset for domain 2; ny x dy.
        Dx: sample intra-domain distance matrix for domain 1; nx x nx.
        Dy: sample intra-domain distance matrix for domain 2; ny x ny.
        eps: regularization parameters $\epsilon_{gw}$ and $\epsilon_{coot}$ or
        indep. regularization for P, P', and Q (P and P' given same
        coefficient).
        rho: relaxation parameters $rho_1^{gw}, rho_2^{gw}, rho_1^{coot}, rho_2^
        {coot}$.
        alpha: tradeoff parameter for GW and COOT loss. 1 recovers GW, 0
        recovers COOT.
        beta: supervision coefficients.
        D: supervision cost matrices for OT with P, P', Q respectively to add
        to cost. Shapes nx x ny and dx x dy respectively.
        entropic_mode: regularize P, P' and Q jointly or independently; "joint"
        for joint and "independent" for independent.
        px: support measures for samples and features of domain 1; tuples of
        shapes nx and dx.
        py: support measures for samples and features of domain 2; tuples of
        shapes ny and dy.
        init_pi: initialization for P, P' and Q; shapes nx x ny and dx x dy.
        init_duals: initialization for duals of P, P' and Q (for use in
        Sinkhorn).
        log: output log of costs and entropic costs in addition to P, P', Q.
        verbose: print cost every BCD iteration.
        early_stopping_tol: tolerance checked at every eval_bcd iteration,
        measured by change in COOT cost.
        nits_gw: number of GW iterations per COOT iteration in BCD.
        tol_gw: tolerance of GW iterations (i.e., tolerance to end GW
        iterations before reaching nits_gw).
        device: torch device to use, if different from object-level decision.

        Returns:

        P, P', Q that minimize UAGW loss, in addition to logs if specified.
        """

        if isinstance(rho, int) or isinstance(rho, float):
            rho = (rho, rho, rho, rho)
        if isinstance(rho, tuple) and len(rho) == 2:
            rho = (rho[0], rho[0], rho[1], rho[1])

        return solver(
            X=X,
            Y=Y,
            Dx=Dx,
            Dy=Dy,
            eps=eps,
            rho=rho,
            alpha=alpha,
            beta=beta,
            D=D,
            px=px,
            py=py,
            entropic_mode=entropic_mode,
            init_pi=(init_pi[0], init_pi[0], init_pi[1]),
            init_duals=(init_duals[0], init_duals[0], init_duals[1]),
            log=log,
            verbose=verbose,
            early_stopping_tol=early_stopping_tol,
            eval_bcd=self.eval_bcd,
            eval_uot=self.eval_uot,
            tol_bcd=self.tol_bcd,
            nits_bcd=self.nits_bcd,
            tol_uot=self.tol_uot,
            nits_uot=self.nits_uot,
            nits_gw=nits_gw,
            tol_gw=tol_gw,
            device=device if device is not None else self.device
        )
