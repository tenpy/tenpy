
from ..linalg.sparse import NpcLinearOperator


class Sweep:

    def __init__(self, EffectiveH):
        self.EffectiveH = EffectiveH  # class type
        self.stats = {}

    def environment_sweeps(self, N_sweeps):
        """Perform `N_sweeps` sweeps without bond optimization to update the environment."""
        if N_sweeps <= 0:
            return
        if self.verbose >= 1:
            print("Updating environment")
        for k in range(N_sweeps):
            self.sweep(optimize=False)
            if self.verbose >= 1:
                print('.', end='', flush=True)
        if self.verbose >= 1:
            print("", flush=True)  # end line

    def sweep(self, optimize=True, **kwargs):
        """One 'sweep' of the DMRG algorithm.

        Iteratate over the bond which is optimized, to the right and
        then back to the left to the starting point.
        If optimize=False, don't actually diagonalize the effective hamiltonian,
        but only update the environment.

        Parameters
        ----------
        optimize : bool
            Whether we actually optimize to find the ground state of the effective Hamiltonian.
            (If False, just update the environments).
        **kwargs : dict
            Further parameters given to :meth:`update_local` and :meth:`post_update_local`

        Returns
        -------
        max_trunc_err : float
            Maximal truncation error introduced.
        max_E_trunc : ``None`` | float
            ``None`` if meas_E_trunc is False, else the maximal change of the energy due to the
            truncation.
        """
        E_trunc_list = []
        trunc_err_list = []
        schedule_i0, update_LP_RP = self._get_sweep_schedule()

        # the actual sweep
        for i0, upd_env in zip(schedule_i0, update_LP_RP):
            if self.verbose >= 10:
                print("in sweep: i0 =", i0)
            # --------- the main work --------------
            update_data = self.update_local(i0, optimize=optimize)
            if update_LP:
                self.update_LP(i0, U)  # (requires updated B)
                for o_env in self.ortho_to_envs:
                    o_env.get_LP(i0 + 1, store=True)
            if update_RP:
                self.update_RP(i0, VH)
                for o_env in self.ortho_to_envs:
                    o_env.get_RP(i0, store=True)
            self.post_update_local(update_data)
            # collect statistics
            self.update_stats['i0'].append(i0)
            self.update_stats['age'].append(age)
            self.update_stats['E_total'].append(E_total)
            self.update_stats['N_lanczos'].append(N_lanczos)
            self.update_stats['time'].append(time.time() - self.time0)
            E_trunc_list.append(E_trunc)
            trunc_err_list.append(trunc_err.eps)

        if optimize:  # count optimization sweeps
            self.sweeps += 1
            if self.chi_list is not None:
                new_chi_max = self.chi_list.get(self.sweeps, None)
                if new_chi_max is not None:
                    self.trunc_params['chi_max'] = new_chi_max
                    if self.verbose >= 1:
                        print("Setting chi_max =", new_chi_max)
            # update mixer
            if self.mixer is not None:
                self.mixer = self.mixer.update_amplitude(self.sweeps)
        if meas_E_trunc:
            return np.max(trunc_err_list), np.max(E_trunc_list)
        else:
            return np.max(trunc_err_list), None


    def get_sweep_schedule(self):
        """Define the schedule of the sweep.

        Returns
        -------

        """
        L = self.env.L
        if self.finite:
            schedule_i0 = list(range(0, L - 1)) + list(range(L - 3, 0, -1))
            update_LP_RP = [[True, False]] * (L - 2) + [[False, True]] * (L - 2)
        else:
            assert (L >= 2)
            schedule_i0 = list(range(0, L)) + list(range(L, 0, -1))
            update_LP_RP = [[True, True]] * 2 + [[True, False]] * (L-2) + \
                           [[True, True]] * 2 + [[False, True]] * (L-2)
        return schedule_i0, update_LP_RP

    def prepare_update(self, i0):
        EffectiveH = self.EffectiveH
        self.eff_H = EffectiveH(self.env, i0)
        return theta


    def update_local(self, i0, theta, **kwargs):
        raise NotImplementedError("needs to be overwritten by subclass")

    def post_update_local(self, **kwargs):
        raise NotImplementedError("needs to be overwritten by subclass")


    def update_LP(self, i0, U):
        """Update left part of the environment.

        Parameters
        ----------
        i0 : int
            Site index. We calculate ``self.env.get_LP(i0+1)``.
        """
        self.env.get_LP(i0 + 1, store=True)  # as implemented directly in the environment

    def update_RP(self, i0, VH):
        """Update right part of the environment.

        Parameters
        ----------
        i0 : int
            Site index. We calculate ``self.env.get_RP(i0)``.
        VH : :class:`~tenpy.linalg.np_conserved.Array`
            The U as returned by SVD, with combined legs, labels ``'vL', '(vR.p1)'``.
        """
        self.env.get_RP(i0 + self.EffectiveH.length - 2, store=True)
        # as implemented directly in the environment




class EffectiveH(NpcLinearOperator):
    # Documentaion: This is the local effective Hamiltonian
    # class attribute length
    # provides matvec, __init__ from env, i0
    length = None

    def matvec(self, theta):
        r"""Apply the effective Hamiltonian to `theta`.

        This function turns :class:`EffectiveH` to a linear operator, which can be
        used for :func:`~tenpy.linalg.lanczos.lanczos`. Pictorially for a two-site effective
        Hamiltonian::

            |        .----theta---.
            |        |    |   |   |
            |       LP----H0--H1--RP
            |        |    |   |   |
            |        .---       --.

        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Wave function to apply the effective Hamiltonian to.

        Returns
        -------
        H_theta : :class:`~tenpy.linalg.np_conserved.Array`
            Result of applying the effective Hamiltonian to `theta`, :math:`H |\theta>`.
        """
        raise NotImplementedError("This function should be implemented in derived classes")


class SingleSiteH(EffectiveH):
    """Class defining the one site Hamiltonian for Lanczos

    Parameters
    ----------


    Attributes
    ----------
    Lp : :class:`tenpy.linalg.np_conserved.Array`
        left part of the environment
    Rp : :class:`tenpy.linalg.np_conserved.Array`
        right part of the environment
    W : :class:`tenpy.linalg.np_conserved.Array`
        MPO which is applied to the 'p0' leg of theta
    """
    length = 1

    def __init__(self, env, i0, fracture=False):
        self.Lp = env.get_LP(  # a,ap,m
        self.Rp = Rp  # b,bp,n
        self.W = W  # m,n,i,ip

    def matvec(self, theta):
        theta = theta.split_legs(['(vL.p.vR)'])
        Lp = self.Lp
        Rp = self.Rp
        x = npc.tensordot(Lp, theta, axes=('vR', 'vL'))
        x = npc.tensordot(x, self.W, axes=(['p', 'wR'], ['p*', 'wL']))
        x = npc.tensordot(x, Rp, axes=(['vR', 'wR'], ['vL', 'wL']))
        #TODO:next line not needed. Since the transpose does not do anything, should not cost anything. Keep for safety ?
        x = x.transpose(['vR*', 'p', 'vL*'])
        x = x.iset_leg_labels(['vL', 'p', 'vR'])
        h = x.combine_legs(['vL', 'p', 'vR'])
        return h


class TwoSiteH(EffectiveH):
    pass # TODO




