"""
.. TODO ::
- Overall docstring for this file.
- Finish Sweep class
- Double-check dependencies
- Rebuild DMRG and TDVP engines as subclasses of sweep
- Do testing
"""
# Copyright 2018 TeNPy Developers

from ..linalg import np_conserved as npc
from ..linalg.sparse import NpcLinearOperator


__all__ = ['Sweep', 'EffectiveH', 'OneSiteH', 'TwoSiteH']

class Sweep:
    """Prototype class for a 'sweeping' algorithm.
    
    Attributes
    ----------
    eff_H : :class:`~tenpy.algorithms.mps_sweep.EffectiveH`.
        Effective Hamiltonian, used in the local updates.
    EffectiveH : class type
        Class of `eff_H`.
    mixer : :class:`Mixer` | ``None``
        If ``None``, no mixer is used (anymore), otherwise the mixer instance.
    stats : dict
        Description
    """
    
    def __init__(self, env, EffectiveH):
        self.env = env
        self.EffectiveH = EffectiveH  # class type
        self.stats = {}
        schedule_i0, update_LP_RP = self.get_sweep_schedule()

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
        """One 'sweep' of a sweeper algorithm.

        Iteratate over the bond which is optimized, to the right and
        then back to the left to the starting point.
        If optimize=False, don't actually diagonalize the effective hamiltonian,
        but only update the environment.

        .. todo ::
        - Remove anything DMRG-specific
        - Make sure all called attributes are actually attributes of the Sweep class.

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
            theta, theta_ortho = self.prepare_update(i0)
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
            # TODO are these DMRG-specific?
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

        One 'sweep' is a full sequence from the leftmost site to the right and 
        back. Only those `LP` and `RP` that can be used later should be updated.
        
        Returns
        -------
        schedule_i0 : list
            List of indices of 'active sites'.
        update_LP_RP : list
            List of bools, which indicate whether to update the `LP` and `RP`.
        """
        L = self.env.L
        if self.env.finite:
            schedule_i0 = list(range(0, L - 1)) + list(range(L - 3, 0, -1))
            update_LP_RP = [[True, False]] * (L - 2) + [[False, True]] * (L - 2)
        else:
            assert (L >= 2)
            schedule_i0 = list(range(0, L)) + list(range(L, 0, -1))
            update_LP_RP = [[True, True]] * 2 + [[True, False]] * (L-2) + \
                           [[True, True]] * 2 + [[False, True]] * (L-2)
        return schedule_i0, update_LP_RP

    def prepare_update(self, i0):
        """Prepare everything to perform a local update.
        
        Parameters
        ----------
        i0 : int
            Index of the (left-most) active site.
        """
        EffectiveH = self.EffectiveH
        self.eff_H = EffectiveH(self.env, i0)
        # return theta  # Are not handling theta here; perhaps in subclass?

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
    """Prototype class for effective Hamiltonians used in sweep algorithms.

    As an example, the effective Hamiltonian for a two-site (DMRG) algorithm 
    looks like:
            |        .---       ---.
            |        |    |   |    |
            |       LP----H0--H1---RP
            |        |    |   |    |
            |        .---       ---.
    where ``H0`` and ``H1`` are MPO tensors.
    
    Attributes
    ----------
    length : int
        Number of (MPS) sites the effective hamiltonian covers.
    """

    # Documentation: This is the local effective Hamiltonian
    # class attribute length
    # provides matvec, __init__ from env, i0
    length = None

    def __init__(self, env, i0):
        raise NotImplementedError("This function should be implemented in derived classes")

    def matvec(self, theta):
        r"""Apply the effective Hamiltonian to `theta`.

        This function turns :class:`EffectiveH` to a linear operator, which can be
        used for :func:`~tenpy.linalg.lanczos.lanczos`. 

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


class OneSiteH(EffectiveH):
    r"""Class defining the one-site Hamiltonian for Lanczos
    
    The effective one-site Hamiltonian ooks like this:
            |        .---   ---.
            |        |    |    |
            |       LP----W0---RP
            |        |    |    |
            |        .---   ---.
    
    TODO orthogonal theta's?
    
    Parameters
    ----------
    
    Attributes
    ----------
    combine : bool
        Whether to combine legs into pipes. This combines the virtual and 
        physical leg for the left site into pipes. This reduces 
        the overhead of calculating charge combinations in the contractions,
        but one :meth:`matvec` is formally more expensive, :math:`O(2 d^3 \chi^3 D)`.
        Is originally from the wo-site method; unclear if it works wel for 1 site.
    length : int
        Number of (MPS) sites the effective hamiltonian covers.
    LHeff : :class:`~tenpy.linalg.np_conserved.Array`
        Left part of the effective Hamiltonian.
        Labels ``'(vR*.p0)', 'wR', '(vR.p0*)'`` for bra, MPO, ket.
    LP : :class:`tenpy.linalg.np_conserved.Array`
        left part of the environment
    RP : :class:`tenpy.linalg.np_conserved.Array`
        right part of the environment
    W : :class:`tenpy.linalg.np_conserved.Array`
        MPO tensor, applied to the 'p' leg of theta
    """
    length = 1

    def __init__(self, env, i0, combine=False):
        self.LP = env.get_LP(i0)
        self.RP = env.get_RP(i0)
        self.W = env.H.get_W(i0)
        self.combine = combine
        if combine:
            combine_Heff()
            

    def matvec(self, theta):
        """Apply the effective Hamiltonian to `theta`.
        
        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Labels: ``vL, p, vR``

        TODO fracture needs self.LHeff, which isn't there yet.
        TODO figure out if more steps can be shared.
        
        Returns
        -------
        theta :class:`~tenpy.linalg.np_conserved.Array`
            Product of `theta` and the effective Hamiltonian.
        """
        LP = self.LP
        RP = self.RP
        labels = theta.get_leg_labels()
        if self.combine: 
            theta = theta.combine_legs(['vL', 'p0'])  # labels 'vL.p0', 'vR'
            theta = npc.tensordot(self.LHeff, theta, axes=['(vR.p0*)', '(vL.p0)'])  # labels 'vR*.p0', 'wR', 'vR'
            theta = npc.tensordot(theta, self.RP, axes=[['wR', 'vR'], ['wL', 'vL']])  # labels 'vR*.p0', 'vL*'
            theta.ireplace_labels(['(vR*.p0)', 'vL*'], ['(vL.p0)', 'vR'])
        else:
            theta = npc.tensordot(self.LP, theta, axes=['vR', 'vL'])
            theta = npc.tensordot(self.W, theta, axes=[['wL', 'p*'], ['wR', 'p']])
            theta = npc.tensordot(theta, self.RP, axes=[['wR', 'vR'], ['wL', 'vL']])
            theta.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
        theta.itranspose(labels)  # if necessary, transpose
        return theta

    def combine_Heff(self):
        """Combine LP with W.

        TODO do we need LP and RP or can we get away with just one? Is there a
        preference?
        """
        LHeff = npc.tensordot(self.LP, self.W, axes=['wR', 'wL'])
        pipeL = LHeff.make_pipe(['vR*', 'p0'])
        self.LHeff = LHeff.combine_legs([['vR*', 'p0'], ['vR', 'p0*']],
                                        pipes=[pipeL, pipeL.conj()],
                                        new_axes=[0, -1])
        # RHeff = npc.tensordot(RP, H2, axes=['wL', 'wR'])  #single-site.
        # pipeR = RHeff.make_pipe(['p1', 'vL*'])
        # self.RHeff = RHeff.combine_legs([['p1', 'vL*'], ['p1*', 'vL']],
        #                                 pipes=[pipeR, pipeR.conj()],
        #                                 new_axes=[-1, 0])


class TwoSiteH(EffectiveH):
    r"""Class defining the two-site Hamiltonian for Lanczos
    
    The effective two-site Hamiltonian ooks like this:
            |        .---       ---.
            |        |    |   |    |
            |       LP----W0--W1---RP
            |        |    |   |    |
            |        .---       ---.
    
    
    TODO orthogonal theta's.
    
    Attributes
    ----------
    combine : bool
        Whether to combine legs into pipes. This combines the virtual and 
        physical leg for the left site and right site into pipes. This reduces 
        the overhead of calculating charge combinations in the contractions,
        but one :meth:`matvec` is formally more expensive, :math:`O(2 d^3 \chi^3 D)`.
    length : int
        Number of (MPS) sites the effective hamiltonian covers.
    LHeff : :class:`~tenpy.linalg.np_conserved.Array`
        Left part of the effective Hamiltonian.
        Labels ``'(vR*.p0)', 'wR', '(vR.p0*)'`` for bra, MPO, ket.
    RHeff : :class:`~tenpy.linalg.np_conserved.Array`
        Right part of the effective Hamiltonian.
        Labels ``'(vL.p1*)', 'wL', '(vL*.p1)'`` for ket, MPO, bra.
    LP : :class:`~tenpy.linalg.np_conserved.Array`
        Left part of the environment.
    RP : :class:`~tenpy.linalg.np_conserved.Array`
        Right part of the environment
    W1 : :class:`~tenpy.linalg.np_conserved.Array`
        Left MPO tensor, applied to the 'p0' leg of theta
    W2 : :class:`~tenpy.linalg.np_conserved.Array`
        Right MPO tensor, applied to the 'p1' leg of theta
    """
    length = 2

    def __init__(self, env, i0, combine=False):
        self.LP = env.get_LP(i0)
        self.RP = env.get_RP(i0 + 1)
        self.W1 = env.H.get_W(i0)
        self.W2 = env.H.get_W(i0 + 1)
        self.combine = combine
        if combine:
            W1 = W1.replace_labels(['p', 'p*'], ['p0', 'p0*'])  # 'wL', 'wR', 'p0', 'p0*'
            W2 = W2.replace_labels(['p', 'p*'], ['p1', 'p1*'])  # 'wL', 'wR', 'p1', 'p1*'
            combine_Heff()
        pass

    def matvec(self, theta):
        """Apply the effective Hamiltonian to `theta`.
        
        Parameters
        ----------
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            Labels: ``vL, p0, p1, vR``
        
        Returns
        -------
        theta :class:`~tenpy.linalg.np_conserved.Array`
            Product of `theta` and the effective Hamiltonian.
        """
        # TODO fracture needs self.LHeff, which isn't there yet.
        # TODO figure out if more steps can be shared.
        LP = self.LP
        RP = self.RP
        labels = theta.get_leg_labels()
        if self.combine: 
            theta = theta.split_legs(['(vL.p.vR)'])
            theta = npc.tensordot(self.LHeff, theta, axes=['(vR.p0*)', '(vL.p0)'])
            theta = npc.tensordot(theta, self.RHeff, axes=[['wR', '(p1.vR)'], ['wL', '(p1*.vL)']])
            theta.ireplace_labels(['(vR*.p0)', '(p1.vL*)'], ['(vL.p0)', '(p1.vR)'])
        else:
            theta = npc.tensordot(self.LP, theta, axes=['vR', 'vL'])
            theta = npc.tensordot(self.W, theta, axes=[['wL', 'p0*'], ['wR', 'p0']])
            theta = npc.tensordot(theta, self.H1, axes=[['wR', 'p1'], ['wL', 'p1*']])
            theta = npc.tensordot(theta, self.RP, axes=[['wR', 'vR'], ['wL', 'vL']])
            theta.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
        theta.itranspose(labels)  # if necessary, transpose
        return theta

    def combine_Heff(self):
        """Combine LP with W1 and RP with W2 to get the effectife parts of the 
        Hamiltonian with piped legs.
        """
        LHeff = npc.tensordot(LP, W1, axes=['wR', 'wL'])
        pipeL = LHeff.make_pipe(['vR*', 'p0'])
        self.LHeff = LHeff.combine_legs([['vR*', 'p0'], ['vR', 'p0*']],
                                        pipes=[pipeL, pipeL.conj()],
                                        new_axes=[0, -1])
        RHeff = npc.tensordot(RP, W2, axes=['wL', 'wR'])
        pipeR = RHeff.make_pipe(['p1', 'vL*'])
        self.RHeff = RHeff.combine_legs([['p1', 'vL*'], ['p1*', 'vL']],
                                        pipes=[pipeR, pipeR.conj()],
                                        new_axes=[-1, 0])





