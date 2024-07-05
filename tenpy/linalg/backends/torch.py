# Copyright (C) TeNPy Developers, GNU GPLv3
from __future__ import annotations
from numpy import prod
import numpy

from .abelian import AbelianBackend
from .abstract_backend import BlockBackend, Block
from .no_symmetry import NoSymmetryBackend
from .fusion_tree_backend import FusionTreeBackend
from ..dtypes import Dtype

__all__ = ['TorchBlockBackend', 'NoSymmetryTorchBackend', 'AbelianTorchBackend',
           'FusionTreeTorchBackend']


class TorchBlockBackend(BlockBackend):

    svd_algorithms = ['gesvdj', 'gesvd']

    def __init__(self, device: str = 'cpu', **kwargs) -> None:
        global torch_module
        try:
            import torch
        except ImportError as e:
            raise ImportError('Could not import torch. Use a different backend or install torch.') from e
        self.device = device
        torch_module = torch
        self.tenpy_dtype_map = {
            torch.float32: Dtype.float32,
            torch.float64: Dtype.float64,
            torch.complex64: Dtype.complex64,
            torch.complex128: Dtype.complex128,
            torch.bool: Dtype.bool,
            None: None,
        }
        self.backend_dtype_map = {
            Dtype.float32: torch.float32,
            Dtype.float64: torch.float64,
            Dtype.complex64: torch.complex64,
            Dtype.complex128: torch.complex128,
            Dtype.bool: torch.bool,
            None: None,
        }
        self.BlockCls = torch.Tensor
        super().__init__(**kwargs)
    
    def as_block(self, a, dtype: Dtype = None, return_dtype: bool = False) -> Block:
        block = torch_module.as_tensor(a, dtype=self.backend_dtype_map[dtype], device=self.device)
        if dtype != Dtype.bool:
            block = 1. * block  # force int to float.
        if return_dtype:
            return block, self.tenpy_dtype_map[block.dtype]
        return block

    def block_abs_argmax(self, block: Block) -> list[int]:
        flat_idx = torch_module.argmax(torch_module.abs(block))
        # OPTIMIZE numpy has np.unravel_indices. no analogue here?
        idcs = []
        for dim in reversed(block.shape):
            flat_idx, idx = divmod(flat_idx, dim)
            idcs.append(idx)
        return idcs

    def block_add_axis(self, a: Block, pos: int) -> Block:
        return torch_module.unsqueeze(a, pos)

    def block_all(self, a) -> bool:
        return torch_module.all(a)
        
    def block_allclose(self, a: Block, b: Block, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        return torch_module.allclose(a, b, rtol=rtol, atol=atol)

    def block_angle(self, a: Block) -> Block:
        return torch_module.angle(a)

    def block_any(self, a) -> bool:
        return torch_module.any(a)

    def _block_argsort(self, block: Block, axis: int) -> Block:
        return torch_module.argsort(block, dim=axis)

    def block_conj(self, a: Block) -> Block:
        return torch_module.conj(a)

    def block_copy(self, a: Block) -> Block:
        return torch_module.tensor(a, device=self.device)

    def block_dtype(self, a: Block) -> Dtype:
        return self.tenpy_dtype_map[a.dtype]

    def block_eigh(self, block: Block, sort: str = None) -> tuple[Block, Block]:
        w, v = torch_module.linalg.eigh(block)
        if sort is not None:
            perm = self.block_argsort(w, sort)
            w = w[perm]
            v = v[:, perm]
        return w, v

    def block_eigvalsh(self, block: Block, sort: str = None) -> Block:
        w = torch_module.linalg.eigvalsh(block)
        if sort is not None:
            perm = self.block_argsort(w, sort)
            w = w[perm]
        return w
    
    def block_enlarge_leg(self, block: Block, mask: Block, axis: int) -> Block:
        shape = list(block.shape)
        shape[axis] = len(mask)
        res = torch_module.zeros(shape, dtype=block.dtype, device=block.device)
        idcs = [slice(None, None, None)] * len(shape)
        idcs[axis] = mask
        res[idcs] = block
        return res
    
    def block_exp(self, a: Block) -> Block:
        return torch_module.exp(a)

    def block_from_diagonal(self, diag: Block) -> Block:
        return torch_module.diag(diag)

    def block_from_mask(self, mask: Block, dtype: Dtype) -> Block:
        M, = mask.shape
        N = torch_module.sum(mask)
        res = torch_module.zeros((N, M), dtype=self.backend_dtype_map[dtype])
        res[torch_module.arange(N), mask] = 1
        return res

    def block_from_numpy(self, a: numpy.ndarray, dtype: Dtype = None) -> Block:
        return torch_module.tensor(a, device=self.device, dtype=self.backend_dtype_map[dtype])

    def block_get_diagonal(self, a: Block, check_offdiagonal: bool) -> Block:
        res = torch_module.diagonal(a)
        if check_offdiagonal:
            if not torch_module.allclose(res, torch_module.diag(a)):
                raise ValueError('Not a diagonal block.')
        return res

    def block_imag(self, a: Block) -> Block:
        return torch_module.imag(a)

    def block_inner(self, a: Block, b: Block, do_dagger: bool) -> float | complex:
        if do_dagger:
            res = torch_module.tensordot(torch_module.conj(a), b, a.ndim)
        else:
            res = torch_module.tensordot(a, b, [range(a.ndim), reversed(range(a.ndim))])
        return res.item()

    def block_item(self, a: Block) -> float | complex:
        return a.item()

    def block_kron(self, a: Block, b: Block) -> Block:
        return torch_module.kron(a, b)

    def block_log(self, a: Block) -> Block:
        return torch_module.log(a)

    def block_max(self, a: Block) -> float | complex:
        return torch_module.max(a)
    
    def block_max_abs(self, a: Block) -> float:
        return torch_module.max(torch_module.max(a))

    def block_min(self, a: Block) -> float | complex:
        return torch_module.min(a)

    def block_norm(self, a: Block, order: int | float = 2, axis: int | None = None) -> float:
        return torch_module.linalg.vector_norm(a, ord=order, dim=axis)
    
    def block_outer(self, a: Block, b: Block) -> Block:
        return torch_module.tensordot(a, b, ([], []))

    def block_permute_axes(self, a: Block, permutation: list[int]) -> Block:
        return torch_module.permute(a, permutation)  # TODO: this is documented as a view. is that a problem?

    def block_random_uniform(self, dims: list[int], dtype: Dtype) -> Block:
        return torch_module.rand(*dims, dtype=self.backend_dtype_map[dtype], device=self.device)

    def block_random_normal(self, dims: list[int], dtype: Dtype, sigma: float) -> Block:
        # Note that if device is CUDA, this function synchronizes the device with the CPU
        mean = torch_module.zeros(size=dims, dtype=self.backend_dtype_map[dtype], device=self.device)
        std = sigma * torch_module.ones_like(mean, device=self.device)
        return torch_module.normal(mean, std)

    def block_real(self, a: Block) -> Block:
        return torch_module.real(a)

    def block_real_if_close(self, a: Block, tol: float) -> Block:
        raise NotImplementedError  # TODO

    def _block_repr_lines(self, a: Block, indent: str, max_width: int, max_lines: int) -> list[str]:
        torch_module.set_printoptions(linewidth=max_width - len(indent))
        lines = [f'{indent}{line}' for line in repr(a).split('\n')]
        torch_module.set_printoptions(profile='default')
        if len(lines) > max_lines:
            first = (max_lines - 1) // 2
            last = max_lines - 1 - first
            lines = lines[:first] + [f'{indent}...'] + lines[-last:]
        return lines

    def block_reshape(self, a: Block, shape: tuple[int]) -> Block:
        return torch_module.reshape(a, tuple(shape))

    def block_shape(self, a: Block) -> tuple[int]:
        return tuple(a.shape)

    def block_sqrt(self, a: Block) -> Block:
        return torch_module.sqrt(a)

    def block_squeeze_legs(self, a: Block, idcs: list[int]) -> Block:
        # TODO (JU) this is ugly... but torch.squeeze squeezes all axes of dim 1, cant control which
        idx = [0 if ax in idcs else slice(None, None, None) for ax in range(len(a.shape))]
        return a[idx]

    def block_stable_log(self, block: Block, cutoff: float) -> Block:
        return torch_module.where(block > cutoff, torch_module.log(block), 0.)

    def block_sum(self, a: Block, ax: int) -> Block:
        return torch_module.sum(a, ax)

    def block_sum_all(self, a: Block) -> float | complex:
        return torch_module.sum(a)

    def block_tdot(self, a: Block, b: Block, idcs_a: list[int], idcs_b: list[int]) -> Block:
        return torch_module.tensordot(a, b, (idcs_a, idcs_b))

    def block_to_dtype(self, a: Block, dtype: Dtype) -> Block:
        return a.type(self.backend_dtype_map[dtype])

    def block_trace_full(self, a: Block) -> float | complex:
        num_trace = a.ndim // 2
        trace_dim = prod(a.shape[:num_trace])
        perm = [*range(num_trace), *reversed(range(num_trace, 2 * num_trace))]
        a = torch_module.reshape(torch_module.permute(a, perm), (trace_dim, trace_dim))
        return a.diagonal(offset=0, dim1=0, dim2=1).sum(0)

    def block_trace_partial(self, a: Block, idcs1: list[int], idcs2: list[int], remaining: list[int]) -> Block:
        a = torch_module.permute(a, remaining + idcs1 + idcs2)
        trace_dim = prod(a.shape[len(remaining):len(remaining)+len(idcs1)])
        a = torch_module.reshape(a, a.shape[:len(remaining)] + (trace_dim, trace_dim))
        return a.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

    def eye_matrix(self, dim: int, dtype: Dtype) -> Block:
        return torch_module.eye(dim, dtype=self.backend_dtype_map[dtype], device=self.device)

    def get_block_element(self, a: Block, idcs: list[int]) -> complex | float | bool:
        return a[tuple(idcs)].item()

    def matrix_dot(self, a: Block, b: Block) -> Block:
        return torch_module.matmul(a, b)

    def matrix_exp(self, matrix: Block) -> Block:
        raise NotImplementedError  # TODO: could not find a torch implementation via their docs...?

    def matrix_log(self, matrix: Block) -> Block:
        raise NotImplementedError  # TODO: could not find a torch implementation via their docs...?

    def matrix_qr(self, a: Block, full: bool) -> tuple[Block, Block]:
        return torch_module.qr(a, some=not full)

    def matrix_svd(self, a: Block, algorithm: str | None) -> tuple[Block, Block, Block]:
        if algorithm is None:
            algorithm = 'gesvd'
        assert algorithm in self.svd_algorithms
        U, S, V = torch_module.linalg.svd(a, full_matrices=False, driver=algorithm)
        return U, S, V

    def ones_block(self, shape: list[int], dtype: Dtype) -> Block:
        return torch_module.ones(list(shape), dtype=self.backend_dtype_map[dtype], device=self.device)

    def synchronize(self):
        """Wait for asynchronous processes (if any) to finish"""
        torch_module.cuda.synchronize(device=self.device)

    def zero_block(self, shape: list[int], dtype: Dtype) -> Block:
        return torch_module.zeros(list(shape), dtype=self.backend_dtype_map[dtype], device=self.device)


class NoSymmetryTorchBackend(NoSymmetryBackend):
    def __init__(self, device: str = 'cpu'):
        block_backend =  TorchBlockBackend(device=device)
        NoSymmetryBackend.__init__(self, block_backend=block_backend)


class AbelianTorchBackend(AbelianBackend):
    def __init__(self, device: str = 'cpu'):
        block_backend =  TorchBlockBackend(device=device)
        AbelianBackend.__init__(self, block_backend=block_backend)


class FusionTreeTorchBackend(FusionTreeBackend):
    def __init__(self, device: str = 'cpu'):
        block_backend =  TorchBlockBackend(device=device)
        FusionTreeBackend.__init__(self, block_backend=block_backend)
