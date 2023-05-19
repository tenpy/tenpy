# Copyright 2023-2023 TeNPy Developers, GNU GPLv3
from __future__ import annotations
from numpy import prod
import numpy
from typing import Any, TYPE_CHECKING

from .abelian import AbstractAbelianBackend
from .abstract_backend import AbstractBlockBackend, Block, Data, Dtype
from .no_symmetry import AbstractNoSymmetryBackend
from .nonabelian import AbstractNonabelianBackend
from ..misc import inverse_permutation
from ..symmetries.spaces import VectorSpace

__all__ = ['TorchBlockBackend', 'NoSymmetryTorchBackend', 'AbelianTorchBackend', 'NonabelianTorchBackend']


if TYPE_CHECKING:
    # can not import Tensor at runtime, since it would be a circular import
    # this clause allows mypy etc to evaluate the type-hints anyway
    from ..tensors import Tensor


class TorchBlockBackend(AbstractBlockBackend):

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
        }
        self.backend_dtype_map = {
            Dtype.float32: torch.float32,
            Dtype.float64: torch.float64,
            Dtype.complex64: torch.complex64,
            Dtype.complex128: torch.complex128,
        }
        self.BlockCls = torch.Tensor
        super().__init__(**kwargs)

    def block_tdot(self, a: Block, b: Block, idcs_a: list[int], idcs_b: list[int]) -> Block:
        return torch_module.tensordot(a, b, (idcs_a, idcs_b))

    def block_shape(self, a: Block) -> tuple[int]:
        return tuple(a.shape)

    def block_item(self, a: Block) -> float | complex:
        return a.item()

    def block_dtype(self, a: Block) -> Dtype:
        return self.tenpy_dtype_map[a.dtype]

    def block_to_dtype(self, a: Block, dtype: Dtype) -> Block:
        return a.type(self.backend_dtype_map[dtype])

    def block_copy(self, a: Block) -> Block:
        return torch_module.tensor(a, device=self.device)

    def _block_repr_lines(self, a: Block, indent: str, max_width: int, max_lines: int) -> list[str]:
        torch_module.set_printoptions(linewidth=max_width - len(indent))
        lines = [f'{indent}{line}' for line in repr(a).split('\n')]
        torch_module.set_printoptions(profile='default')
        if len(lines) > max_lines:
            first = (max_lines - 1) // 2
            last = max_lines - 1 - first
            lines = lines[:first] + [f'{indent}...'] + lines[-last:]
        return lines

    def block_outer(self, a: Block, b: Block) -> Block:
        return torch_module.tensordot(a, b, ([], []))

    def block_inner(self, a: Block, b: Block, do_conj: bool, axs2: list[int] | None) -> complex:
        # TODO is this faster than torch.sum(a * b.transpose[axs2]) ?
        axs1 = list(range(len(a.shape)))
        if axs2 is None:
            axs2 = axs1
        if do_conj:
            a = torch_module.conj(a)
        return torch_module.tensordot(a, b, (axs1, axs2))

    def block_permute_axes(self, a: Block, permutation: list[int]) -> Block:
        return torch_module.permute(a, permutation)  # TODO: this is documented as a view. is that a problem?

    def block_trace_full(self, a: Block, idcs1: list[int], idcs2: list[int]) -> float | complex:
        a = torch_module.permute(a, idcs1 + idcs2)
        trace_dim = prod(a.shape[:len(idcs1)])
        a = torch_module.reshape(a, (trace_dim, trace_dim))
        return a.diagonal(offset=0, dim1=0, dim2=1).sum(0)

    def block_trace_partial(self, a: Block, idcs1: list[int], idcs2: list[int], remaining: list[int]) -> Block:
        a = torch_module.permute(a, remaining + idcs1 + idcs2)
        trace_dim = prod(a.shape[len(remaining):len(remaining)+len(idcs1)])
        a = torch_module.reshape(a, a.shape[:len(remaining)] + (trace_dim, trace_dim))
        return a.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

    def block_conj(self, a: Block) -> Block:
        return torch_module.conj(a)

    def block_allclose(self, a: Block, b: Block, rtol: float, atol: float) -> bool:
        return torch_module.allclose(a, b, rtol=rtol, atol=atol)

    def block_squeeze_legs(self, a: Block, idcs: list[int]) -> Block:
        # TODO (JU) this is ugly... but torch.squeeze squeezes all axes of dim 1, cant control which
        idx = [0 if ax in idcs else slice(None, None, None) for ax in range(len(a.shape))]
        return a[idx]

    def block_add_axis(self, a: Block, pos: int) -> Block:
        return torch_module.unsqueeze(a, pos)

    def block_norm(self, a: Block) -> float:
        return torch_module.norm(a)

    def block_max_abs(self, a: Block) -> float:
        return torch_module.max(torch_module.max(a))

    def block_reshape(self, a: Block, shape: tuple[int]) -> Block:
        return torch_module.reshape(a, shape)

    def matrix_dot(self, a: Block, b: Block) -> Block:
        return torch_module.matmul(a, b)

    def matrix_svd(self, a: Block, algorithm: str | None) -> tuple[Block, Block, Block]:
        if algorithm is None:
            algorithm = 'gesvd'
        assert algorithm in self.svd_algorithms
        U, S, V = torch_module.linalg.svd(a, full_matrices=False, driver=algorithm)
        return U, S, V

    def matrix_qr(self, a: Block, full: bool) -> tuple[Block, Block]:
        return torch_module.qr(a, some=not full)

    def matrix_exp(self, matrix: Block) -> Block:
        raise NotImplementedError  # TODO: could not find a torch implementation via their docs...?

    def matrix_log(self, matrix: Block) -> Block:
        raise NotImplementedError  # TODO: could not find a torch implementation via their docs...?

    def block_random_uniform(self, dims: list[int], dtype: Dtype) -> Block:
        return torch_module.rand(*dims, dtype=self.backend_dtype_map[dtype], device=self.device)

    def block_random_normal(self, dims: list[int], dtype: Dtype, sigma: float) -> Block:
        # Note that if device is CUDA, this function synchronizes the device with the CPU
        mean = torch_module.zeros(size=dims, dtype=self.backend_dtype_map[dtype], device=self.device)
        std = sigma * torch_module.ones_like(mean, device=self.device)
        return torch_module.normal(mean, std)

    def block_from_numpy(self, a: numpy.ndarray) -> Block:
        return torch_module.tensor(a, device=self.device)

    def zero_block(self, shape: list[int], dtype: Dtype) -> Block:
        return torch_module.zeros(shape, dtype=self.backend_dtype_map[dtype], device=self.device)

    def eye_block(self, legs: list[int], dtype: Dtype) -> Data:
        matrix_dim = prod(legs)
        eye = torch_module.eye(matrix_dim, dtype=self.backend_dtype_map[dtype], device=self.device)
        eye = torch_module.reshape(eye, legs + legs)
        return eye


class NoSymmetryTorchBackend(TorchBlockBackend, AbstractNoSymmetryBackend):
    def __init__(self, device: str = 'cpu'):
        TorchBlockBackend.__init__(self, device=device)
        AbstractNoSymmetryBackend.__init__(self)


class AbelianTorchBackend(TorchBlockBackend, AbstractAbelianBackend):
    def __init__(self, device: str = 'cpu'):
        TorchBlockBackend.__init__(self, device=device)
        AbstractAbelianBackend.__init__(self)


class NonabelianTorchBackend(TorchBlockBackend, AbstractNonabelianBackend):
    def __init__(self, device: str = 'cpu'):
        TorchBlockBackend.__init__(self, device=device)
        AbstractNonabelianBackend.__init__(self)
