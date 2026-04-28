"""Tests dealing with different devices.

It is nice to have this in a dedicated module, since depending on hardware, we may have
a lot of expected skips.
"""

import pytest

from cyten import Dtype, tensors


@pytest.mark.deselect_invalid_ChargedTensor_cases
@pytest.mark.parametrize('cls', [tensors.SymmetricTensor, tensors.ChargedTensor])
@pytest.mark.parametrize('device1', ['cpu', 'mps', 'cuda'])
@pytest.mark.parametrize('device2', ['cpu', 'mps', 'cuda'])
def test_device_control(cls, device1, device2, compatible_backend, make_compatible_tensor):
    # skip if unavailable:
    try:
        x = compatible_backend.block_backend.ones_block([1], dtype=Dtype.complex64, device=device1)
    except Exception:
        pytest.skip(reason=f'device {device1} not available / not supported')
    try:
        x = compatible_backend.block_backend.ones_block([1], dtype=Dtype.complex64, device=device2)
    except Exception:
        pytest.skip(reason=f'device {device2} not available / not supported')

    dtype = Dtype.complex64  # some devices do not support float64 / complex128

    T1 = make_compatible_tensor(cls=cls, dtype=dtype, device=device1)
    assert T1.device == compatible_backend.block_backend.as_device(device1)
    T1.test_sanity()

    T2 = tensors.on_device(T1, device=device2, copy=True)
    assert T1.device == compatible_backend.block_backend.as_device(device1)
    T1.test_sanity()
    assert T2.device == compatible_backend.block_backend.as_device(device2)
    T2.test_sanity()

    T3 = tensors.on_device(T1, device=device2, copy=False)
    assert T3 is T1
    assert T3.device == compatible_backend.block_backend.as_device(device2)
    T3.test_sanity()
