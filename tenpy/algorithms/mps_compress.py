r"""Compression of a MPS.


"""
import numpy as np

from ..linalg import np_conserved as npc
from .truncation  import truncate


def compress_MPS(psi, trunc_par):
	r"""Takes an MPS and compresses it.

	Parameters
	----------
	psi: MPS
		MPS to be compressed
	trunc_par: dict
		See :func:`truncate`
	"""

	# Do QR starting from the left
	B=psi.get_B(0,form=None)
	for i in range(psi.L-1):
		B=B.combine_legs(['vL', 'p'])
		q,r =npc.qr(B, inner_labels=['vR', 'vL'])	
		B=q.split_legs()
		psi.set_B(i,B,form=None)
		B=psi.get_B(i+1,form=None)
		B=npc.tensordot(r,B, axes=('vR', 'vL'))
	# Do SVD from right to left, truncate the singular values according to trunc_par
	for i in range(psi.L-1,0,-1):
		B=B.combine_legs(['p', 'vR'])
		u, s, vh = npc.svd(B, inner_labels=['vR', 'vL'])
		mask, norm_new, err = truncate(s, trunc_par)
		vh.iproject(mask, 'vL')
		vh=vh.split_legs()
		s=s[mask]/norm_new
		u.iproject(mask, 'vR')
		psi.set_B(i, vh, form='B')
		B=psi.get_B(i-1, form=None)
		B=npc.tensordot(B, u, axes=('vR', 'vL'))
		B.iscale_axis(s, 'vR')
		psi.set_SL(i, s)
	psi.set_B(0, B, form='B')
	
