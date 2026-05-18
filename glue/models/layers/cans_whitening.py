"""
Whitening layer based on CANS (Combined accelerated Newton–Schulz) iteration.

This module implements a variant of the whitening layer that computes the
inverse square root of the covariance matrix using the matrix sign function
derived from a polar decomposition.  The implementation follows these
principles:

1.  Given a positive semi‑definite covariance matrix ``sigma`` of shape
    ``(d, d)``, we build a block matrix

    ``block = [[0, sigma], [I, 0]]``

    and compute its polar decomposition.  The orthogonal factor of the polar
    decomposition coincides with the matrix sign of ``block``, and its
    off‑diagonal blocks contain the desired matrix square root and its
    inverse.  More precisely, if ``sign(block)`` is partitioned as

    ``sign(block) = [[0, S], [S_inv, 0]]``,

    then ``S`` is the principal square root of ``sigma`` and ``S_inv`` is
    ``sigma`` raised to the power ``−1/2``.  See Higham (2020) for a
    derivation【537627499015668†L190-L199】.

2.  We approximate the matrix sign via the CANS iteration from Grishin
    &amp; Kate (2023).  The iteration orthogonalizes an input matrix by
    repeatedly applying low‑degree polynomial filters.  Compared to the
    standard Newton–Schulz iteration, CANS is numerically more stable on
    ill‑conditioned inputs.  The implementation below uses the cubic
    iteration (degree=3) by default and exposes the number of iterations
    through the ``iterations`` attribute.

3.  During training we maintain exponential moving averages (EMAs) of the
    per‑batch estimates of the mean, covariance, the whitening matrix and the
    square‑root factor ``H``.  These running statistics are used during
    evaluation when ``use_running_stats_train`` or
    ``use_only_running_stats_eval`` is enabled.  The additional buffer
    ``running_H`` stores the average square root of the covariance matrix.

The public class ``WhiteningCANS2d`` inherits from ``Whitening2d`` and
implements a ``whiten_matrix`` method that returns the inverse square root
matrix.  The square root itself is returned as a side effect to update
``running_H``.

Note
----
This implementation builds the CANS block matrices for the whole mini-batch
and applies the cubic iteration with batched matrix multiplications on the
same device as the input tensors.
"""

import torch
from torch import Tensor

from models.layers.whitening import Whitening2d

# einops is used in forward_test.  Importing it lazily here avoids a hard
# dependency when the layer is instantiated but never used for evaluation.
import einops


def explicit3_coefficients(A: Tensor, B: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return x and x^3 coefficients for the optimal cubic CANS polynomial."""
    e = torch.sqrt((A.square() + A * B + B.square()) / 3)
    denom = 2 * e.pow(3) + A.square() * B + B.square() * A
    scale = 2 / denom
    coeff_x = scale * (A.square() + A * B + B.square())
    coeff_x3 = -scale
    err = (2 * e.pow(3) - A.square() * B - B.square() * A) / denom
    return coeff_x, coeff_x3, err


@torch.no_grad()
def spectral_norm_estimate(input_tensor: Tensor, power_iterations: int = 20) -> Tensor:
    """Estimate batched spectral norms without leaving the current device."""
    batch, _, cols = input_tensor.shape
    eps = torch.finfo(input_tensor.dtype).eps
    v = torch.randn(batch, cols, 1, dtype=input_tensor.dtype, device=input_tensor.device)
    v = v / v.norm(dim=1, keepdim=True).clamp_min(eps)

    for _ in range(power_iterations):
        u = input_tensor @ v
        u = u / u.norm(dim=1, keepdim=True).clamp_min(eps)
        v = input_tensor.transpose(-2, -1) @ u
        v = v / v.norm(dim=1, keepdim=True).clamp_min(eps)

    sigma = u.transpose(-2, -1) @ input_tensor @ v
    return sigma.abs().view(batch, 1, 1)


@torch.no_grad()
def cans_iteration(A: torch.Tensor, n: int, a: float, degree: int = 3, preprocess: bool = False,
                   preprocess_iters: int = 4, delta: float = 0.99) -> torch.Tensor:
    """Perform a CANS iteration to orthogonalize a matrix.

    Parameters
    ----------
    A : Tensor
        A square real matrix to be orthogonalized.  The algorithm will
        internally transpose the matrix if it has more columns than rows.
    n : int
        Maximum number of iterations.
    a : float
        Left boundary of the approximation interval.  A sensible default is
        ``0.0``.  See the reference for details.
    degree : int, optional
        Degree of the polynomial approximation.  The GPU-native implementation
        currently supports only ``3``.  Default is ``3``.
    preprocess : bool, optional
        Whether to perform a few warm‑up iterations using delta‑orthogonal
        polynomials before the main iteration.  Default is ``False``.
    preprocess_iters : int, optional
        Number of preprocess iterations.  Only used when ``preprocess`` is
        ``True``.
    delta : float, optional
        Target accuracy for the preprocessing stage.  Ignored when
        ``preprocess`` is ``False``.

    Returns
    -------
    Tensor
        An approximate orthogonal matrix ``Q`` such that ``Q^T Q ≈ I``.

    Notes
    -----
    The solver is intentionally run under ``torch.no_grad``.  The whitening
    matrix is treated as a per-step normalization statistic, so gradients flow
    through the normalized activations but not through every CANS iterate.
    """
    if degree != 3:
        raise NotImplementedError("The GPU-native CANS implementation currently supports degree=3 only")
    if preprocess:
        raise NotImplementedError("GPU-native CANS preprocessing is not implemented")

    original_dtype = A.dtype
    compute_dtype = torch.float32 if A.dtype in (torch.float16, torch.bfloat16) else A.dtype
    A = A.to(dtype=compute_dtype).clone()
    if A.dim() == 2:
        A = A.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    if A.shape[-2] < A.shape[-1]:
        A = A.transpose(-2, -1)

    A2 = A.transpose(-2, -1) @ A
    A3 = A @ A2
    denom = torch.linalg.matrix_norm(A3, ord="fro", dim=(-2, -1), keepdim=True).pow(1.0 / 3.0)
    denom = denom.clamp_min(torch.finfo(A.dtype).eps)
    A = A / denom

    b = 1.0  # right boundary is fixed to 1.0 for the normalization
    a_t = torch.as_tensor(a, dtype=A.dtype, device=A.device)
    b_t = torch.as_tensor(b, dtype=A.dtype, device=A.device)
    for _ in range(n):
        A2 = A.transpose(-2, -1) @ A
        A3 = A @ A2
        coeff_x, coeff_x3, err = explicit3_coefficients(a_t, b_t)
        A = (coeff_x * A + coeff_x3 * A3).detach()
        a_t = (1 - err).detach()
        b_t = (1 + err).detach()

    A = A.to(dtype=original_dtype)
    return A.squeeze(0) if squeeze_output else A


class WhiteningCANS2d(Whitening2d):
    """Whitening layer using CANS iteration for numerical stability.

    This layer replaces the iterative normalisation schemes in
    ``WhiteningSing2dIterNorm`` and ``WhiteningMatrixSign2dIterNorm`` with a
    CANS‑based approximation of the matrix sign function.  The computed
    whitening matrix is the bottom left block of the sign of ``[[0, sigma], [I, 0]]``
    scaled appropriately.  The corresponding square root ``H`` is extracted
    from the top right block and tracked via a running average.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register an additional buffer to store the running average of the
        # square‑root factor H (sigma^{1/2}).  This will be used during
        # evaluation when ``use_only_running_stats_eval`` is enabled.
        factory_kwargs = {"device": kwargs.get("device", None), "dtype": kwargs.get("dtype", None)}
        self.register_buffer(
            "running_H", torch.eye(self.num_features, **factory_kwargs, requires_grad=False)
        )
        self.running_H: Tensor

    def reset_running_stats(self) -> None:
        super().reset_running_stats()
        if hasattr(self, "running_H") and self.running_H is not None:
            self.running_H.copy_(torch.eye(
                self.num_features,
                dtype=self.running_H.dtype,
                device=self.running_H.device,
            ))

    def update_running_statistic(self, running_statistic: str, value: Tensor) -> None:
        """Override to support updating the running square‑root factor.

        In addition to ``running_mean``, ``running_covariance`` and
        ``running_whitening``, this method updates ``running_H`` if
        ``running_statistic`` equals ``running_H``.  Otherwise it falls back
        to the implementation in the base class.
        """
        if running_statistic == "running_H":
            cur = getattr(self, running_statistic)
            with torch.no_grad():
                cur.copy_((1 - self.momentum) * cur + self.momentum * value.detach())
        else:
            super().update_running_statistic(running_statistic, value)

    def whiten_matrix(self, sigma: Tensor, eye: Tensor) -> Tensor:
        """Compute the inverse square root of ``sigma`` using CANS.

        Parameters
        ----------
        sigma : Tensor
            Covariance matrix of shape (batch, d, d).
        eye : Tensor
            Identity matrix of shape (batch, d, d).  Unused here but
            maintained for interface compatibility.

        Returns
        -------
        Tensor
            Whitening matrix of shape (batch, d, d) approximating
            ``sigma^{-1/2}``.

        Side Effects
        ------------
        Updates ``running_H`` with the batch‑mean of the computed
        ``sigma^{1/2}`` factor.  This allows the layer to use the
        accumulated square‑root during evaluation.
        """
        B, d, _ = sigma.shape
        output_dtype = sigma.dtype
        compute_dtype = torch.float32 if sigma.dtype in (torch.float16, torch.bfloat16) else sigma.dtype
        sigma_stat = sigma.detach().to(dtype=compute_dtype)
        zeros = torch.zeros_like(sigma_stat)
        eye_d = torch.eye(d, dtype=sigma_stat.dtype, device=sigma_stat.device).expand(B, d, d)

        upper = torch.cat([zeros, sigma_stat], dim=-1)
        lower = torch.cat([eye_d, zeros], dim=-1)
        block = torch.cat([upper, lower], dim=-2)

        scale = spectral_norm_estimate(block)
        scale = scale.clamp_min(torch.finfo(block.dtype).eps)
        U = cans_iteration(block / scale, n=self.iterations, a=0.0, degree=3, preprocess=False)

        scale_sqrt = scale.sqrt()
        H_batch = U[:, :d, d:] * scale_sqrt
        wm = (U[:, d:, :d] / scale_sqrt).to(dtype=output_dtype)
        # Update running H with the batch mean
        if self.training and self.track_running_stats:
            self.update_running_statistic("running_H", H_batch.mean(dim=0))
        return wm

    def forward_test(self, x: Tensor, attention_mask: Tensor, n) -> Tensor:
        """Override evaluation to use running statistics for H as needed."""
        batch_size, w_dim = x.size(0), x.size(-1)
        # If only running stats should be used, simply apply the stored whitening
        # matrix.  The square root ``running_H`` is not used directly here but
        # retained for completeness.
        if self.use_only_running_stats_eval:
            xn = x - self.running_mean
            decorrelated = torch.bmm(
                xn,
                einops.repeat(self.running_whitening, "feats1 feats2 -> batch feats1 feats2", batch=batch_size),
            )
            return decorrelated
        # Otherwise compute a fresh estimate of sigma but blend it with the
        # running statistics using momentum.  We also compute the whitening
        # matrix via CANS and update running_H.
        m = x.sum(1, keepdim=True) / n[:, None, None] if isinstance(n, torch.Tensor) else x.mean(1, keepdim=True)
        m = (1 - self.momentum) * self.running_mean + self.momentum * m
        xn = x - m
        eye, sigma = self.calc_eye_sigma(xn, w_dim=w_dim, batch_size=batch_size, n=n)
        sigma = (1 - self.momentum) * self.running_covariance[None, :, :] + self.momentum * sigma
        wh_matrix = self.whiten_matrix(sigma=sigma, eye=eye)
        # Update running averages
        self.update_running_statistic("running_mean", m.mean(dim=0))
        self.update_running_statistic("running_covariance", sigma.mean(dim=0))
        self.update_running_statistic("running_whitening", wh_matrix.mean(dim=0))
        decorrelated = torch.bmm(xn, wh_matrix)
        return decorrelated
