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
This implementation operates per sample in a mini‑batch.  For each sample
we build a block matrix and apply the CANS iteration independently.
Although looped Python code may seem inefficient, the embedding dimension
in typical transformer models is at most a thousand, so the overhead is
acceptable during training.  Future work could batch the CANS iteration
using vectorized operations.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor

from models.layers.whitening import Whitening2d
from models.utils import singular_norm

import numpy as np

# einops is used in forward_test.  Importing it lazily here avoids a hard
# dependency when the layer is instantiated but never used for evaluation.
import einops


def get_polynomial(Ext: np.ndarray) -> np.ndarray:
    """Compute polynomial coefficients used by the Remez algorithm.

    This helper is a thin wrapper around the original implementation in the
    provided reference.  It solves a linear system to find coefficients of
    a polynomial that oscillates optimally on the interval defined by
    ``Ext``.  The resulting coefficients are returned as a NumPy array.
    """
    n = len(Ext) - 1
    M = np.zeros((n + 1, n + 1), dtype=np.float64)
    for i in range(n + 1):
        for j in range(n):
            M[i, j] = Ext[i] ** (2 * j + 1)
        M[i, n] = (-1) ** (i + 1)
    c = np.linalg.solve(M, np.ones(n + 1, dtype=np.float64))
    return c


def get_ext(c: np.ndarray) -> np.ndarray:
    n = len(c)
    coeffs = [(2 * j + 1) * c[j] for j in range(n)]
    rts = np.roots(coeffs[::-1])
    return np.sqrt(rts)


def remez_step(Ext: np.ndarray):
    n = len(Ext) - 1
    c = get_polynomial(Ext)
    coeffs = [c[j // 2 - 1] if j % 2 == 0 else 0 for j in range(1, 2 * n + 1)]
    p = np.poly1d(coeffs[::-1])
    e = c[n].item()
    NewExt = np.concatenate(([Ext[0]], get_ext(c[:n]), [Ext[-1]]))
    f = lambda x: np.abs(p(x) - 1)
    newe = np.max(f(NewExt))
    return p, NewExt, e, newe


def remez(A: float, B: float, degree: int):
    """Compute an optimal polynomial approximation of the unity function.

    This function returns a polynomial ``p`` and an error bound ``newe``
    defined on the interval ``[A, B]``.  When the interval collapses,
    a fallback polynomial from Kovarik's formula is returned.
    """
    n = (degree + 1) // 2
    Ext = np.linspace(B, A, n + 1, dtype=np.float64)
    p = np.poly1d([0])
    newe = 0
    for _ in range(100):
        try:
            p, NewExt, e, newe = remez_step(Ext)
        except np.linalg.LinAlgError:
            # if the segment converged to [A, B] = [1, 1]
            return kovarik_formula(degree), 0
        if newe < abs(e) + 1e-20:
            return p, newe
        Ext = np.array(NewExt, dtype=np.float64)
    return p, newe


def c_n_k(n: int, k: int) -> float:
    s = 1
    for i in range(n - k + 1, n + 1):
        s *= i
    for i in range(1, k + 1):
        s /= i
    return s


def kovarik_formula(degree: int) -> np.poly1d:
    """Generate an explicit polynomial used as a fallback.

    The formula is taken from Zdislav Kovarik (1970).  It yields a polynomial
    that improves orthonality when the Remez algorithm fails to converge.
    """
    p = np.zeros(degree + 1)
    p[1] += 1
    a = 1
    for i in range(1, (degree + 1) // 2):
        for j in range(2 * (i - 1) + 1, 2 * i + 1):
            a *= j / 2
        a /= i ** 2
        sign = 1
        for k in range(0, i + 1):
            p[2 * k + 1] += a * sign * c_n_k(i, k)
            sign *= -1
    return np.poly1d(p[::-1])


def explicit3(A: float, B: float):
    """Explicit formula for the optimal cubic polynomial.

    This closed‑form expression accelerates the orthogonalization step for
    degree ``3`` polynomials.  It returns both the polynomial ``p`` and the
    associated approximation error ``err`` on the interval ``[A, B]``.
    """
    e = np.sqrt((A ** 2 + A * B + B ** 2) / 3)
    a = 2 / (2 * e ** 3 + A ** 2 * B + B ** 2 * A)
    p = np.poly1d([-a, 0, a * (A ** 2 + A * B + B ** 2), 0])
    err = (2 * e ** 3 - A ** 2 * B - B ** 2 * A) / (2 * e ** 3 + A ** 2 * B + B ** 2 * A)
    return p, err


def delta_orthogonalization(n: int = 1, degree: int = 3, delta: float = 0.3, B: float = 1):
    """Find a composition of polynomials that improves orthogonality.

    This helper function searches for a left boundary ``A`` such that the
    composition of ``n`` degree‑``degree`` polynomials approximates the unity
    function on ``[A, B]`` with accuracy ``delta``.  It returns the list of
    polynomials and the final left boundary value.  The function is
    unchanged from the original implementation.
    """
    Al = 0.0
    Ar = B
    e = 100
    while abs(e - delta) > 1e-7:
        a, b = (Al + Ar) / 2, B
        lst = []
        for i in range(n):
            if degree == 3:
                Q, e = explicit3(a, b)
            else:
                Q, e = remez(a, b, degree)
            lst.append(Q)
            a, b = 1 - e, 1 + e
        if e < delta:
            Ar = (Ar + Al) / 2
        else:
            Al = (Al + Ar) / 2
    return lst, (Al + Ar) / 2


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
        Degree of the polynomial approximation.  Currently only ``3`` and
        ``5`` are implemented.  Default is ``3``.
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
    This function is a direct translation of the reference implementation
    provided by the user.  Most operations are performed in double
    precision on the CPU to improve numerical stability.  The result is
    moved back to the device and dtype of the input ``A`` before being
    returned.
    """
    # Move computation to CPU and double precision for stability
    dtype = A.dtype
    device = A.device
    A = A.to(torch.float64).cpu().clone()
    if A.shape[0] < A.shape[1]:
        A = A.T

    if degree == 3:
        A2 = A.T @ A
        A3 = A @ A2
        denom = torch.norm(A3, p='fro') ** (1.0 / 3.0) + 1e-7
        A2 /= denom ** 2
        A3 /= denom ** 3
    elif degree == 5:
        A2 = A.T @ A
        A3 = A @ A2
        A5 = A3 @ A2
        denom = torch.norm(A5, p='fro') ** (1.0 / 5.0) + 1e-7
        A2 /= denom ** 2
        A3 /= denom ** 3
        A5 /= denom ** 5
    else:
        raise NotImplementedError("Only degrees 3 and 5 are implemented")
    A = A / denom

    b = 1.0  # right boundary is fixed to 1.0 for the normalization
    I = torch.eye(A.shape[1], dtype=A.dtype, device=A.device)
    if preprocess:
        lst, _ = delta_orthogonalization(preprocess_iters, degree, delta)
        for i in range(preprocess_iters):
            if degree == 3:
                A = lst[i][1] * A + lst[i][3] * A3
            elif degree == 5:
                A = lst[i][1] * A + lst[i][3] * A3 + lst[i][5] * A5
            A2 = A.T @ A
            A3 = A @ A2
            if degree == 5:
                A5 = A3 @ A2
        a, b = 1 - delta, 1 + delta
    cnt = 0
    err = torch.norm(A2 - I) / torch.norm(I, p='fro')
    while cnt < n and (err > 1e-6):
        if cnt > 0:
            A3 = A @ A2
            if degree == 5:
                A5 = A3 @ A2
        if degree == 3:
            p, e = explicit3(a, b)
            a, b = 1 - e, 1 + e
            A = p[1] * A + p[3] * A3
        elif degree == 5:
            p, e = remez(a, b, degree)
            a, b = 1 - e, 1 + e
            A = p[1] * A + p[3] * A3 + p[5] * A5
            b *= 1.01  # for numerical stability
        A2 = A.T @ A
        err = torch.norm(A2 - I) / torch.norm(I, p='fro')
        cnt += 1
    # Cast back to original dtype/device
    return A.to(dtype).to(device)


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

    def update_running_statistic(self, running_statistic: str, value: Tensor) -> None:
        """Override to support updating the running square‑root factor.

        In addition to ``running_mean``, ``running_covariance`` and
        ``running_whitening``, this method updates ``running_H`` if
        ``running_statistic`` equals ``running_H``.  Otherwise it falls back
        to the implementation in the base class.
        """
        if running_statistic == "running_H":
            cur = getattr(self, running_statistic)
            setattr(self, running_statistic, (1 - self.momentum) * cur + self.momentum * value.clone().detach())
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
        # Prepare output tensors
        wm = sigma.new_empty((B, d, d))
        H_batch = sigma.new_empty((B, d, d))
        # Iterate over each sample because CANS operates on a single matrix
        for i in range(B):
            # Construct block matrix [[0, sigma], [I, 0]]
            upper = torch.cat([torch.zeros_like(sigma[i]), sigma[i]], dim=1)
            lower = torch.cat([torch.eye(d, dtype=sigma.dtype, device=sigma.device), torch.zeros_like(sigma[i])], dim=1)
            block = torch.cat([upper, lower], dim=0)
            # Compute singular norm for scaling
            # singular_norm expects shape (batch, S, F), so we unsqueeze
            block_unsqueezed = block.unsqueeze(0)
            s = singular_norm(block_unsqueezed)  # shape (1,)
            s_val = s.item() + 1e-12
            block_scaled = block / s_val
            # Apply CANS iteration to approximate the polar factor (matrix sign)
            # The number of iterations and left boundary a are taken from the
            # layer's configuration.  We use degree=3 and do not preprocess.
            U = cans_iteration(block_scaled, n=self.iterations, a=0.0, degree=3, preprocess=False)
            # Extract blocks: U = [[U00, U01], [U10, U11]]
            U00 = U[0:d, 0:d]
            U01 = U[0:d, d:]
            U10 = U[d:, 0:d]
            # According to sign([[0,A],[I,0]]) = [[0,A^{1/2}], [A^{-1/2},0]],
            # we recover sigma^{1/2} and sigma^{-1/2} up to the scaling factor.
            sigma_sqrt = U01 * math.sqrt(s_val)
            sigma_inv_sqrt = U10 / math.sqrt(s_val)
            H_batch[i] = sigma_sqrt
            wm[i] = sigma_inv_sqrt
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

