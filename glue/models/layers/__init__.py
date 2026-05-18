"""
Expose commonly used layer classes at the package level.

Importing these classes into ``__init__.py`` makes them available as
``models.layers.Whitening2d``, ``models.layers.WhiteningCANS2d`` and so on.
"""

from .whitening import (
    Whitening2d,
    WhiteningSing2dIterNorm,
    WhiteningMatrixSign2dIterNorm,
    WhiteningTrace2dIterNorm,
)

# CANS‑based whitening layer
from .cans_whitening import WhiteningCANS2d

# Additional specialized layers
# Import the butterfly multiply class with the correct name.  The
# implementation defines ``BlockdiagButterflyMultiply`` (note the
# lowercase 'd' in ``diag``), so import that symbol instead of
# ``BlockDiagonalButterflyMultiply``.  See
# ``blockdiag_butterfly_multiply.py`` for details.
from .blockdiag_butterfly_multiply import BlockdiagButterflyMultiply
from .gs_orthogonal import GSOrthogonal
from .gsoft import GSOFTLayer
