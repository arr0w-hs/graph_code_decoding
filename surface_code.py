import numpy as np
from qecsim.models.rotatedplanar import RotatedPlanarCode

# from AI

def rotated_surface_code(d: int):
    """
    Construct a rotated planar surface code with parameters [[d^2, 1, d]].

    Returns
    -------
    Hx : np.ndarray
        X part of the stabilizer check matrix.

    Hz : np.ndarray
        Z part of the stabilizer check matrix.

    H : np.ndarray
        Binary symplectic stabilizer matrix [Hx | Hz].

    logical_x : np.ndarray
        Binary symplectic representation of logical X.

    logical_z : np.ndarray
        Binary symplectic representation of logical Z.
    """
    if not isinstance(d, int):
        raise TypeError("d must be an integer")

    if d < 3:
        raise ValueError("qecsim requires d >= 3")

    code = RotatedPlanarCode(d, d)

    # H has shape (n-k, 2n) = (d^2 - 1, 2d^2)
    H = np.asarray(code.stabilizers, dtype=np.uint8)

    n = d * d
    Hx = H[:, :n]
    Hz = H[:, n:]

    # One encoded qubit, so one logical X and one logical Z.
    logical_x = np.asarray(code.logical_xs[0], dtype=np.uint8)
    logical_z = np.asarray(code.logical_zs[0], dtype=np.uint8)

    # Internal qecsim validation:
    # stabilizers commute, logicals commute with stabilizers,
    # and logical X anticommutes with logical Z.
    code.validate()

    assert code.n_k_d == (d * d, 1, d)
    assert H.shape == (d * d - 1, 2 * d * d)

    return Hx, Hz, H, logical_x, logical_z