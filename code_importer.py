import numpy as np
from qecsim.models.basic import FiveQubitCode, SteaneCode
from qecsim.models.color import Color666Code
from qecsim.models.planar import PlanarCode
from qecsim.models.rotatedplanar import RotatedPlanarCode
from qecsim.models.toric import ToricCode
from qecsim.models.rotatedtoric import RotatedToricCode


def make_qecsim_code(name: str, d: int = 3):
    name = name.lower().replace("-", "_").replace(" ", "_")

    constructors = {
        "five_qubit": lambda: FiveQubitCode(),
        "steane": lambda: SteaneCode(),
        "color": lambda: Color666Code(d),
        "color_666": lambda: Color666Code(d),
        "planar": lambda: PlanarCode(d, d),
        "surface": lambda: PlanarCode(d, d),
        "rotated_planar": lambda: RotatedPlanarCode(d, d),
        "rotated_surface": lambda: RotatedPlanarCode(d, d),
        "toric": lambda: ToricCode(d, d),
        "rotated_toric": lambda: RotatedToricCode(d, d),
    }

    try:
        return constructors[name]()
    except KeyError:
        available = ", ".join(sorted(constructors))
        raise ValueError(
            f"Unknown code {name!r}. Available codes: {available}"
        )
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


def surface_code(d: int):
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

    code = PlanarCode(d, d)

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

    # assert code.n_k_d == (d * d, 1, d)
    # assert H.shape == (d * d - 1, 2 * d * d)

    return Hx, Hz, H, logical_x, logical_z