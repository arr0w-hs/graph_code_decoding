import numpy as np
from qecsim.models.basic import FiveQubitCode, SteaneCode
from qecsim.models.color import Color666Code
from qecsim.models.planar import PlanarCode
from qecsim.models.rotatedplanar import RotatedPlanarCode
from qecsim.models.toric import ToricCode
from qecsim.models.rotatedtoric import RotatedToricCode
import tableau as ta
from typing import Literal


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

# ---- BB code constructor (self-contained) -----------------------------------
def _shift(N):
    S=np.zeros((N,N),dtype=int)
    for i in range(N): S[i,(i+1)%N]=1
    return S

def bb_tableau(lm_tuple:tuple):
    """Builds the bivariate bicylce codes from https://arxiv.org/pdf/2308.07915. Returns the stabiliser tableau T"""

    if lm_tuple not in [(3,3), (3,6),(6,6),(9, 6),(6,12),(12,6),(12,12),(15,3)]:
        raise ValueError('lm_tuple must be one of the following: {[(3,3), (3,6),(6,6),(9, 6),(6,12),(12,6),(12,12)]}')

    BB_FAMILY= {
    # (l, m): (A_terms, B_terms)   -- IBM-style series, k=12, treewidth Theta(n)
    (6, 6):  ([('x', 3), ('y', 1), ('y', 2)], [('y', 3), ('x', 1), ('x', 2)]),
    (9, 6):  ([('x',3),('y',1),('y',2)], [('y',3),('x',1),('x',2)]),
    (15,3): ([('x',9),('y',1),('y',2)], [('x',0),('x',2),('x',7)]),
    (6, 12): ([('x', 3), ('y', 1), ('y', 2)], [('y', 3), ('x', 1), ('x', 2)]),
    (12, 12):([('x', 3), ('y', 2), ('y', 7)], [('y', 3), ('x', 1), ('x', 2)]),
    (12, 6): ([('x', 3), ('y', 1), ('y', 2)], [('y', 3), ('x', 1), ('x', 2)]),
    (3, 3): ([('x',3),('y',1),('y',2)], [('y',3),('x',1),('x',2)]),   # n=18, k=8
    (3, 6): ([('x',3),('y',1),('y',2)], [('y',3),('x',1),('x',2)])}  # n=36, k=8

    A_terms=BB_FAMILY[lm_tuple][0]
    B_terms=BB_FAMILY[lm_tuple][1]
    l=lm_tuple[0]
    m=lm_tuple[1]
    Il,Im=np.eye(l,dtype=int),np.eye(m,dtype=int);
    Sl,Sm=_shift(l),_shift(m)
    xp=lambda k: np.kron(np.linalg.matrix_power(Sl,k%l)%2,Im)%2
    yp=lambda k: np.kron(Il,np.linalg.matrix_power(Sm,k%m)%2)%2
    def bld(ts):
        M=np.zeros((l*m,l*m),dtype=int)
        for v,k in ts: M=(M+(xp(k) if v=='x' else yp(k)))%2
        return M
    A,B=bld(A_terms),bld(B_terms)
    HX=np.hstack([A,B])%2; HZ=np.hstack([B.T,A.T])%2
    z=np.zeros_like(HX)
    T=np.vstack([np.hstack([HX,z]), np.hstack([z,HZ])]).astype(int)
    T=ta.to_gf2_tableau(T).row_reduce()
    T=T[np.any(np.asarray(T),axis=1)] #keep only non-zero rows of T
    return None,None,T, None, None #this does not return logical x and z! Will be computed later, this is just to match the output
    #of the other functions