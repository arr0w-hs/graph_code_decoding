import re
import math
import requests
import numpy as np
from bs4 import BeautifulSoup

# from chatgpt

def prime_from_q(q: int) -> int:
    """
    CodeTables QECC pages use q = p^2 for additive quantum codes over GF(p^2).
    Examples:
      q=4  -> qubit stabilizer, p=2
      q=9  -> qutrit stabilizer, p=3
      q=25 -> p=5
    """
    p = int(round(math.sqrt(q)))
    if p * p != q:
        raise ValueError(f"Expected q=p^2, got q={q}")
    return p


def fetch_codetables_qecc(n: int, k: int, q: int = 4) -> str:
    """
    Fetch one QECC CodeTables page.
    Example URL:
      https://www.codetables.de/QECC/QECC.php?k=4&n=34&q=9
    """
    url = f"https://www.codetables.de/QECC/QECC.php?k={k}&n={n}&q={q}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.text


def extract_stabilizer_matrix(html: str, n: int, q: int = 4):
    """
    Extract rows of the form:
      [a1 a2 ... an | b1 b2 ... bn]

    Returns:
      Hx, Hz, H
    where H = [Hx | Hz].
    """
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")

    # Find bracketed rows containing a vertical bar.
    row_pattern = re.compile(r"\[([0-9\s]+)\|([0-9\s]+)\]")
    rows = row_pattern.findall(text)

    if not rows:
        raise ValueError(
            "No stabilizer matrix found on this page. "
            "Some CodeTables entries only give bounds or construction notes."
        )

    X_rows = []
    Z_rows = []

    for left, right in rows:
        x = [int(a) for a in left.split()]
        z = [int(b) for b in right.split()]

        if len(x) != n or len(z) != n:
            raise ValueError(
                f"Bad row length: got len(X)={len(x)}, len(Z)={len(z)}, expected n={n}"
            )

        X_rows.append(x)
        Z_rows.append(z)

    p = prime_from_q(q)

    Hx = np.array(X_rows, dtype=int) % p
    Hz = np.array(Z_rows, dtype=int) % p
    H = np.hstack([Hx, Hz]) % p

    return Hx, Hz, H


def check_stabilizer_commutes(Hx, Hz, q: int = 4) -> bool:
    """
    Check symplectic commutation condition.

    For qubits:
      Hx Hz^T + Hz Hx^T = 0 mod 2

    For odd prime p:
      Hx Hz^T - Hz Hx^T = 0 mod p
    """
    p = prime_from_q(q)

    if p == 2:
        symp = (Hx @ Hz.T + Hz @ Hx.T) % 2
    else:
        symp = (Hx @ Hz.T - Hz @ Hx.T) % p

    return np.all(symp == 0)

if __name__ == "__main__":

  from stabiliser_code import find_logical_op_basis
  from galois import GF2

  # Example: [[34,4,10]] over GF(3^2), so q=9 and p=3
  html = fetch_codetables_qecc(n=9, k=1)

  Hx, Hz, H = extract_stabilizer_matrix(html, n=9)

  print("Hx shape:", Hx.shape)
  print("Hz shape:", Hz.shape)
  print("H shape :", H.shape)
  print("commutes:", check_stabilizer_commutes(Hx, Hz))

  print(H)
  H = GF2(H)

  a = find_logical_op_basis(H, 9)
  print(a)