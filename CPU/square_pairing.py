"""Square-lattice pairing symmetry channels."""

from CPU.lattice_structure import SQUARE_BONDS


def pairing_channels(shell, direction, coupling):
    """Resolve the even singlet and odd triplet fields on the two forward bonds."""
    first, second = SQUARE_BONDS[shell-1]
    s1, p1 = direction(*first)
    s2, p2 = direction(*second)
    d_name = "d_xy" if shell == 2 else "d_x2-y2"
    # On the diagonal shell, first and second point NE and SE.
    px, py = ((p1 + p2) / 2, (p1 - p2) / 2) if shell == 2 else (p1, p2)
    return {
        f"{shell}NN extended s": coupling * (s1 + s2) / 2,
        f"{shell}NN {d_name}": coupling * (s1 - s2) / 2,
        f"{shell}NN p_x": coupling * px,
        f"{shell}NN p_y": coupling * py,
    }
