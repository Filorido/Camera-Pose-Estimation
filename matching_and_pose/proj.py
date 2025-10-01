import numpy as np

def proj(P: np.ndarray, c3d: np.ndarray):
    """
    Calcola la proiezione prospettica di punti 3D in coordinate pixel.

    Parametri
    ----------
    P : np.ndarray, shape (3,4)
        Matrize di proiezione (camera matrix).
    c3d : np.ndarray, shape (N,3)
        Coordinate dei punti 3D (righe).

    Ritorna
    -------
    u, v : np.ndarray, shape (N,)
        Coordinate pixel arrotondate di ciascun punto.
    """
    # Controlli preliminari
    if P.shape != (3, 4):
        raise ValueError("P deve essere di forma (3,4)")
    if c3d.ndim != 2 or c3d.shape[1] != 3:
        raise ValueError("c3d deve essere di forma (N,3)")

    # Trasforma in omogenee 4×N
    ones = np.ones((c3d.shape[0], 1))
    h3d = np.hstack((c3d, ones)).T  # 4×N

    # Applica la proiezione: 3×4 @ 4×N -> 3×N
    h2d = P @ h3d                  # 3×N

    # Normalizzazione in coordinate cartesiane
    x = h2d[0, :] / h2d[2, :]
    y = h2d[1, :] / h2d[2, :]

    # Arrotonda ai pixel interi e restituisci vettori colonna
    u = np.round(x).astype(int)
    v = np.round(y).astype(int)

    return u, v
