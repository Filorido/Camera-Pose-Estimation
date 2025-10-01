import numpy as np

def vtrans(a: np.ndarray, d: int) -> np.ndarray:
    """
    Vec‐transpose operator.
    Trasforma un array a di forma (d*s, n) in un array b di forma (d*n, s).

    Parameters
    ----------
    a : np.ndarray, shape (d*s, n)
        L'array di input.
    d : int
        Il “blocco” di righe: s = (a.shape[0] // d).

    Returns
    -------
    b : np.ndarray, shape (d*n, s)
    """
    # dimensioni
    
    #modifica: rows, n = a.shape
    
    a = np.atleast_2d(a).T  # Forza vettore colonna
    rows, n = a.shape
    if rows % d != 0:
        raise ValueError("Il numero di righe di 'a' deve essere un multiplo di d.")
    s = rows // d

    # output iniziale
    b = np.zeros((d * n, s), dtype=a.dtype)

    if n < s:
        # caso n < s: itera sulle colonne di a
        for i in range(n):
            # estrai la colonna i (lunghezza d*s) e rimodella in (d, s)
            block = a[:, i].reshape(d, s)
            # inseriscila in b nelle righe i*d:(i+1)*d
            b[i*d:(i+1)*d, :] = block

    else:
        # caso n >= s: itera sui blocchi di righe di a
        for i in range(s):
            # estrai il blocco di righe i*d:(i+1)*d e rimodella in vettore (d*n,)
            block = a[i*d:(i+1)*d, :].reshape(d * n)
            # assegnalo alla colonna i di b
            b[:, i] = block

    return b
