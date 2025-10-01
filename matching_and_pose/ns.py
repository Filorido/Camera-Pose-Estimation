import numpy as np
import warnings

def ns(A: np.ndarray) -> np.ndarray:
    """
    Risolve il problema del nullo A @ v = 0, restituendo
    il vettore v che span la null-space di A (nullity = 1).
    
    Se la condizione numerica è troppo alta (>200) viene emesso un warning.
    """
    # Calcola la SVD di A
    # numpy.linalg.svd restituisce U, s (vettore di valori singolari), Vh (V trasposto)
    U, s, Vh = np.linalg.svd(A)
    # Ricava V
    V = Vh.T
    
    # Controllo del condition number secondo lo script MATLAB:
    # c = D(1) / D(end-1)
    # dove D(1) = s[0], D(end-1) = s[-2]
    if s.size < 2:
        # non ha senso, ma evitiamo IndexError
        c = np.inf
    else:
        c = s[0] / s[-2]
    if c > 200:
        warnings.warn(f"ns: condition number is {c:.0f}", UserWarning)
    
    # Il null-space vector è l'ultima colonna di V
    v = V[:, -1]
    return v
