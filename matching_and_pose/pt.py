import numpy as np

def p2t(H: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Applica un'omografia 2D H (3x3) a un insieme di punti m in forma 2×N o N×2.
    
    Parametri
    ----------
    H : array shape (3,3)
      Matrice di omografia proiettiva 2D.
    m : array shape (2,N) o (N,2)
      Punti cartesiani 2D.
    
    Ritorna
    -------
    mt : array shape (2,N) o (N,2)
      Punti trasformati, nella stessa orientazione di input.
    """
    H = np.asarray(H)
    m = np.asarray(m)
    # Controllo forma H
    if H.shape != (3, 3):
        raise ValueError("Formato errato della matrice di trasformazione (3x3)!!")
    
    # Preparo i punti in forma 2×N
    transpose_back = False
    if m.ndim != 2 or (m.shape[0] != 2 and m.shape[1] != 2):
        raise ValueError("Le coordinate immagine devono essere shape (2,N) o (N,2).")
    if m.shape[1] == 2:
        m = m.T
        transpose_back = True
    
    # Omogenee 3×N
    ones = np.ones((1, m.shape[1]))
    c3d = np.vstack((m, ones))        # 3×N
    # Proiezione
    h2d = H @ c3d                     # 3×N
    # Normalizzazione: divido le prime due righe per la terza
    mt_h = h2d[:2, :] / h2d[2:3, :]
    
    mt = mt_h
    return mt.T if transpose_back else mt


def p3t(T: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Applica una trasformazione proiettiva 3D T (4x4) a un insieme di punti w in forma 3×N o N×3.
    
    Parametri
    ----------
    T : array shape (4,4)
      Matrice di trasformazione omogenea 3D.
    w : array shape (3,N) o (N,3)
      Punti cartesiani 3D.
    
    Ritorna
    -------
    wt : array shape (3,N) o (N,3)
      Punti trasformati, nella stessa orientazione di input.
    """
    T = np.asarray(T)
    w = np.asarray(w)
    # Controllo forma T
    if T.shape != (4, 4):
        raise ValueError("Formato errato della matrice di trasformazione (4x4)!!")
    
    # Preparo i punti in forma 3×N
    transpose_back = False
    if w.ndim != 2 or (w.shape[0] != 3 and w.shape[1] != 3):
        raise ValueError("Le coordinate mondo devono essere shape (3,N) o (N,3).")
    if w.shape[1] == 3:
        w = w.T
        transpose_back = True
    
    # Omogenee 4×N
    ones = np.ones((1, w.shape[1]))
    tmp = T @ np.vstack((w, ones))   # 4×N
    # Normalizzazione: divido le prime tre righe per la quarta
    wt_h = tmp[:3, :] / tmp[3:4, :]
    
    wt = wt_h
    return wt.T if transpose_back else wt

def pt(H: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Applica una trasformazione proiettiva 2D o 3D.
    
    Se H è 3×3, richiama p2t;  
    se H è 4×4, richiama p3t;  
    altrimenti solleva un errore.
    
    Parametri
    ----------
    H : np.ndarray, shape (3,3) o (4,4)
        Matrice di omografia del piano o dello spazio.
    m : np.ndarray
        Punti da trasformare (shape coerente con H).
    
    Ritorna
    -------
    mt : np.ndarray
        Punti trasformati, stessa orientazione di input.
    """
    if H.shape == (3, 3):
        return p2t(H, m)
    elif H.shape == (4, 4):
        return p3t(H, m)
    else:
        raise ValueError("Trasformazione non implementata: H deve essere 3×3 o 4×4.")
