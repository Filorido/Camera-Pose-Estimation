import numpy as np

def rigid_transform(G: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Applica una trasformazione rigida G a un insieme di punti w.
    
    Parametri:
    - G: matrice di trasformazione rigida 3x4 (o 4x4, si prende solo la parte 3x4)
    - w: array numpy di shape (3, N), punti 3D in coordinate cartesiane

    Ritorna:
    - wr: punti trasformati (3, N)
    """
    G = np.asarray(G)
    w = np.asarray(w)

    if G.shape[1] != 4 or (G.shape[0] != 3 and G.shape[0] != 4):
        raise ValueError("Formato errato della matrice di trasformazione: deve essere 3x4 o 4x4")

    if G.shape[0] == 4:
        G = G[:3, :]

    if w.shape[0] != 3:
        raise ValueError("Le coordinate dei punti devono essere cartesiane, shape (3, N)")

    # Aggiunge una riga di 1 per trasformare in coordinate omogenee
    HM = np.vstack((w, np.ones((1, w.shape[1]))))

    # Applica la trasformazione rigida
    wr = G @ HM
    return wr


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcola il Root Mean Square Error (RMSE) tra due array x e y.

    Parametri:
    - x, y: array numpy (vettori o matrici) della stessa forma

    Ritorna:
    - e: valore RMSE
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x e y devono avere la stessa forma")

    diff = x - y
    e = np.linalg.norm(diff, 'fro') / np.sqrt(x.size - 1)
    return e

def absolute(X, Y, method='noscale'):
    """
    Risolve l'orientamento assoluto tra due insiemi di punti 3D.
    
    X, Y: array Nx3 di punti corrispondenti
    method: 'noscale' (default) o 'scale'
    
    Restituisce:
    - G: matrice 3x4 [R | t]
    - s: scala (se richiesta)
    - res: errore RMSE (se richiesto)
    """

    # Reshape Nx3
    X = X
    Y = Y.T
    n = Y.shape[0]

    # Calcola i centroidi
    cm = np.sum(Y, axis=0) / n
    cd = np.sum(X, axis=0) / n

    # Sottrai i centroidi usando trasformazioni rigide
    T_cm = np.vstack((np.eye(3), -cm)).T
    T_cd = np.vstack((np.eye(3), -cd)).T

    Yb = rigid_transform(T_cm, Y.T).T
    Xb = rigid_transform(T_cd, X.T).T

    # Scala
    if method == 'scale':
        s = np.linalg.norm(Xb) / np.linalg.norm(Yb)
    elif method == 'noscale':
        s = 1
    else:
        raise ValueError("Metodo non valido. Usa 'scale' o 'noscale'.")

    Xb_scaled = s * Xb
    cd_scaled = cd * (1/s)

    # Rotazione tramite SVD
    K = Xb_scaled.T @ Yb
    U, _, Vt = np.linalg.svd(K)
    S = np.diag([1, 1, np.linalg.det(U @ Vt)])
    R = U @ S @ Vt

    # Traslazione
    t = cd_scaled.reshape(-1, 1) - R @ cm.reshape(-1, 1)

    # Composizione della trasformazione
    G = np.hstack((R, t))

    # Calcolo RMSE
    res = rmse(X, rigid_transform(G, Y.T).T / s)

    return G, s, res
