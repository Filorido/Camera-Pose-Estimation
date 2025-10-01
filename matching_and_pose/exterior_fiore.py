import numpy as np
import absolute 
import ns
import pt
from numpy.linalg import svd, matrix_rank, inv
import vtrans

def exterior_fiore(A, model3d, data2d):
    """
    Calcola l'orientamento esterno usando l'algoritmo di Fiore.

    Parametri:
        A: matrice 3x3 di intrinseci (deve essere normalizzata: A[2,2] == 1)
        model3d: punti 3D (shape: 3xN)
        data2d: punti immagine (shape: 2xN)

    Restituisce:
        G: matrice rigida 3x4 (rotazione + traslazione)
        s: fattore di scala
    """
    if not np.isclose(A[2, 2], 1.0):
        raise ValueError("La matrice A deve essere normalizzata (A[2,2] == 1)")

    # Coordinate immagine normalizzate
    m = pt.pt(np.linalg.inv(A), data2d)

    # Omogenee
    m = np.vstack([m, np.ones((1, m.shape[1]))])  # 3xN

    # Aggiunge riga di 1 anche a model3d
    S = np.vstack([model3d, np.ones((1, model3d.shape[1]))])  # 4xN

    # SVD su S
    U, X, Vt = svd(S)
    i = matrix_rank(S)
    V = Vt.T
    V2 = V[:, i:]

    numP = data2d.shape[1]

    # Costruzione matrice D
    D_blocks = []
    for i in range(numP):
        col = np.zeros((3, numP))
        col[:, i] = m[:, i]
        D_blocks.append(col)
    D = np.vstack(D_blocks)  # (3*numP)xN

    # L = kron(V2'.T, eye(3)) @ D
    L = np.kron(V2.T, np.eye(3)) @ D

    # Soluzione del sistema Lz = 0
    z = ns.ns(L)  # trova il vettore nullo

    # Correggi segno sulla profondità
    z = z * np.sign(z[0])

    # Applica la trasformazione rigida
    M = vtrans.vtrans(D @ z, 3)
    
    flat = M.flatten()
    reshaped = flat.reshape(-1, 3)
    
    G, s, _ = absolute.absolute(reshaped, model3d, method='scale')

    return G, s