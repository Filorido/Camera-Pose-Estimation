import numpy as np
from scipy.spatial.transform import Rotation

def set_unity_cam(Iw: int, Ih: int,
                  K: np.ndarray,
                  R: np.ndarray,
                  t: np.ndarray,
                  sensor_x_mm: float = 35.0):
    """
    Calcola i parametri da inserire in Unity (Physical Camera e Transform)
    a partire da K, R, t standard.

    Parameters
    ----------
    Iw, Ih : int
        Larghezza e altezza dell'immagine (in pixel).
    K : array (3,3)
        Matrice intrinseca.
    R : array (3,3)
        Matrice di rotazione camera->world.
    t : array (3,) or (3,1)
        Vettore di traslazione camera->world.
    sensor_x_mm : float
        Ampiezza del sensore in mm (default 35.0).

    Returns
    -------
    fmm : float
        Focale equivalente in mm per Unity.
    sensor_x_mm : float
        Valore di Sensor X (mm).
    sensor_y_mm : float
        Sensor Y (mm) calcolato.
    ls_x, ls_y : float
        Lens shift X e Y per Unity.
    euler_unity : array (3,)
        Angoli di rotazione [X, Y, Z] in gradi (ordine Unity).
    T_u : array (3,)
        Posizione della camera in Unity (world space).
    """

    # 1. Estrai parametri intrinseci
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # 2. Calcola focal length in mm
    fmm = fx * (sensor_x_mm / Iw)

    # 3. Calcola SensorY a partire da fmm e fy
    sensor_y_mm = fmm * (Ih / fy)

    # 4. Lens shifts normalizzati (-0.5..0.5 circa)
    ls_x =  (cx - (Iw/2)) / Iw
    ls_y =  (cy - (Ih/2)) / Ih

    # 5. Trasforma extrinsics per il sistema di Unity
    #    - specchio sull'asse Y (Sy)
    #    - swap Y<->Z (YZ)
    Sy = np.diag([1, -1, 1])
    YZ = np.array([[1,0,0],
                   [0,0,1],
                   [0,1,0]], float)

    # camera->world in Unity
    R_u = YZ @ (Sy @ R).T
    # posizione
    t = t.reshape(3,)
    T_u = YZ @ (-(Sy @ R).T @ (Sy @ t))

    # 6. Estrazione degli angoli di Eulero in ordine Z-X-Y
    #    Unity applica rotazione: Z poi X poi Y.
    #    Usiamo SciPy con `as_euler('ZXY')` che restituisce [z, x, y].
    #    Poi rechordiamo in [x, y, z] e convertiamo in gradi.
    # Per conformità alla riorganizzazione MATLAB:
    #  R_eun  = [R_u[:,2], R_u[:,0], R_u[:,1]] come colonne
    #  R_eun1 = riga3, riga1, riga2
    R_eun = np.column_stack((R_u[:,2], R_u[:,0], R_u[:,1]))
    R_eun1 = np.vstack((R_eun[2,:],
                        R_eun[0,:],
                        R_eun[1,:]))

    rot = Rotation.from_matrix(R_eun1)
    z_x_y = rot.as_euler('ZXY', degrees=False)  # [z, x, y] in rad
    # ricomponi [x, y, z] in gradi
    euler_unity = np.rad2deg([z_x_y[2], z_x_y[0], z_x_y[1]])

    return fmm, sensor_x_mm, sensor_y_mm, ls_x, ls_y, euler_unity, T_u
