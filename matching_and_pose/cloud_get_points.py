from plyfile import PlyData
import numpy as np

def cloud_get_points(zephyr_ply_file: str,
                     visibility_point_file: str,
                     img_name: str):
    """
    Analizza l'output di Zephyr:
    - Legge la cloud di punti da un file .ply
    - Cerca nel file di visibilità solo la sezione di `img_name`
      ed estrae gli indici e le coordinate 2D

    Parametri
    ----------
    zephyr_ply_file : str
        Percorso al file .ply (sparse point cloud).
    visibility_point_file : str
        Percorso al file di visibilità (testuale).
    img_name : str
        Basename dell'immagine di cui estrarre la visibilità
        (es. '20250124_113557.jpg').

    Ritorna
    -------
    p2D : np.ndarray, shape (n_points, 2)
        Coordinate 2D dei punti visibili per `img_name`.
    p3D : np.ndarray, shape (n_points, 3)
        Coordinate 3D corrispondenti nella cloud.
    """
    # 1) Lettura del .ply
    
    plydata = PlyData.read(zephyr_ply_file)
    vertex = plydata['vertex']
    X = np.vstack((vertex['x'], vertex['y'], vertex['z'])).T  # shape (N,3)

    # 2) Lettura del file di visibilità, sezione img_name
    ids = []
    coords2D = []
    with open(visibility_point_file, 'r') as f:
        lines = f.readlines()

    current = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # inizio di una sezione
        if line.startswith("Visibility for camera"):
            current = line.split("Visibility for camera",1)[1].strip()
            # salta la riga del count
            i += 2
            continue

        # se siamo nella sezione desiderata, leggi
        if current == img_name and line:
            parts = line.split()
            if len(parts) == 3:
                try:
                    idx = int(parts[0])
                    x2d = float(parts[1])
                    y2d = float(parts[2])
                    ids.append(idx)
                    coords2D.append((x2d, y2d))
                except ValueError:
                    pass
        i += 1

    if not ids:
        raise ValueError(f"Immagine '{img_name}' non trovata in {visibility_point_file}")

    p2D = np.array(coords2D, dtype=np.float32)   # (n,2)
    
    # 3) Estrazione dei punti 3D corrispondenti
    indices = np.array(ids, dtype=int)
    p3D = X[indices, :]                          # (n,3)

    return p2D, p3D
