import numpy as np
from PIL import Image, ExifTags

def get_internals(img_path: str, sensor_width_mm: float = 35.0) -> np.ndarray:
    """
    Legge un'immagine JPEG, estrae i metadati EXIF per la lunghezza focale
    espressa come "35mm equivalent", e costruisce la matrice intrinseca K.

    Parametri
    ----------
    img_path : str
        Percorso all'immagine.
    sensor_width_mm : float, opzionale
        Larghezza del sensore in mm (default 35.0).

    Ritorna
    -------
    K : np.ndarray, shape (3,3)
        Matrice intrinseca [[fp, 0, u0],
                              [0, fp, v0],
                              [0,  0,  1]].
    """
    # 1) Leggi l'immagine per ottenere dimensioni
    with Image.open(img_path) as img:
        W, H = img.size  # width, height
        exif = img._getexif() or {}

    # 2) Mappa tag EXIF in nomi leggibili
    tag_map = {ExifTags.TAGS[k]: k for k in ExifTags.TAGS}

    # 3) Estrai FocalLengthIn35mmFilm o, in alternativa, FocalLength
    focal_mm = None
    if 'FocalLengthIn35mmFilm' in tag_map:
        tag_35 = tag_map['FocalLengthIn35mmFilm']
        val = exif.get(tag_35)
        if val is not None:
            # Spesso è un tuple (num, den)
            focal_mm = val[0] / val[1] if isinstance(val, tuple) else float(val)

    if focal_mm is None and 'FocalLength' in tag_map:
        tag_f = tag_map['FocalLength']
        val = exif.get(tag_f)
        if val is not None:
            focal_mm = val[0] / val[1] if isinstance(val, tuple) else float(val)

    if focal_mm is None:
        raise ValueError("Lunghezza focale non trovata nei metadati EXIF.")

    # 4) Calcola fp in pixel
    fp = (focal_mm * W) / sensor_width_mm

    # 5) Centro principale
    u0 = W / 2.0
    v0 = H / 2.0

    # 6) Costruisci K
    K = np.array([
        [fp,  0,  u0],
        [ 0, fp,  v0],
        [ 0,  0,   1]
    ], dtype=float)

    return K
