import numpy as np
from plyfile import PlyData
import re

def plyread(path: str, mode: str = None):
    """
    Legge un file PLY (versione 1.0).
    
    Parametri
    ----------
    path : str
        Percorso al file .ply.
    mode : str, opzionale
        Se 'tri' o 'Tri', restituisce (tri, pts, data, comments).
        Altrimenti restituisce (data, comments).
    
    Ritorna
    -------
    Se mode is None:
      data : dict
        Mappa ogni elemento PLY al suo contenuto, 
        es. data['vertex']['x'] è un array numpy.
      comments : list of str
        Commenti letti dall'header.
    
    Se mode == 'tri':
      tri : np.ndarray, shape (M,3)
        Indici dei vertici di ciascun triangolo (1-based).
      pts : np.ndarray, shape (N,3)
        Coordinate dei vertici.
      data : come sopra
      comments : come sopra
    """
    # 1) Apri e leggi l'header
    with open(path, 'rb') as f:
        # Leggi fino a "end_header"
        header = []
        comments = []
        while True:
            line = f.readline().decode('utf-8').rstrip('\r\n')
            header.append(line)
            if line.startswith('comment '):
                comments.append(line[len('comment '):])
            if line == 'end_header':
                break
        # Posizione dati
        data_start = f.tell()
    
    # 2) Usa plyfile.PlyData per il parsing
    ply = PlyData.read(path)
    
    # 3) Estrai i dati in un dizionario
    data = {}
    for element in ply.elements:
        name = element.name
        props = {prop.name: element.data[prop.name] for prop in element.properties}
        data[name] = props
    
    # 4) Se non servono triangoli, restituisci subito
    if mode is None:
        return data, comments
    
    # 5) ALTRIMENTI estrai vertici e facce e crea tri, pts
    # Individua il nome dell'elemento vertex
    vert_names = ['vertex', 'Vertex', 'point', 'Point', 'pts', 'Pts']
    vert_key = next((k for k in data if k in vert_names), None)
    if vert_key is None:
        raise ValueError("Elemento 'vertex' non trovato nel PLY.")
    vdict = data[vert_key]
    # Costruisci array pts Nx3
    pts = np.vstack((vdict['x'], vdict['y'], vdict['z'])).T
    
    # Cerca l'elemento face
    face_names = ['face', 'Face', 'poly', 'Poly', 'tri', 'Tri']
    face_key = next((k for k in data if k in face_names), None)
    if face_key is None:
        # niente facce → restituisci solo pts
        return None, pts, data, comments
    
    # Identifica il campo con gli indici dei vertici
    prop_names = list(data[face_key].keys())
    idx_names = ['vertex_indices','vertex_indexes','vertex_index','indices','indexes']
    idx_key = next((n for n in prop_names if n in idx_names), None)
    if idx_key is None:
        # nessuna lista di indici → restituisci come sopra
        return None, pts, data, comments
    
    # data[face_key][idx_key] è una lista di array per ogni faccia
    faces_list = data[face_key][idx_key]  # è array di dtype=object
    # Converti in array di triangoli
    tri = []
    for face in faces_list:
        # Se la faccia ha >3 vertici, fan‑triangola
        if len(face) == 3:
            tri.append(face)
        else:
            # fan‑triangulation: (v0,v1,v2), (v0,v2,v3), ...
            for i in range(1, len(face)-1):
                tri.append([face[0], face[i], face[i+1]])
    tri = np.array(tri, dtype=int) + 1  # MATLAB è 1‑based
    
    return tri, pts, data, comments
