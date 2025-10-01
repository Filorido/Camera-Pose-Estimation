 #!/usr/bin/env python3
import os
import numpy as np
import torch
import cv2
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

"""
    max_num_keypoints (int): massimo numero di keypoint da estrarre 
    (–> più alto = più punti, ma più rumore e calcolo).
"""

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)

def run_lightglue(img0: np.ndarray, img1: np.ndarray):
    """
    Esegue il matching con LightGlue e disegna i match sul risultato,
    aggiungendo in overlay il numero di corrispondenze trovate.
    Args:
        img0, img1: immagini in formato numpy array HxWx3 (RGB, uint8)
    Returns:
        kp0, kp1: array dei keypoints corrispondenti Nx2
        conf: array delle confidence score di matching (N,)
        viz: immagine congiunta, linee di match e numero di match
    """
    # Normalize images to [0,1] tensors
    t0 = torch.from_numpy(img0.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    t1 = torch.from_numpy(img1.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    # Estrazione feature e matching
    feats0 = extractor.extract(t0)
    feats1 = extractor.extract(t1)
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    # Estrai keypoints e matches
    kpts0 = feats0["keypoints"].cpu().numpy()
    kpts1 = feats1["keypoints"].cpu().numpy()
    matches = matches01["matches"].cpu().numpy().astype(int)

    # Corrispondenze vere
    kp0 = kpts0[matches[:, 0]]
    kp1 = kpts1[matches[:, 1]]

    # Confidence
    if "matches_confidence" in matches01:
        conf = matches01["matches_confidence"].cpu().numpy()
    else:
        conf = np.ones(len(kp0), dtype=np.float32)

    # Prepara immagine di visualizzazione
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    # Ridimensiona altezza se necessaria
    if h0 != h1:
        new_h = min(h0, h1)
        img0 = cv2.resize(img0, (int(w0 * new_h / h0), new_h))
        img1 = cv2.resize(img1, (int(w1 * new_h / h1), new_h))
        h0, w0 = img0.shape[:2]

    viz = np.concatenate([img0, img1], axis=1).copy()

    # Disegna linee di match
    line_width = 2
    for (x0, y0), (x1, y1) in zip(kp0, kp1):
        pt0 = (int(x0), int(y0))
        pt1 = (int(x1) + w0, int(y1))
        cv2.line(viz, pt0, pt1, color=(0, 255, 0), thickness=line_width)
        cv2.circle(viz, pt0, radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(viz, pt1, radius=3, color=(0, 0, 255), thickness=-1)

    # Aggiungi overlay con numero di match
    text = f"{len(kp0)} match con LightGlue"
    cv2.putText(
        viz, text,
        org=(10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.0,
        color=(255, 0, 0),
        thickness=2,
        lineType=cv2.LINE_AA
    )

    return kp0, kp1, conf, viz