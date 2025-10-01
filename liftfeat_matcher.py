#!/usr/bin/env python3
import numpy as np
import cv2
from models.liftfeat_wrapper import LiftFeat, MODEL_PATH

def run_liftfeat(img0: np.ndarray, img1: np.ndarray):
    """
    Esegue il matching con LiftFeat.
    Restituisce keypoints, confidence e immagine di visualizzazione,
    con overlay del numero di match trovati.
    
    detect_threshold controlla quanto “forti” devono essere i punti di interesse perché vengano restituiti.

        Valori più bassi → più keypoint (ma potenzialmente più rumore).

        Valori più alti → meno keypoint, più selettivi.
    """
    
    lf = LiftFeat(weight=MODEL_PATH, detect_threshold=0.2)
    data0 = lf.extract(img0)
    data1 = lf.extract(img1)
    kpts0 = data0['keypoints'].cpu().numpy()
    desc0 = data0['descriptors'].cpu().numpy()
    kpts1 = data1['keypoints'].cpu().numpy()
    desc1 = data1['descriptors'].cpu().numpy()

    # Matching con BFMatcher + ratio test
    """
        Il fattore 0.9 è la soglia di Lowe:

        Riducendola (es. 0.7–0.8) → match più sicuri, ma meno numerosi.

        Aumentandola (es. 0.95) → match più abbondanti ma più “rumorosi”.
    """
    bf = cv2.BFMatcher()
    raw = bf.knnMatch(desc0, desc1, k=2)
    good = [m for m, n in raw if m.distance < 0.84 * n.distance]

    # Estrai punti corrispondenti
    ref_pts = np.array([kpts0[m.queryIdx] for m in good], dtype=np.float32)
    dst_pts = np.array([kpts1[m.trainIdx] for m in good], dtype=np.float32)

    conf = np.ones(len(ref_pts), dtype=np.float32)

    viz = warp_and_draw(ref_pts, dst_pts, img0, img1)

    # Aggiungi overlay con numero di match
    text = f"{len(good)} match con LiftFeat"
    cv2.putText(
        viz, text,
        org=(10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.0,
        color=(255, 0, 0),
        thickness=2,
        lineType=cv2.LINE_AA
    )

    return ref_pts, dst_pts, conf, viz

def warp_and_draw(ref_pts, dst_pts, img1, img2):
    h, w = img1.shape[:2]
    H, mask = cv2.findHomography(
        ref_pts, dst_pts,
        cv2.USAC_MAGSAC,
        h * 0.01 + w * 0.01,
        maxIters=1000,
        confidence=0.999
    )
    mask = mask.flatten()

    # Disegna i vertici proiettati
    corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1,1,2)
    warped = cv2.perspectiveTransform(corners, H)

    img2c = img2.copy()
    for p in warped.reshape(-1, 2):
        cv2.circle(img2c, (int(p[0]), int(p[1])), 5, (0,255,0), 2)

    kp1 = [cv2.KeyPoint(float(x), float(y), 5.0) for x, y in ref_pts]
    kp2 = [cv2.KeyPoint(float(x), float(y), 5.0) for x, y in dst_pts]
    matches = [cv2.DMatch(i, i, 0) for i, m in enumerate(mask) if m]

    return cv2.drawMatches(
        img1, kp1,
        img2c, kp2,
        matches, None,
        matchColor=(0,255,0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )