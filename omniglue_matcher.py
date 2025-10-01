 #!/usr/bin/env python3
import numpy as np
from src import omniglue
from src.omniglue import utils

def run_omniglue(img0: np.ndarray, img1: np.ndarray):
    """
    Esegue il matching con OmniGlue.
    Restituisce keypoints, confidence e immagine di visualizzazione.
    """
    og = omniglue.OmniGlue(
        og_export="./models/omniglue.onnx",
        sp_export="./models/sp_v6.onnx",
        dino_export="./models/dinov2_vitb14_pretrain.pth",
    )
    kp0, kp1, conf = og.FindMatches(img0, img1)
    idx = [i for i, val in enumerate(conf) if val > 0.01] # <-- match threshold 
    kp0, kp1 = kp0[idx], kp1[idx]
    conf = conf[idx]

    viz = utils.visualize_matches(
        img0, img1, kp0, kp1, np.eye(len(kp0)),
        show_keypoints=True, highlight_unmatched=True,
        title=f"{len(kp0)} match con OmniGlue", line_width=2
    )
    
    return kp0, kp1, conf, viz