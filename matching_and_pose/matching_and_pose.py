#!/usr/bin/env python3
import os
import sys
import json
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import cloud_get_points
import getInternals
import exterior_fiore
import set_unity_camera

from socket_server import JSONSocketOneShot

def read_matches(file_path):
    """
    Legge matches_output.txt e restituisce due array Nx2:
     - ref: keypoints nell'immagine di riferimento
     - tgt: keypoints nell'immagine target
    Si ignorano le righe di confidence.
    """
    sections = [[], []]  # 0: ref, 1: tgt
    current_section = -1

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('Keypoints Image 0'):
                current_section = 0
                continue
            if line.startswith('Keypoints Image 1'):
                current_section = 1
                continue
            if line.startswith('Match Confidence'):
                break  # salta tutto il resto
            if current_section in (0, 1):
                parts = line.split()
                try:
                    x, y = float(parts[0]), float(parts[1])
                except:
                    continue
                sections[current_section].append([x, y])

    ref = np.array(sections[0], dtype=np.float32)
    tgt = np.array(sections[1], dtype=np.float32)
    if ref.shape != tgt.shape:  # check if te matched points are equal between images
        raise ValueError("Numero di punti incoerente tra ref e tgt")
    return ref, tgt   # Return Nx2 NumPy arrays 

def select_files_window():   # open GUI to select .PLY and .txt files
    selected = {'ply': '', 'vis': ''}

    def browse(ext, key, label):
        file = filedialog.askopenfilename(
            title=f"Seleziona {ext}",
            filetypes=[(f"{ext} files", f"*.{ext.lower()}")]
        )
        if file:
            selected[key] = file
            label.config(text=os.path.basename(file), bg="#27ae60", fg="white")

    def on_ok():
        if not selected['ply'] or not selected['vis']:
            messagebox.showerror("Errore", "Devi selezionare entrambi i file.")
            return
        window.destroy()

    # Finestra principale
    window = tk.Tk()
    window.title("Seleziona file PLY e Visibility")
    window.geometry("500x200")
    window.configure(bg="#2c3e50")

    # Stile generale
    style = ttk.Style()
    style.theme_use("clam")  # Supporta la personalizzazione dei colori

    style.configure("Dark.TFrame", background="#2c3e50")
    style.configure("Dark.TLabel", background="#2c3e50", foreground="white", font=("Helvetica", 10))
    style.configure("Dark.TButton", background="#3498db", foreground="white", font=("Helvetica", 10, "bold"))
    style.map("Dark.TButton", background=[("active", "#2980b9")])

    frame = ttk.Frame(window, style="Dark.TFrame", padding=20)
    frame.pack(fill="both", expand=True)

    # Riga 0: PLY
    ttk.Label(frame, text="SamPointCloud.ply:", style="Dark.TLabel").grid(row=0, column=0, padx=10, pady=8, sticky="w")
    ply_label = tk.Label(frame, text="Nessun file", bg="#34495e", fg="white", padx=5, pady=3)
    ply_label.grid(row=0, column=1, sticky="we")
    ttk.Button(frame, text="Sfoglia", command=lambda: browse("PLY", "ply", ply_label), style="Dark.TButton").grid(row=0, column=2, padx=5)

    # Riga 1: TXT
    ttk.Label(frame, text="Visibility.txt:", style="Dark.TLabel").grid(row=1, column=0, padx=10, pady=8, sticky="w")
    vis_label = tk.Label(frame, text="Nessun file", bg="#34495e", fg="white", padx=5, pady=3)
    vis_label.grid(row=1, column=1, sticky="we")
    ttk.Button(frame, text="Sfoglia", command=lambda: browse("TXT", "vis", vis_label), style="Dark.TButton").grid(row=1, column=2, padx=5)

    # OK button
    ttk.Button(frame, text="OK", command=on_ok, style="Dark.TButton").grid(row=2, column=1, pady=15)

    frame.columnconfigure(1, weight=1)

    window.mainloop()

    return selected['ply'], selected['vis']   # return the two files path


def main():  # read the two images on prompt
    if len(sys.argv) < 3:
        print("Usage: python matching_and_pose.py <image1> <image2>")
        sys.exit(1)

    ref_img_path = sys.argv[1]  # ref image
    tgt_img_path = sys.argv[2]  # target image
    ref_img_name = os.path.basename(ref_img_path)

    # Selezione PLY e Visibility
    ply_file, vis_file = select_files_window()

    # Estrai nuvola e proiezioni
    # p2D -> 2D coordinates of reference image
    # p3D -> 3D coordinates of scene 
    
    p2D, p3D = cloud_get_points.cloud_get_points(ply_file, vis_file, ref_img_name)
    
    # Load images with matplotlib
    ref_img = plt.imread(ref_img_path)
    tgt_img = plt.imread(tgt_img_path)

    # Read only the matched points (not the confidence)
    matches_file = './output/matches_output.txt'
    f_ref, f_tgt = read_matches(matches_file)

    # Allignment 2D→3D point with KD-Tree on 2D point projected on 3D cloud 
    # Find the nearest point for each matched 2D keypoint  
    # Filter the more long distance corrispondences (< 3 px)
    #
    tree = cKDTree(p2D)
    dist, idx = tree.query(f_ref, k=1)
    spatial_mask = dist < 3.0  

    # take only the coherent points for the pose estimation 
    p3D_filt   = p3D[idx[spatial_mask]]
    f_ref_filt = f_ref[spatial_mask]
    f_tgt_filt = f_tgt[spatial_mask]
    
    # KK -> intrinsic camera matrix 
    # G -> pose matrix
    # scale -> scale to adapt 3D -> 2D
    KK = getInternals.get_internals(tgt_img_path)
    G, scale = exterior_fiore.exterior_fiore(KK, p3D_filt.T, f_tgt_filt.T)

    # convert camera parameters in Unity like format (focal, euler, position) 
    Ih, Iw = tgt_img.shape[:2]
    f_mm, sx, sy, lsx, lsy, euler_deg, pos_u = set_unity_camera.set_unity_cam(
        Iw, Ih, KK, G[:, :3], G[:, 3]
    )

    # saving of parameters
    # intrinsic matrix, pose, scale in a JSON file
    params = {
        "intrinsics_K": KK.flatten().tolist(),
        "pose_G":       G.flatten().tolist(),
        "scale_s":      float(scale),
        "unity": {
            "focal_mm":     float(f_mm),
            "sensor_x_mm":  float(sx),
            "sensor_y_mm":  float(sy),
            "lens_shift_x": float(lsx),
            "lens_shift_y": float(lsy),
            "euler_deg":    [float(a) for a in euler_deg],
            "position":     [float(c) for c in pos_u]
        }
    }
    
    os.makedirs('./output', exist_ok=True)
    out_file = './output/camera_parameters.json'
    with open(out_file, 'w') as jf:
        json.dump(params, jf, indent=2)
    
    # send JSON parameters to a local server on port 5005 
    sender = JSONSocketOneShot(host='127.0.0.1', port=5005)
    sender.send_once(params)

    return out_file

if __name__ == '__main__':
    out_json = main()
    # Popup di conferma
    root = tk.Tk()
    root.withdraw()
    """messagebox.showinfo("Fine", f"Parametri salvati in:\n{out_json}")"""
    root.destroy()

""" 
This script does:
    1. Reads two matched images and PLY files + visibility.
    2. Extracts visible 2D and 3D points.
    3. Reads keypoints matched by images.
    4. Filters inconsistent points and does 2D→3D alignment via KD-tree.
    5. Calculate camera poses and parameters for Unity.
    6. Save everything in a JSON (camera_parameters.json).
    7. It can send these parameters via socket to other software.
    
"""