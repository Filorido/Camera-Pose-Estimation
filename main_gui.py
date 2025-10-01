#!/usr/bin/env python3
import os
import subprocess
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk
import numpy as np

from omniglue_matcher import run_omniglue
from liftfeat_matcher import run_liftfeat
from lightglue_matcher import run_lightglue

class MatchingApp:
    #
    #   
    #  Set all the logic and interface of the program
    #
    #
    def __init__(self, root):     
        self.root = root
        self.root.title("Matching and pose estimation")   # main window
        self.root.geometry("700x600")
        self.root.configure(bg="#2c3e50")

        self.style = ttk.Style() #button style
        self.style.configure("TButton", font=("Helvetica", 10, "bold"), padding=6,
                             background="#3498db", foreground="white")
        self.style.map("TButton", background=[('active', '#2980b9')])
        self.style.configure("TLabel", font=("Helvetica", 10), background="#2c3e50", foreground="white")
        self.style.configure("TFrame", background="#2c3e50")
        self.style.configure("TRadiobutton", background="#2c3e50", foreground="white")

        # Avaible algorithms 
        # StringVar allow to follow the selected algorithm in GUI 
        self.algorithms = ["OmniGlue", "LiftFeat", "LightGlue"]
        self.selected_alg = tk.StringVar(value=self.algorithms[0])
        self.image1_path = None
        self.image2_path = None
        self.tk_output_image = None
        self._last_array = None

        self.build_ui()

    def build_ui(self):    
        #
        # Principal frame with title and subtitles
        # Combobox to choose algorithm
        # Buttons to select images reference and target
        # Canvas to visualize matching result
        # Status label to update the user about the state of the operation
        #
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        title_label = tk.Label(main_frame, text="Matching Algorithm and Pose Estimation",
                              font=("Helvetica", 16, "bold"), fg="#ecf0f1", bg="#34495e",
                              padx=10, pady=10)
        title_label.pack(fill="x", pady=(0, 15))

        # Selezione algoritmo
        alg_frame = ttk.Frame(main_frame)
        alg_frame.pack(fill="x", pady=10)
        ttk.Label(alg_frame, text="Seleziona algoritmo di matching:").pack(side="left", padx=(0,10))
        ttk.Combobox(alg_frame, textvariable=self.selected_alg, values=self.algorithms,
                     state="readonly", width=15, font=("Helvetica",10)).pack(side="left")

        # Selezione immagini
        img_frame = ttk.Frame(main_frame)
        img_frame.pack(fill="x", pady=10)
        btn_img1 = tk.Button(img_frame, text="Scegli immagine reference", command=self.load_image1,
                             bg="#3498db", fg="white", font=("Helvetica",10,"bold"), padx=10, pady=5)
        btn_img1.pack(fill="x", pady=5)
        self.img1_label = ttk.Label(img_frame, text="Nessun file selezionato",
                                    background="#34495e", foreground="#ecf0f1", padding=5)
        self.img1_label.pack(fill="x", pady=2)

        btn_img2 = tk.Button(img_frame, text="Scegli immagine target", command=self.load_image2,
                             bg="#3498db", fg="white", font=("Helvetica",10,"bold"), padx=10, pady=5)
        btn_img2.pack(fill="x", pady=5)
        self.img2_label = ttk.Label(img_frame, text="Nessun file selezionato",
                                    background="#34495e", foreground="#ecf0f1", padding=5)
        self.img2_label.pack(fill="x", pady=2)

        # Esegui matching
        run_btn = tk.Button(main_frame, text="Esegui Matching", command=self.run_matching,
                            bg="#3498db", fg="white", font=("Helvetica",10,"bold"), padx=15, pady=10)
        run_btn.pack(fill="x", pady=5)

        # Stato
        self.status_label = ttk.Label(main_frame, text="Seleziona le immagini e premi 'Esegui Matching'.",
                                     foreground="#3498db", font=("Helvetica",9,"italic"))
        self.status_label.pack()

        # Canvas per risultato
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill="both", expand=True, pady=10)
        ttk.Label(canvas_frame, text="Risultato Matching:", font=("Helvetica",10,"bold")).pack(anchor="w", pady=(0,5))
        self.canvas = tk.Canvas(canvas_frame, bg="white", highlightbackground="#7f8c8d", highlightthickness=2)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind('<Configure>', self._on_canvas_resize)

    def load_image1(self):
        path = filedialog.askopenfilename(filetypes=[("Image files","*.jpg *.png *.jpeg")])
        if path:
            # update GUI label and store image path
            self.image1_path = path
            self.img1_label.config(text=os.path.basename(path), background="#27ae60")

    def load_image2(self):
        path = filedialog.askopenfilename(filetypes=[("Image files","*.jpg *.png *.jpeg")])
        if path:
            self.image2_path = path
            self.img2_label.config(text=os.path.basename(path), background="#27ae60")
    
    def reset_ui(self):
        self.image1_path = None
        self.image2_path = None
        self.img1_label.config(text="Nessun file selezionato", background="#34495e")
        self.img2_label.config(text="Nessun file selezionato", background="#34495e")
        self.status_label.config(text="Seleziona le immagini e premi 'Esegui Matching'.")
        self.canvas.delete("all")
        self._last_array = None

    def resize_image(self, img, max_width=800):
        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = np.array(Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS))
            return resized, scale
        return img, 1.0

    def run_matching(self):
        if not (self.image1_path and self.image2_path):
            messagebox.showerror("Errore","Seleziona entrambe le immagini.")
            return
        self.status_label.config(text=f"Matching con {self.selected_alg.get()} in corso...")
        self.root.update()
        
        # Load images
        img0 = np.array(Image.open(self.image1_path).convert('RGB'))
        img1 = np.array(Image.open(self.image2_path).convert('RGB'))

        # Resize before matching as Numpy array
        img0_small, scale0 = self.resize_image(img0)
        img1_small, scale1 = self.resize_image(img1)
        #print(f"img0 resized by {scale0}, img1 by {scale1}")

        # Matching
        # Call the selected algorithm to search keypoints,
        # confidence scores and viz image to visualize 
        # the corrispondences
        if self.selected_alg.get() == "OmniGlue":
            kp0_s, kp1_s, conf, viz = run_omniglue(img0_small, img1_small)
        elif self.selected_alg.get() == "LiftFeat":
            kp0_s, kp1_s, conf, viz = run_liftfeat(img0_small, img1_small)
        else:
            kp0_s, kp1_s, conf, viz = run_lightglue(img0_small, img1_small)

        # Return keypoints to the original size 
        kp0 = kp0_s * np.array([1/scale0, 1/scale0])
        kp1 = kp1_s * np.array([1/scale1, 1/scale1])

        # Save keypoints e confidence in a file
        os.makedirs("output", exist_ok=True)
        with open(os.path.join("output", "matches_output.txt"), "w") as f:
            f.write("Keypoints Image 0:\n"); np.savetxt(f, kp0, fmt="%.6f")
            f.write("\nKeypoints Image 1:\n"); np.savetxt(f, kp1, fmt="%.6f")
            f.write("\nMatch Confidence Scores:\n"); np.savetxt(f, conf, fmt="%.6f")

        # Visualize matching on GUI
        self.status_label.config(text="Inferenza completata. Output visualizzato.")
        self._display_scaled(viz)   # adabt to the canvas

        # Ask if he wants to execute the pose estimation
        if messagebox.askyesno("Conferma Matching", "Matching soddisfacente? Vuoi eseguire pose estimation?" ):
            subprocess.Popen([
                "python", "./matching_and_pose/matching_and_pose.py",
                self.image1_path, self.image2_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            self.status_label.config(text="Pose estimation avviata in background.")
        else:
            self.reset_ui()

    def _display_scaled(self, array):
        """Adapt image to canvas without distortion"""
        self._last_array = array   # save actual image on _last_array
        self.canvas.delete("all")   # delete all on canvas before re-drawing
        img = Image.fromarray(array)   # convert NumPy array in PIL image
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()  # obtain actual canvas dimensions
        scale = min(cw / img.width, ch / img.height) # evalutate scale factor to re-drawing the image maintaining the dimensions 
        nw, nh = int(img.width * scale), int(img.height * scale)  # garantee that image will be not cut
        img_resized = img.resize((nw, nh), Image.LANCZOS)  #LANCZOS -> high quality filter to scaling 
        self.tk_output_image = ImageTk.PhotoImage(img_resized) #convert PIL in an object compatible with Tkinter
        x, y = (cw - nw) // 2, (ch - nh) // 2   # coordinates to center image in canvas
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.tk_output_image)

    def _on_canvas_resize(self, event):  #called every canvas dimension change
        if self._last_array is not None:
            self._display_scaled(self._last_array)  # scaled the image with the new dimension of the canvas

if __name__ == '__main__':
    root = tk.Tk()
    app = MatchingApp(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
