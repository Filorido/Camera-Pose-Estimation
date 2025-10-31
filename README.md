# Camera Pose Estimation 

GUI application for feature matching between image pairs and 2D→3D pose estimation.

## Repository Structure

```
Camera_Pose_Estimation/
├── lightglue/                # LightGlue source code
├── matching_and_pose/        # Pose estimation scripts
│   ├── cloud_get_points.py
│   ├── getInternals.py
│   ├── exterior_fiore.py
│   ├── matching_and_pose.py
│   ├── proj.py
│   ├── set_unity_camera.py
│   └── ...
├── omniglue_matcher.py       # OmniGlue wrapper
├── liftfeat_matcher.py       # LiftFeat wrapper
├── lightglue_matcher.py      # LightGlue wrapper
├── main_gui.py               # Main Tkinter application
├── matching_and_pose.py      # Top-level matching and pose script
├── requirements.txt
└── README.md                 # This file
```

## Requirements

* Python 3.8+
* PyTorch
* OpenCV (cv2)
* NumPy
* PIL/Pillow
* Tkinter
* SciPy
* Matplotlib
* LightGlue ([https://github.com/cvg/LightGlue](https://github.com/cvg/LightGlue))

Install dependencies with:

```sh
pip install -r requirements.txt
```

## Component Description

### main_gui.py

Tkinter GUI application that allows you to:

* Select two images (reference and target)
* Choose the matching algorithm among `OmniGlue`, `LiftFeat`, and `LightGlue`
* Run feature matching and visualize results in real time
* Save keypoints and confidence values to `output/matches_output.txt`
* Optionally launch the pose estimation script upon confirmation

### omniglue_matcher.py / liftfeat_matcher.py / lightglue_matcher.py

Wrappers for each matching algorithm:

* Image normalization
* Feature extraction and correspondence detection
* Return of keypoints, confidence arrays, and visualization image

### matching_and_pose/matching_and_pose.py

Top-level script for 2D→3D pose estimation:

1. Loads PLY point cloud files and visibility files
2. Extracts 3D points and their 2D projections in the reference image
3. Reads `output/matches_output.txt` to obtain matched keypoints
4. Aligns 2D and 3D points via KD-Tree and distance thresholding
5. Computes the transformation matrix (`exterior_fiore`) and Unity parameters
6. Saves intrinsic/extrinsic parameters to JSON (`output/camera_parameters.json`)

Supporting functions can be found in:

* `cloud_get_points.py`: PLY + visibility parsing
* `getInternals.py`: camera calibration
* `exterior_fiore.py`: pose estimation
* `proj.py`: projections and utilities
* `set_unity_camera.py`: parameter conversion for Unity

## Execution

1. **Launch the graphical interface:**

   ```sh
   python main_gui.py
   ```

2. **Select the two images**, choose the matching algorithm, and click `Run Matching`.

3. **Confirm** if the results are satisfactory to proceed with pose estimation.

After completion, the `output/` folder will contain:

* `matches_output.txt`: matched keypoints and confidence values
* `camera_parameters.json`: camera parameters for Unity

---

© 2025 Progetto_PG Team

