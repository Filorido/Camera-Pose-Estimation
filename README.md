# Progetto\_PG

Applicazione GUI per il matching di feature tra coppie di immagini e stima di posa 2D→3D.

## Struttura del repository

```
Progetto_PG/
├── __pycache__/
├── lightglue/                # Codice sorgente LightGlue
├── matching_and_pose/        # Script per stima di posa
│   ├── cloud_get_points.py
│   ├── getInternals.py
│   ├── exterior_fiore.py
│   ├── matching_and_pose.py
│   ├── proj.py
│   ├── set_unity_camera.py
│   └── ...
├── models/                   # Modelli pre-addestrati (se presenti)
├── output/                   # Cartella di output (file di matching e parametri)
├── res/                      # Risorse varie
├── src/                      # Codice di supporto (utilities generiche)
├── third_party/              # Codici di terze parti inclusi
├── utils/                    # Utility e helper
├── weights/                  # Pesi dei modelli LightGlue
├── omniglue_matcher.py       # Wrapper OmniGlue
├── liftfeat_matcher.py       # Wrapper LiftFeat
├── lightglue_matcher.py      # Wrapper LightGlue
├── main_gui.py               # Applicazione Tkinter principale
├── matching_and_pose.py      # Script top-level di matching e posa
├── pyproject.toml
├── requirements.txt
└── README.md                 # Questo file
```

## Requisiti

* Python 3.8+
* PyTorch
* OpenCV (cv2)
* NumPy
* PIL/Pillow
* Tkinter
* SciPy
* Matplotlib
* LightGlue ([https://github.com/cvg/LightGlue](https://github.com/cvg/LightGlue))

Installa le dipendenze con:

```sh
pip install -r requirements.txt
```

## Descrizione dei componenti

### main\_gui.py

Applicazione GUI Tkinter che permette di:

* Selezionare due immagini (reference e target)
* Scegliere l'algoritmo di matching tra `OmniGlue`, `LiftFeat` e `LightGlue`
* Eseguire il matching e visualizzare i risultati in tempo reale
* Salvare i keypoint e le confidence in `output/matches_output.txt`
* Lanciare, su conferma, lo script di stima di posa

### omniglue\_matcher.py / liftfeat\_matcher.py / lightglue\_matcher.py

Wrapper per ciascun algoritmo di matching:

* Normalizzazione delle immagini
* Estrazione delle feature e corrispondenze
* Ritorno di array di keypoint, confidence e immagine di visualizzazione

### matching\_and\_pose/matching\_and\_pose.py

Script top-level per la stima di posa 2D→3D:

1. Carica i file PLY della nuvola di punti e il file di visibilità
2. Estrae punti 3D e le loro proiezioni 2D nell'immagine di riferimento
3. Legge il file `output/matches_output.txt` per ottenere i keypoint matchati
4. Allinea i punti 2D ai 3D tramite KD-Tree e soglia di distanza
5. Calcola la matrice di trasformazione (`exterior_fiore`) e i parametri per Unity
6. Salva i parametri di intrinseci/estrinseci in JSON (`output/camera_parameters.json`)

Le funzioni supportate si trovano in:

* `cloud_get_points.py`: parsing PLY + visibility
* `getInternals.py`: calibrazione camera
* `exterior_fiore.py`: stima della posa
* `proj.py`: proiezioni e utilità
* `set_unity_camera.py`: conversione dei parametri per Unity

## Esecuzione

1. **Avvia l'interfaccia grafica:**

   ```sh
   python main_gui.py
   ```
2. **Seleziona le due immagini**, scegli l'algoritmo e clicca `Esegui Matching`.
3. **Conferma** se i risultati sono soddisfacenti per proseguire con la stima di posa.

Al termine, troverai in `output/`:

* `matches_output.txt`: keypoints e confidence del matching
* `camera_parameters.json`: parametri di camera per Unity

---

© 2025 Progetto\_PG Team
