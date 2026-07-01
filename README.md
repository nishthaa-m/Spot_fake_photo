# Spot the Fake Photo AI (Screen Recapture Detector)

A lightweight computer vision system to detect whether an image is a **REAL** photo or a **PHOTO OF A SCREEN** (recapture fraud).

---

## 🎥 Live Demo Video
> 🔗 **[Click here to watch the Live Demo Video](PASTE_YOUR_RECORDED_VIDEO_LINK_HERE)**

---

## 🛠️ Project Structure
* `predict.py`: CLI script to run predictions.
* `train.py`: Script to train the classifier on your dataset.
* `model_assets.pkl`: Pre-trained classifier weights and scaling assets (~27 KB).
* `app.py` & `static/`: Local Flask web server and glassmorphic UI frontend for the live webcam scanner.
* `SUBMISSION_NOTE.md`: Detailed submission report containing evaluation accuracy, latency, and cost analysis.

---

## 🚀 Setup & Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install torch torchvision pillow scikit-learn flask flask-cors requests
```

---

## 💻 How to Use

### 1. Run Predictions via Command Line (CLI)
To check an image, run:
```bash
python predict.py path/to/your/image.jpg
```
* **Output**: Prints a single decimal value from `0` to `1`.
  * Closer to `0` = Real Photo
  * Closer to `1` = Screen Recapture

### 2. Run the Live Web Camera Scanner
To launch the interactive webcam demo locally:
1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open `http://localhost:5000` in your web browser.
3. Select your webcam, point it at a real object or a screen, and click **Scan Frame** or toggle **Auto-Scan Mode** to see real-time fraud scores.

### 3. Retrain the Model (Optional)
If you want to retrain the classifier on your local dataset (expects folders `real/` and `screen/` under the data directory specified in the script):
```bash
python train.py
```
This updates `model_assets.pkl` with the newly trained parameters.
