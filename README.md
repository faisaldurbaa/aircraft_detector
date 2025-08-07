# ✈️ Aircraft Detection Web Application

A user-friendly web application built with Streamlit for detecting aircraft in images and videos using a fine-tuned YOLOv8 model.

Access from [Streamlit](https://aircraftdetector.streamlit.app/). (Warning: Web Version is extremely slow due to cloud limitations.)

![App Screenshot](/assets/app_screenshot.png)

## ✨ Features

-   **Multi-Image Processing:** Upload and process multiple images in a single batch.
-   **Video Analysis:** Process video files with a real-time frame-by-frame preview.
-   **Adjustable Parameters:** Fine-tune the model's **Confidence** and **IoU** thresholds directly from the UI.
-   **Multiple View Modes:** View results side-by-side with the original, or in a convenient grid layout.
-   **Bulk Downloads:** Download all processed images as a single `.zip` archive.
-   **Performance Metrics:** View detailed metrics like inference time and total detections.

## 🛠️ Local Installation & Usage

Follow these steps to run the application on your local machine.

**1. Clone the Repository:**
```bash
git clone https://github.com/faisaldurbaa/aircraft_detector.git
cd aircraft_detector
```

**2. Set Up Virtual Environment:**
It's highly recommended to use a virtual environment.
```bash
# Create the environment
python -m venv venv

# Activate the environment
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**3. Install Dependencies:**
This project uses Git LFS for model storage. Make sure you have it installed. Then, install the required Python packages.
```bash
# Install Git LFS (if you haven't already)
git lfs install

# Pull the large model file
git lfs pull

# Install Python packages
pip install -r requirements.txt
```

**4. Run the Streamlit App:**
```bash
streamlit run app.py
```
The application should now be open and running in your web browser!

## ⚙️ Model Information

-   **Model:** `YOLOv8m`
-   **Fine-tuned on:** [Military Aircraft Detection Dataset YOLO format](https://www.kaggle.com/datasets/rookieengg/military-aircraft-detection-dataset-yolo-format)
-   **Performance Target:** Optimized for a balance of speed and accuracy.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.