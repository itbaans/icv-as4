# Content-Based Image Retrieval (CBIR)

A classical Computer Vision CBIR system that retrieves structurally and texturally similar images using traditional feature extractors (HOG, LBP, Colour Histograms, SIFT, and Colour SIFT). No deep learning is used. 

Supports whole-image retrieval and bounding-box ROI (Region of Interest) queries.

## Simple Setup

Follow these steps to get the project running locally.

### 1. Create a Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Requirements
```powershell
pip install -r requirements.txt
```

### 3. Download the Data
Run the provided scripts to automatically download and prepare the datasets into their proper directories inside the `data/` folder:
```powershell
python src/download_food101.py
python src/download_paris6k.py
```

### 4. Build the Index
Pre-compute the feature vectors for all images structure to ensure fast retrieval.
From the project root directory, run:
```powershell
python src/build_index.py
```

### 5. Run the Application
Start the interface server using `uvicorn`:
```powershell
cd src
uvicorn server:app --reload --port 7860
```

Open your browser and navigate to `http://127.0.0.1:7860` to access the CBIR interface.
