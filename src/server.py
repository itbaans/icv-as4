"""
CBIR FastAPI backend
Run:  uvicorn server:app --reload --port 7860
"""
import os, io, base64, random
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from feature_extractors import extract_hog, extract_color_hist, extract_lbp, extract_sift_bovw, extract_color_sift_bovw
from retrieval import load_index

app = FastAPI(title="CBIR API")

# ── serve the single-page frontend ──────────────────────────
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


# ────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────
FEAT_MAP = {
    "hog":        extract_hog,
    "lbp":        extract_lbp,
    "color_hist": extract_color_hist,
}

def _ds_dir(dataset: str) -> str:
    return config.FOOD_101_DIR if dataset == "food-101" else config.PARIS_6K_DIR

def _safe_cosine_normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)

def _ndarray_to_b64(arr: np.ndarray, fmt: str = "jpeg") -> str:
    ok, buf = cv2.imencode(f".{fmt}", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode() if ok else ""

def _path_to_b64(path: str) -> str:
    img = cv2.imread(path)
    if img is None:
        return ""
    ok, buf = cv2.imencode(".jpeg", img)
    return base64.b64encode(buf).decode() if ok else ""


# ────────────────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────────────────
@app.get("/api/classes")
def get_classes(dataset: str = "food-101"):
    d = _ds_dir(dataset)
    if not os.path.exists(d):
        return []
    return sorted([x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))])

@app.get("/api/images")
def get_images(dataset: str, cls: str):
    d = os.path.join(_ds_dir(dataset), cls)
    if not os.path.exists(d):
        return []
    return sorted([f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

@app.get("/api/image")
def get_image(dataset: str, cls: str, img: str):
    path = os.path.join(_ds_dir(dataset), cls, img)
    b64 = _path_to_b64(path)
    if not b64:
        raise HTTPException(404, "Image not found")
    return {"b64": b64, "path": path}

@app.get("/api/random")
def get_random(dataset: str = "food-101"):
    d = _ds_dir(dataset)
    classes = [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]
    if not classes:
        raise HTTPException(404, "No classes found")
    cls = random.choice(classes)
    imgs = [f for f in os.listdir(os.path.join(d, cls)) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if not imgs:
        raise HTTPException(404, "No images")
    img = random.choice(imgs)
    b64 = _path_to_b64(os.path.join(d, cls, img))
    return {"cls": cls, "img": img, "b64": b64}


class RetrieveRequest(BaseModel):
    dataset: str = "food-101"
    query_class: Optional[str] = None
    feature: str = "hog"            # hog | lbp | color_hist | sift | color_sift
    metric: str = "euclidean"       # euclidean | cosine | chi2
    method: str = "brute_force"     # brute_force | kd_tree | pca_brute | pca_kd_tree
    top_k: int = 10
    # image sent as base64 RGB jpeg
    image_b64: str


@app.post("/api/retrieve")
def retrieve(req: RetrieveRequest):
    # decode image
    try:
        raw = base64.b64decode(req.image_b64)
        arr = np.frombuffer(raw, np.uint8)
        img_rgb = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        query_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")

    # load index
    try:
        features, paths, labels, vocab = load_index(req.dataset, req.feature)
    except Exception as e:
        raise HTTPException(500, f"Index not found — run build_index.py first. ({e})")

    # extract query feature
    try:
        fn_map = {
            "hog":        lambda b: extract_hog(b),
            "lbp":        lambda b: extract_lbp(b),
            "color_hist": lambda b: extract_color_hist(b),
            "sift":       lambda b: extract_sift_bovw(b, vocab),
            "color_sift": lambda b: extract_color_sift_bovw(b, vocab),
        }
        q_feat = fn_map[req.feature](query_bgr)
    except Exception as e:
        raise HTTPException(500, f"Feature extraction failed: {e}")

    # cosine fix: L2-normalise
    if req.metric == "cosine":
        features_s = _safe_cosine_normalise(features.copy())
        q_feat_s   = _safe_cosine_normalise(q_feat.reshape(1, -1)).squeeze()
        dist_search = "euclidean"
    else:
        features_s  = features
        q_feat_s    = q_feat
        dist_search = req.metric

    # search
    try:
        from retrieval import IndexOptimizer
        opt = IndexOptimizer(features_s, metric=dist_search, mode=req.method)
        top_indices, top_distances = opt.search(q_feat_s, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(500, f"Search error: {e}")

    # build result list
    results = []
    n_correct = 0
    for idx, dist in zip(top_indices, top_distances):
        lbl = labels[idx]
        is_correct = bool(req.query_class) and lbl == req.query_class
        if is_correct:
            n_correct += 1
        results.append({
            "rank":       len(results) + 1,
            "label":      lbl,
            "distance":   float(dist),
            "is_correct": is_correct,
            "b64":        _path_to_b64(paths[idx]),
        })

    precision = n_correct / len(results) if results else 0
    return {
        "results":   results,
        "precision": precision,
        "n_correct": n_correct,
        "total":     len(results),
    }