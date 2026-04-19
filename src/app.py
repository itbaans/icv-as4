import os
import glob
import random
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import gradio as gr

import config
from feature_extractors import extract_hog, extract_color_hist, extract_lbp, extract_sift_bovw, extract_color_sift_bovw
from retrieval import load_index

# ─────────────────────────────────────────────────────────────
# PALETTE  (muted / restful)
# ─────────────────────────────────────────────────────────────
PAL = dict(
    bg          = "#F5F4F0",   # warm off-white canvas
    surface     = "#FAFAF8",   # card surface
    border      = "#E2DFD8",   # subtle border
    text_hi     = "#2C2B28",   # near-black headings
    text_lo     = "#7A796F",   # muted body
    accent      = "#5C7A6A",   # sage green accent
    accent_lt   = "#EAF0EC",   # tint of accent
    match_fg    = "#3B6B52",   # result match colour
    miss_fg     = "#8F5B4A",   # result miss colour
    match_bg    = "#EBF3EE",
    miss_bg     = "#F6EDE9",
)

CUSTOM_CSS = f"""
/* ── Reset & Body ── */
body, .gradio-container {{
    font-family: 'DM Sans', 'Helvetica Neue', sans-serif;
    background: {PAL['bg']} !important;
    color: {PAL['text_hi']};
}}

/* ── Top header bar ── */
.cbir-header {{
    padding: 28px 0 20px;
    border-bottom: 1px solid {PAL['border']};
    margin-bottom: 24px;
}}
.cbir-header h1 {{
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: {PAL['text_hi']};
    margin: 0;
}}
.cbir-header p {{
    font-size: 0.88rem;
    color: {PAL['text_lo']};
    margin: 4px 0 0;
}}

/* ── Section headings ── */
.section-label {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {PAL['text_lo']};
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid {PAL['border']};
}}

/* ── Cards / Panels ── */
.panel {{
    background: {PAL['surface']} !important;
    border: 1px solid {PAL['border']} !important;
    border-radius: 10px !important;
    padding: 18px 20px !important;
}}

/* ── Dropdowns & Inputs ── */
.gr-form, .gr-box {{
    background: {PAL['surface']} !important;
    border-color: {PAL['border']} !important;
}}

/* ── Primary button ── */
button.primary {{
    background: {PAL['accent']} !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    transition: opacity 0.15s;
}}
button.primary:hover {{ opacity: 0.85; }}

/* ── Secondary button ── */
button.secondary {{
    background: {PAL['accent_lt']} !important;
    border: 1px solid {PAL['accent']} !important;
    color: {PAL['accent']} !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}}

/* ── Tabs ── */
.tab-nav button {{
    border-radius: 8px 8px 0 0 !important;
    font-weight: 500 !important;
}}
.tab-nav button.selected {{
    background: {PAL['accent_lt']} !important;
    color: {PAL['accent']} !important;
    border-bottom: 2px solid {PAL['accent']} !important;
}}

/* ── Image component ── */
.image-preview img {{
    border-radius: 8px;
    border: 1px solid {PAL['border']};
}}

/* ── Results image ── */
.results-panel img {{
    border-radius: 10px;
    border: 1px solid {PAL['border']};
    width: 100%;
}}

/* ── Crop instruction banner ── */
.crop-banner {{
    background: {PAL['accent_lt']};
    border: 1px solid {PAL['accent']};
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: {PAL['match_fg']};
    margin-top: 8px;
}}

/* ── Status badge ── */
.status-idle   {{ color: {PAL['text_lo']}; font-size: 0.82rem; }}
.status-ready  {{ color: {PAL['match_fg']}; font-weight: 600; font-size: 0.82rem; }}
.status-error  {{ color: {PAL['miss_fg']}; font-weight: 600; font-size: 0.82rem; }}

/* ── Metric pills ── */
.metric-row {{
    display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px;
}}
.metric-pill {{
    background: {PAL['accent_lt']};
    color: {PAL['match_fg']};
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-weight: 600;
}}
"""

# ─────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────
def get_classes(dataset_name):
    ds_dir = config.FOOD_101_DIR if dataset_name == "food-101" else config.PARIS_6K_DIR
    if not os.path.exists(ds_dir):
        return []
    return sorted([d for d in os.listdir(ds_dir) if os.path.isdir(os.path.join(ds_dir, d))])

def get_images_for_class(dataset_name, class_name):
    if not class_name:
        return []
    ds_dir = config.FOOD_101_DIR if dataset_name == "food-101" else config.PARIS_6K_DIR
    class_dir = os.path.join(ds_dir, class_name)
    if not os.path.exists(class_dir):
        return []
    return sorted([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

def load_image_from_path(dataset_name, class_name, image_name):
    if not class_name or not image_name:
        return None
    ds_dir = config.FOOD_101_DIR if dataset_name == "food-101" else config.PARIS_6K_DIR
    path = os.path.join(ds_dir, class_name, image_name)
    img = cv2.imread(path)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None

def random_image(dataset_name):
    classes = get_classes(dataset_name)
    if not classes:
        return None, None, None
    c = random.choice(classes)
    imgs = get_images_for_class(dataset_name, c)
    if not imgs:
        return c, None, None
    i = random.choice(imgs)
    return c, i, load_image_from_path(dataset_name, c, i)

# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  (with safe cosine)
# ─────────────────────────────────────────────────────────────
FEAT_MAP = {
    "HOG":           "hog",
    "LBP":           "lbp",
    "Color Hist":    "color_hist",
    "Standard SIFT": "sift",
    "Color SIFT":    "color_sift",
}

def _safe_cosine_normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows so that dot-product == cosine similarity.
    Avoids division-by-zero for zero-vectors."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms

def _extract_query_feat(f_name, query_bgr, vocab):
    if f_name == "hog":
        return extract_hog(query_bgr)
    elif f_name == "color_hist":
        return extract_color_hist(query_bgr)
    elif f_name == "lbp":
        return extract_lbp(query_bgr)
    elif f_name == "sift":
        return extract_sift_bovw(query_bgr, vocab)
    elif f_name == "color_sift":
        return extract_color_sift_bovw(query_bgr, vocab)
    raise ValueError(f"Unknown feature: {f_name}")

# ─────────────────────────────────────────────────────────────
# RETRIEVAL RESULT RENDERER
# ─────────────────────────────────────────────────────────────
def _render_results(query_img_rgb, query_class, top_indices, top_distances, paths, labels, dist_label):
    n_results = len(top_indices)
    n_cols = 5
    n_rows = 1 + (n_results + n_cols - 1) // n_cols  # query row + result rows

    fig = plt.figure(figsize=(n_cols * 3.2, n_rows * 3.4 + 0.6),
                     facecolor=PAL['bg'])
    fig.suptitle(
        f"Retrieval Results  ·  {dist_label}  ·  Top {n_results}",
        fontsize=11, color=PAL['text_lo'],
        x=0.5, y=0.985, ha='center', va='top',
        fontfamily='monospace',
    )

    # ── Query image (full width top row, centred in first column) ──
    ax_q = fig.add_subplot(n_rows, n_cols, 1)
    ax_q.imshow(query_img_rgb)
    ax_q.set_facecolor(PAL['surface'])
    ax_q.set_title("Query", fontsize=10, fontweight='bold',
                   color=PAL['text_hi'], pad=6)
    for spine in ax_q.spines.values():
        spine.set_edgecolor(PAL['accent'])
        spine.set_linewidth(2)
    ax_q.set_xticks([]); ax_q.set_yticks([])

    # Leave slots 2..n_cols empty (visual breathing room)
    for k in range(2, n_cols + 1):
        ax_blank = fig.add_subplot(n_rows, n_cols, k)
        ax_blank.axis('off')
        ax_blank.set_facecolor(PAL['bg'])

    # ── Result images ──
    n_correct = 0
    for i, (idx, d) in enumerate(zip(top_indices, top_distances)):
        slot = n_cols + i + 1  # second row onwards
        ax = fig.add_subplot(n_rows, n_cols, slot)
        ax.set_facecolor(PAL['surface'])

        res_bgr = cv2.imread(paths[idx])
        if res_bgr is not None:
            ax.imshow(cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB))

        lbl = labels[idx]
        is_correct = bool(query_class) and (lbl == query_class)
        if is_correct:
            n_correct += 1
        fg  = PAL['match_fg'] if is_correct else PAL['miss_fg']
        tag = "✓" if is_correct else "✗"

        for spine in ax.spines.values():
            spine.set_edgecolor(fg)
            spine.set_linewidth(1.8)
        ax.set_title(
            f"#{i+1}  {tag}  {lbl[:18]}\n{d:.4f}",
            fontsize=8.5, color=fg, pad=4,
        )
        ax.set_xticks([]); ax.set_yticks([])

    # ── Precision badge (bottom-right corner of figure) ──
    if query_class:
        prec = n_correct / n_results if n_results else 0
        fig.text(
            0.99, 0.01,
            f"Precision@{n_results}: {prec:.0%}  ({n_correct}/{n_results} correct)",
            ha='right', va='bottom', fontsize=9,
            color=PAL['match_fg'], fontfamily='monospace',
        )

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.canvas.draw()
    snapshot = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return snapshot


# ─────────────────────────────────────────────────────────────
# MAIN RETRIEVAL FUNCTION
# ─────────────────────────────────────────────────────────────
def perform_retrieval(query_img, query_class, dataset_name,
                      feature_type, retrieval_method, distance_metric,
                      query_mode, top_k):

    if query_img is None:
        return None, "⚠ No query image selected."

    f_name = FEAT_MAP.get(feature_type, "hog")

    # ── Load index ──
    try:
        features, paths, labels, vocab = load_index(dataset_name, f_name)
    except Exception as e:
        return None, f"❌ Index not found — run `python src/build_index.py` first.\n{e}"

    # ── Handle crop-mode: query_img comes in as dict with 'composite' key ──
    if query_mode == "Crop Selection Mode":
        if isinstance(query_img, dict):
            # Gradio ImageEditor returns {'background':…, 'layers':…, 'composite':…}
            # We use 'composite' which is the cropped/annotated result
            arr = query_img.get("composite") or query_img.get("background")
            if arr is None:
                return None, "⚠ Please draw a crop selection first."
            query_arr = np.array(arr, dtype=np.uint8)
        else:
            query_arr = np.array(query_img, dtype=np.uint8)
    else:
        if isinstance(query_img, dict):
            arr = query_img.get("composite") or query_img.get("background")
            query_arr = np.array(arr, dtype=np.uint8) if arr is not None else np.array(query_img, dtype=np.uint8)
        else:
            query_arr = np.array(query_img, dtype=np.uint8)

    query_bgr = cv2.cvtColor(query_arr, cv2.COLOR_RGB2BGR)

    # ── Extract query feature ──
    try:
        q_feat = _extract_query_feat(f_name, query_bgr, vocab)
    except Exception as e:
        return None, f"❌ Feature extraction failed: {e}"

    # ── Distance / search ──
    metric_map  = {"Euclidean": "euclidean", "Cosine": "cosine", "Chi-Squared": "chi2"}
    mode_map    = {
        "Brute Force":      "brute_force",
        "KD-Tree":          "kd_tree",
        "PCA + Brute Force":"pca_brute",
        "PCA + KD-Tree":    "pca_kd_tree",
    }
    dist  = metric_map.get(distance_metric, "euclidean")
    mode  = mode_map.get(retrieval_method, "brute_force")

    # ── Fix: cosine similarity needs L2-normalised features ──
    if dist == "cosine":
        features_used = _safe_cosine_normalise(features.copy())
        q_feat_used   = _safe_cosine_normalise(q_feat.reshape(1, -1)).squeeze()
        dist_for_search = "euclidean"   # normalised euclidean == cosine distance
    else:
        features_used  = features
        q_feat_used    = q_feat
        dist_for_search = dist

    try:
        from retrieval import IndexOptimizer
        optimizer = IndexOptimizer(features_used, metric=dist_for_search, mode=mode)
        top_indices, top_distances = optimizer.search(q_feat_used, top_k=int(top_k))
    except Exception as e:
        return None, f"❌ Search error: {e}"

    snap = _render_results(
        query_arr, query_class,
        top_indices, top_distances,
        paths, labels,
        dist_label=distance_metric,
    )
    n_correct = sum(1 for idx in top_indices if labels[idx] == query_class) if query_class else 0
    prec = f"{n_correct}/{len(top_indices)} correct  ({n_correct/len(top_indices):.0%})" if query_class else "—"
    status = f"✓ Retrieved {len(top_indices)} results · Precision@K: {prec}"
    return snap, status


# ─────────────────────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────────────────────
_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("DM Sans"), "Helvetica Neue", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill         = PAL['bg'],
    block_background_fill        = PAL['surface'],
    block_border_color           = PAL['border'],
    block_border_width           = "1px",
    block_radius                 = "10px",
    block_label_text_color       = PAL['text_lo'],
    block_label_text_size        = "sm",
    block_title_text_color       = PAL['text_hi'],
    input_background_fill        = PAL['surface'],
    input_border_color           = PAL['border'],
    button_primary_background_fill = PAL['accent'],
    button_primary_text_color    = "#FFFFFF",
    button_secondary_background_fill = PAL['accent_lt'],
    button_secondary_text_color  = PAL['accent'],
    button_secondary_border_color = PAL['accent'],
)

with gr.Blocks(title="CBIR Explorer") as app:

    # ── Header ──
    gr.HTML(f"""
    <div class="cbir-header">
        <h1>🔍 CBIR Explorer</h1>
        <p>Content-Based Image Retrieval · Paris-6k &amp; Food-101</p>
    </div>
    """)

    with gr.Tabs():

        # ═══════════════════════════════════════════════════════
        # TAB 1: SETUP & QUERY
        # ═══════════════════════════════════════════════════════
        with gr.TabItem("① Select Query"):

            with gr.Row(equal_height=False):

                # ── Left column: dataset + image picker ──
                with gr.Column(scale=1, min_width=260):
                    gr.HTML('<div class="section-label">Dataset</div>')
                    dataset_dd = gr.Dropdown(
                        ["food-101", "paris6k"],
                        label="Dataset", value="food-101", container=False,
                    )

                    gr.HTML('<div class="section-label" style="margin-top:16px">Class &amp; Image</div>')
                    class_dd = gr.Dropdown(
                        choices=get_classes("food-101"),
                        label="Class", container=False,
                    )
                    image_dd = gr.Dropdown(
                        label="Image", container=False,
                    )
                    random_btn = gr.Button("⟳  Surprise Me", variant="secondary", size="sm")

                    gr.HTML('<div class="section-label" style="margin-top:16px">Query Mode</div>')
                    query_mode_radio = gr.Radio(
                        ["Full Picture", "Crop Selection Mode"],
                        label="", value="Full Picture", container=False,
                    )
                    crop_info = gr.HTML(
                        value=f'<div class="crop-banner">✂ Draw a selection box on the image, then click <b>Retrieve</b>.</div>',
                        visible=False,
                    )

                # ── Right column: image preview ──
                with gr.Column(scale=2):
                    gr.HTML('<div class="section-label">Query Image Preview</div>')
                    query_img_editor = gr.ImageEditor(
                        label="",
                        type="numpy",
                        height=420,
                        interactive=True,
                        transforms=("crop",),
                        layers=False,
                        brush=False,
                        eraser=False,
                        visible=False,
                        elem_classes=["image-preview"],
                    )
                    query_img_plain = gr.Image(
                        label="",
                        type="numpy",
                        height=420,
                        interactive=False,
                        elem_classes=["image-preview"],
                        visible=True,
                    )


        # ═══════════════════════════════════════════════════════
        # TAB 2: RETRIEVAL CONFIG
        # ═══════════════════════════════════════════════════════
        with gr.TabItem("② Configure Retrieval"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="section-label">Feature Descriptor</div>')
                    feature_dd = gr.Dropdown(
                        ["HOG", "LBP", "Color Hist", "Standard SIFT", "Color SIFT"],
                        value="HOG", label="", container=False,
                    )
                    gr.HTML("""
                    <div style="font-size:0.8rem;color:#7A796F;margin-top:6px;line-height:1.5">
                    <b>HOG</b> — shape / gradient structure<br>
                    <b>LBP</b> — local texture patterns<br>
                    <b>Color Hist</b> — global colour distribution<br>
                    <b>SIFT BoVW</b> — keypoint bag-of-words<br>
                    <b>Color SIFT</b> — colour-augmented SIFT
                    </div>
                    """)

                with gr.Column(scale=1):
                    gr.HTML('<div class="section-label">Distance Metric</div>')
                    distance_dd = gr.Dropdown(
                        ["Euclidean", "Cosine", "Chi-Squared"],
                        value="Euclidean", label="", container=False,
                    )
                    gr.HTML('<div class="section-label" style="margin-top:16px">Retrieval Method</div>')
                    retrieval_dd = gr.Dropdown(
                        ["Brute Force", "KD-Tree", "PCA + Brute Force", "PCA + KD-Tree"],
                        value="Brute Force", label="", container=False,
                    )

                with gr.Column(scale=1):
                    gr.HTML('<div class="section-label">Top-K Results</div>')
                    topk_slider = gr.Slider(
                        minimum=5, maximum=20, value=10, step=1,
                        label="", container=False,
                    )
                    gr.HTML("""
                    <div style="font-size:0.8rem;color:#7A796F;margin-top:24px;line-height:1.5">
                    <b>Tip:</b> Cosine distance is automatically handled via L2 normalisation,
                    preventing zero-vector crashes.
                    </div>
                    """)


        # ═══════════════════════════════════════════════════════
        # TAB 3: RESULTS
        # ═══════════════════════════════════════════════════════
        with gr.TabItem("③ Results"):
            with gr.Row():
                retrieve_btn = gr.Button("🔍  Retrieve", variant="primary", size="lg", scale=3)
                status_box   = gr.Textbox(
                    label="", placeholder="Status will appear here…",
                    interactive=False, scale=5,
                )
            results_img = gr.Image(
                label="",
                type="numpy",
                interactive=False,
                elem_classes=["results-panel"],
                buttons=["download", "fullscreen"],
            )

    # ─────────────────────────────────────────────────────────
    # CALLBACKS
    # ─────────────────────────────────────────────────────────
    def _update_classes(ds):
        cls = get_classes(ds)
        return gr.update(choices=cls, value=cls[0] if cls else None)

    def _update_images(ds, cls):
        imgs = get_images_for_class(ds, cls)
        return gr.update(choices=imgs, value=imgs[0] if imgs else None)

    def _load_query(ds, cls, img):
        loaded = load_image_from_path(ds, cls, img)
        # Feed same numpy array to both components; editor wraps it properly
        return gr.update(value=loaded), gr.update(value=loaded)

    def _random(ds):
        c, i, im = random_image(ds)
        return (
            gr.update(value=c, choices=get_classes(ds)),
            gr.update(value=i, choices=get_images_for_class(ds, c)),
            gr.update(value=im),   # plain
            gr.update(value=im),   # editor
        )

    def _toggle_mode(mode):
        is_crop = (mode == "Crop Selection Mode")
        return (
            gr.update(visible=is_crop),      # crop_info banner
            gr.update(visible=is_crop),      # editor
            gr.update(visible=not is_crop),  # plain viewer
        )

    def _retrieve(plain_img, editor_img, query_class, dataset,
                  feature, method, metric, mode, top_k):
        # Pick the right source depending on mode
        if mode == "Crop Selection Mode":
            img_input = editor_img
        else:
            img_input = plain_img
        snap, status = perform_retrieval(
            img_input, query_class, dataset,
            feature, method, metric, mode, top_k,
        )
        return snap, status

    # Wire up
    dataset_dd.change(_update_classes, [dataset_dd], [class_dd])
    class_dd.change(_update_images, [dataset_dd, class_dd], [image_dd])
    image_dd.change(_load_query, [dataset_dd, class_dd, image_dd],
                    [query_img_plain, query_img_editor])
    random_btn.click(_random, [dataset_dd],
                     [class_dd, image_dd, query_img_plain, query_img_editor])
    query_mode_radio.change(_toggle_mode, [query_mode_radio],
                            [crop_info, query_img_editor, query_img_plain])
    retrieve_btn.click(
        _retrieve,
        inputs=[query_img_plain, query_img_editor, class_dd, dataset_dd,
                feature_dd, retrieval_dd, distance_dd, query_mode_radio, topk_slider],
        outputs=[results_img, status_box],
    )


if __name__ == "__main__":
    app.launch(theme=_THEME, css=CUSTOM_CSS)