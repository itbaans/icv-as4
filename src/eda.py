"""
eda.py  —  Exploratory Data Analysis for Food-101 & Paris-6K
Generates figures saved to  report/figures/eda_*.png
Run from the project root:  python src/eda.py
"""

import os, sys, random, warnings
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
from skimage.feature import hog, local_binary_pattern
from collections import defaultdict

# ── allow   import config   from src/ ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import config

warnings.filterwarnings("ignore")

# ── Output folder ─────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "report", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY    = "#1B3A6B"
STEEL   = "#4472C4"
AMBER   = "#ED7D31"
GREEN   = "#70AD47"
PLUM    = "#7030A0"
ROSE    = "#C00000"
TEAL    = "#00B0F0"
BG      = "#F7F9FC"
GRID    = "#E0E6EE"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   "#CCCCCC",
    "axes.grid":        True,
    "grid.color":       GRID,
    "grid.linewidth":   0.8,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.titlecolor":  NAVY,
    "axes.labelsize":   10,
    "axes.labelcolor":  "#333333",
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  9,
    "legend.framealpha": 0.9,
    "savefig.dpi":      180,
    "savefig.bbox":     "tight",
    "savefig.facecolor": BG,
})

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def collect_dataset(ds_dir, max_per_class=None):
    """Return {class: [path, ...]} for every class folder."""
    data = {}
    for cls in sorted(os.listdir(ds_dir)):
        cls_dir = os.path.join(ds_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if max_per_class:
            imgs = imgs[:max_per_class]
        if imgs:
            data[cls] = imgs
    return data


def load_rgb(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


def dominant_hue(img_rgb):
    """Return the mean H from the HSV hue channel (0-180)."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return hsv[:, :, 0].mean()


def mean_saturation(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return hsv[:, :, 1].mean()


def mean_brightness(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return hsv[:, :, 2].mean()


def gradient_energy(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2).mean()


def lbp_entropy(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=16, R=2, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=18, range=(0, 18))
    hist = hist / (hist.sum() + 1e-7)
    return -np.sum(hist * np.log(hist + 1e-10))    # Shannon entropy


def aspect_ratio(path):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    return w / h


def sample_paths(data_dict, n=40):
    """Sample up to n random images across all classes."""
    all_paths = [p for paths in data_dict.values() for p in paths]
    return random.sample(all_paths, min(n, len(all_paths)))


def section_header(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Class-size distribution for both datasets
# ═══════════════════════════════════════════════════════════════════════════════

def fig_class_distribution(food_data, paris_data):
    section_header("Fig 1 — Class distribution")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Class-Size Distribution", fontsize=14, fontweight="bold",
                 color=NAVY, y=1.01)

    for ax, data, color, title in [
        (axes[0], food_data,  STEEL, "Food-101  (101 classes, subset)"),
        (axes[1], paris_data, AMBER, "Paris-6K  (12 landmark classes)"),
    ]:
        classes  = list(data.keys())
        counts   = [len(data[c]) for c in classes]
        short_cl = [c.replace("_", " ").title() for c in classes]

        # Horizontal bar chart, sorted descending
        order   = np.argsort(counts)[::-1]
        classes  = [short_cl[i] for i in order]
        counts   = [counts[i] for i in order]

        bars = ax.barh(classes, counts, color=color, alpha=0.85, edgecolor="white",
                       linewidth=0.6)
        ax.set_xlabel("Number of Images")
        ax.set_title(title, pad=8)
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=7 if len(classes) > 20 else 9)

        # Annotate bar values
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    str(cnt), va="center", ha="left", fontsize=7, color="#555")

        mean_c = np.mean(counts)
        ax.axvline(mean_c, color=ROSE, linestyle="--", linewidth=1.3,
                   label=f"Mean = {mean_c:.0f}")
        ax.legend()

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "eda_01_class_distribution.png")
    plt.savefig(path)
    plt.close()
    print(f"  -> {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Aspect ratio distribution
# ═══════════════════════════════════════════════════════════════════════════════

def fig_aspect_ratio(food_data, paris_data):
    section_header("Fig 2 — Aspect ratio")
    random.seed(0)

    food_ratios  = []
    paris_ratios = []

    for paths in food_data.values():
        for p in random.sample(paths, min(8, len(paths))):
            r = aspect_ratio(p)
            if r: food_ratios.append(r)

    for paths in paris_data.values():
        for p in random.sample(paths, min(30, len(paths))):
            r = aspect_ratio(p)
            if r: paris_ratios.append(r)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Image Aspect-Ratio Distribution (W/H)", fontsize=13,
                 fontweight="bold", color=NAVY)

    for ax, ratios, color, label in [
        (axes[0], food_ratios,  STEEL, "Food-101"),
        (axes[1], paris_ratios, AMBER, "Paris-6K"),
    ]:
        ax.hist(ratios, bins=30, color=color, alpha=0.8, edgecolor="white",
                linewidth=0.5)
        ax.axvline(1.0, color=ROSE, ls="--", lw=1.2, label="square (1.0)")
        ax.axvline(np.mean(ratios), color=GREEN, ls="-.", lw=1.2,
                   label=f"mean = {np.mean(ratios):.2f}")
        ax.set_xlabel("Aspect Ratio (W / H)")
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "eda_02_aspect_ratio.png")
    plt.savefig(path)
    plt.close()
    print(f"  -> {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Per-class mean HSV statistics (Food-101)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_food_hsv_per_class(food_data, n_classes=30):
    section_header("Fig 3 — Food-101 per-class HSV stats")
    random.seed(1)
    classes = sorted(food_data.keys())[:n_classes]   # first 30 alphabetically
    hues, sats, brights = [], [], []

    for cls in classes:
        paths = random.sample(food_data[cls], min(20, len(food_data[cls])))
        h_vals, s_vals, v_vals = [], [], []
        for p in paths:
            img = load_rgb(p)
            if img is None: continue
            h_vals.append(dominant_hue(img))
            s_vals.append(mean_saturation(img))
            v_vals.append(mean_brightness(img))
        hues.append(np.mean(h_vals) if h_vals else 0)
        sats.append(np.mean(s_vals) if s_vals else 0)
        brights.append(np.mean(v_vals) if v_vals else 0)

    short = [c.replace("_", " ").title() for c in classes]
    x = np.arange(len(classes))
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Food-101 — Per-Class Mean HSV Statistics\n"
                 "(30 alphabetically-first classes, 20 images each)",
                 fontsize=13, fontweight="bold", color=NAVY)

    data_pairs = [
        (axes[0], hues,    STEEL,  "Mean Hue (0–180°)",    "Hue Channel"),
        (axes[1], sats,    GREEN,  "Mean Saturation (0–255)", "Saturation Channel"),
        (axes[2], brights, AMBER,  "Mean Value/Brightness (0–255)", "Brightness Channel"),
    ]
    for ax, vals, clr, ylabel, ttl in data_pairs:
        bars = ax.bar(x, vals, color=clr, alpha=0.82, edgecolor="white", linewidth=0.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ttl, pad=4)
        ax.axhline(np.mean(vals), color=ROSE, ls="--", lw=1.1,
                   label=f"mean = {np.mean(vals):.1f}")
        ax.legend(fontsize=8)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(short, rotation=45, ha="right", fontsize=7.5)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "eda_03_food101_hsv_per_class.png")
    plt.savefig(path)
    plt.close()
    print(f"  -> {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Paris-6K per-class gradient energy + LBP entropy + brightness
# ═══════════════════════════════════════════════════════════════════════════════

def fig_paris_feature_stats(paris_data):
    section_header("Fig 4 — Paris-6K per-class feature stats")
    random.seed(2)
    classes = sorted(paris_data.keys())
    grad_means, lbp_entropies, bright_means, sat_means = [], [], [], []

    for cls in classes:
        paths = random.sample(paris_data[cls], min(40, len(paris_data[cls])))
        ge, le, bm, sm = [], [], [], []
        for p in paths:
            img = load_rgb(p)
            if img is None: continue
            ge.append(gradient_energy(img))
            le.append(lbp_entropy(img))
            bm.append(mean_brightness(img))
            sm.append(mean_saturation(img))
        grad_means.append(np.mean(ge) if ge else 0)
        lbp_entropies.append(np.mean(le) if le else 0)
        bright_means.append(np.mean(bm) if bm else 0)
        sat_means.append(np.mean(sm) if sm else 0)

    short = [c.replace("_", " ").title() for c in classes]
    x = np.arange(len(classes))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Paris-6K — Per-Class Visual Statistics (40 images/class)",
                 fontsize=13, fontweight="bold", color=NAVY)

    panels = [
        (axes[0, 0], grad_means,   STEEL, "Mean Gradient Energy", "Edge strength"),
        (axes[0, 1], lbp_entropies, AMBER, "LBP Histogram Entropy", "Texture complexity"),
        (axes[1, 0], bright_means, GREEN, "Mean Brightness (V)", "Illumination level"),
        (axes[1, 1], sat_means,    PLUM,  "Mean Saturation (S)", "Colour richness"),
    ]
    for ax, vals, clr, ylabel, ttl in panels:
        ax.bar(x, vals, color=clr, alpha=0.82, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ttl, pad=4)
        ax.axhline(np.mean(vals), color=ROSE, ls="--", lw=1.1,
                   label=f"mean = {np.mean(vals):.2f}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "eda_04_paris6k_feature_stats.png")
    plt.savefig(path)
    plt.close()
    print(f"  -> {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Intra-class colour variance vs inter-class colour variance (Food)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_intra_vs_inter_variance(food_data, n_classes=25):
    section_header("Fig 5 — Intra vs inter class HSV variance (Food-101)")
    random.seed(3)
    classes = sorted(food_data.keys())[:n_classes]

    class_hue_means = {}
    class_hue_stds  = {}

    for cls in classes:
        paths = random.sample(food_data[cls], min(30, len(food_data[cls])))
        hues = []
        for p in paths:
            img = load_rgb(p)
            if img is None: continue
            hues.append(dominant_hue(img))
        class_hue_means[cls] = np.mean(hues) if hues else 0
        class_hue_stds[cls]  = np.std(hues)  if hues else 0

    # Sort by intra-class std (descending = most variable first)
    order   = sorted(classes, key=lambda c: class_hue_stds[c], reverse=True)
    short   = [c.replace("_", " ").title() for c in order]
    stds    = [class_hue_stds[c]  for c in order]
    means   = [class_hue_means[c] for c in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Food-101 — Intra-Class Colour Variability\n"
                 "(Hue channel, 30 images/class)",
                 fontsize=13, fontweight="bold", color=NAVY)

    # Left: intra-class std
    axes[0].barh(short, stds, color=STEEL, alpha=0.82, edgecolor="white")
    axes[0].axvline(np.mean(stds), color=ROSE, ls="--", lw=1.2,
                    label=f"mean σ = {np.mean(stds):.1f}")
    axes[0].set_xlabel("Std Dev of Mean Hue across images")
    axes[0].set_title("Intra-Class Hue Variance (higher = more variable)", pad=6)
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", labelsize=7.5)
    axes[0].legend()

    # Right: scatter std vs mean
    axes[1].scatter(means, stds, color=AMBER, alpha=0.8, edgecolors=NAVY,
                    linewidths=0.5, s=60)
    for cls, m, s in zip(order, means, stds):
        if s > np.percentile(stds, 80) or s < np.percentile(stds, 20):
            axes[1].annotate(cls.replace("_", " ").title(), (m, s),
                             textcoords="offset points", xytext=(4, 2),
                             fontsize=6.5, color=NAVY)
    axes[1].set_xlabel("Mean Hue (class centroid)")
    axes[1].set_ylabel("Intra-Class Hue Std Dev")
    axes[1].set_title("Hue Std Dev vs. Mean Hue — classes cluster poorly", pad=6)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "eda_05_food101_colour_variance.png")
    plt.savefig(path)
    plt.close()
    print(f"  -> {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Gradient energy distribution comparison (Food vs Paris)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_gradient_energy_comparison(food_data, paris_data):
    section_header("Fig 6 — Gradient energy comparison")
    random.seed(4)

    def collect_ge(data, n=400):
        paths = []
        for ps in data.values():
            paths += random.sample(ps, min(n // len(data) + 2, len(ps)))
        random.shuffle(paths)
        ge = []
        for p in paths[:n]:
            img = load_rgb(p)
            if img is None: continue
            ge.append(gradient_energy(img))
        return ge

    food_ge  = collect_ge(food_data, 400)
    paris_ge = collect_ge(paris_data, 400)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Gradient Energy Distribution — Food-101 vs Paris-6K",
                 fontsize=13, fontweight="bold", color=NAVY)

    # Histogram overlay
    bins = np.linspace(0, max(max(food_ge), max(paris_ge)), 50)
    axes[0].hist(food_ge,  bins=bins, color=STEEL, alpha=0.65, label="Food-101",
                 edgecolor="white", linewidth=0.4)
    axes[0].hist(paris_ge, bins=bins, color=AMBER, alpha=0.65, label="Paris-6K",
                 edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel("Mean Gradient Magnitude")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution Overlap")
    axes[0].legend()

    # Box plot
    bp = axes[1].boxplot([food_ge, paris_ge], labels=["Food-101", "Paris-6K"],
                         patch_artist=True, notch=True,
                         medianprops=dict(color=ROSE, linewidth=2))
    for patch, color in zip(bp["boxes"], [STEEL, AMBER]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[1].set_ylabel("Mean Gradient Magnitude")
    axes[1].set_title("Boxplot Comparison")

    # Per-class bar chart for Paris
    paris_classes = sorted(paris_data.keys())
    paris_class_ge = []
    for cls in paris_classes:
        ps = random.sample(paris_data[cls], min(30, len(paris_data[cls])))
        ge = [gradient_energy(load_rgb(p)) for p in ps if load_rgb(p) is not None]
        paris_class_ge.append(np.mean(ge) if ge else 0)

    short = [c.replace("_", " ").title() for c in paris_classes]
    axes[2].bar(short, paris_class_ge, color=AMBER, alpha=0.82, edgecolor="white")
    axes[2].set_xticklabels(short, rotation=40, ha="right", fontsize=8)
    axes[2].set_ylabel("Mean Gradient Magnitude")
    axes[2].set_title("Paris-6K Gradient by Landmark")
    axes[2].axhline(np.mean(paris_class_ge), color=ROSE, ls="--", lw=1.2,
                    label=f"mean = {np.mean(paris_class_ge):.1f}")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "eda_06_gradient_energy.png")
    plt.savefig(path)
    plt.close()
    print(f"  -> {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — HOG visualisation: 4 food + 4 paris sample images
# ═══════════════════════════════════════════════════════════════════════════════

def fig_hog_visualisation(food_data, paris_data):
    section_header("Fig 7 — HOG visualisation grid")
    random.seed(5)

    def pick_and_hog(data, cls):
        p = random.choice(data[cls])
        img_rgb = load_rgb(p)
        img_gray = cv2.cvtColor(cv2.resize(img_rgb, (128, 128)), cv2.COLOR_RGB2GRAY)
        _, hog_img = hog(img_gray, orientations=9,
                         pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                         block_norm="L2-Hys", visualize=True,
                         feature_vector=True)
        return cv2.resize(img_rgb, (128, 128)), hog_img

    food_classes  = random.sample(list(food_data.keys()),  4)
    paris_classes = random.sample(list(paris_data.keys()), 4)

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("HOG Visualisation — Raw Image vs. Gradient Map\n"
                 "(Left: Food-101   |   Right: Paris-6K)",
                 fontsize=13, fontweight="bold", color=NAVY)

    gs = gridspec.GridSpec(2, 16, figure=fig, hspace=0.5, wspace=0.15)

    # Food: cols 0-7   (4 pairs of [img | hog])
    for i, cls in enumerate(food_classes):
        img, hog_img = pick_and_hog(food_data, cls)
        ax_img = fig.add_subplot(gs[0, i*2])
        ax_hog = fig.add_subplot(gs[0, i*2+1])
        ax_img.imshow(img)
        ax_hog.imshow(hog_img, cmap="inferno")
        ax_img.set_title(cls.replace("_", " ").title(), fontsize=7.5,
                         color=NAVY, pad=3)
        ax_img.axis("off"); ax_hog.axis("off")

    # Paris: second row
    for i, cls in enumerate(paris_classes):
        img, hog_img = pick_and_hog(paris_data, cls)
        ax_img = fig.add_subplot(gs[1, i*2])
        ax_hog = fig.add_subplot(gs[1, i*2+1])
        ax_img.imshow(img)
        ax_hog.imshow(hog_img, cmap="inferno")
        ax_img.set_title(cls.replace("_", " ").title(), fontsize=7.5,
                         color=NAVY, pad=3)
        ax_img.axis("off"); ax_hog.axis("off")

    path = os.path.join(OUT_DIR, "eda_07_hog_visualisation.png")
    plt.savefig(path)
    plt.close()
    print(f"  -> {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Average HSV colour per class (Paris-6K) — illumination analysis
# ═══════════════════════════════════════════════════════════════════════════════

def fig_paris_colour_profile(paris_data):
    section_header("Fig 8 — Paris-6K colour profile per landmark")
    random.seed(6)
    classes = sorted(paris_data.keys())

    h_means, s_means, v_means = [], [], []
    for cls in classes:
        ps = random.sample(paris_data[cls], min(40, len(paris_data[cls])))
        hh, ss, vv = [], [], []
        for p in ps:
            img = load_rgb(p)
            if img is None: continue
            hh.append(dominant_hue(img))
            ss.append(mean_saturation(img))
            vv.append(mean_brightness(img))
        h_means.append(np.mean(hh) if hh else 0)
        s_means.append(np.mean(ss) if ss else 0)
        v_means.append(np.mean(vv) if vv else 0)

    short = [c.replace("_", " ").title() for c in classes]
    x = np.arange(len(classes))
    w = 0.26

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, h_means, width=w, label="Mean Hue (H)", color=STEEL,
           alpha=0.85, edgecolor="white")
    ax.bar(x,     s_means, width=w, label="Mean Saturation (S)", color=AMBER,
           alpha=0.85, edgecolor="white")
    ax.bar(x + w, v_means, width=w, label="Mean Brightness (V)", color=GREEN,
           alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Pixel Value (0–255 / 0–180)")
    ax.set_title("Paris-6K — HSV Colour Profile per Landmark Class\n"
                 "High brightness + low saturation = stone facades; "
                 "variable hue = illumination variation",
                 fontsize=11, fontweight="bold", color=NAVY)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "eda_08_paris6k_colour_profile.png")
    plt.savefig(path)
    plt.close()
    print(f"  -> {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — LBP entropy: food vs paris (texture complexity)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_lbp_entropy_comparison(food_data, paris_data):
    section_header("Fig 9 — LBP entropy comparison")
    random.seed(7)

    def class_entropy(data, n_per_class=20):
        result = {}
        for cls, paths in data.items():
            ps = random.sample(paths, min(n_per_class, len(paths)))
            vals = []
            for p in ps:
                img = load_rgb(p)
                if img is None: continue
                vals.append(lbp_entropy(img))
            result[cls] = np.mean(vals) if vals else 0
        return result

    food_ent  = class_entropy(food_data,  20)
    paris_ent = class_entropy(paris_data, 40)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("LBP Texture Entropy per Class\n"
                 "(higher entropy = more complex / heterogeneous texture)",
                 fontsize=13, fontweight="bold", color=NAVY)

    for ax, ent_dict, color, ds_name in [
        (axes[0], food_ent,  STEEL, "Food-101"),
        (axes[1], paris_ent, AMBER, "Paris-6K"),
    ]:
        classes = sorted(ent_dict, key=ent_dict.get, reverse=True)
        vals    = [ent_dict[c] for c in classes]
        short   = [c.replace("_", " ").title() for c in classes]
        ax.barh(short, vals, color=color, alpha=0.82, edgecolor="white")
        ax.axvline(np.mean(vals), color=ROSE, ls="--", lw=1.2,
                   label=f"mean = {np.mean(vals):.2f}")
        ax.set_xlabel("Mean LBP Entropy (nats)")
        ax.set_title(ds_name)
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=7 if len(classes) > 20 else 9)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "eda_09_lbp_entropy.png")
    plt.savefig(path)
    plt.close()
    print(f"  -> {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Sample image mosaic (qualitative)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_sample_mosaics(food_data, paris_data):
    section_header("Fig 10 — Sample mosaics")
    random.seed(8)

    def make_mosaic(data, rows, cols, title, path, thumb=96):
        picks = []
        for cls in random.sample(list(data.keys()), min(rows * cols, len(data))):
            p = random.choice(data[cls])
            img = load_rgb(p)
            if img is not None:
                picks.append((cls, cv2.resize(img, (thumb, thumb))))
            if len(picks) == rows * cols:
                break

        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols * 1.3, rows * 1.5 + 0.6))
        fig.suptitle(title, fontsize=12, fontweight="bold", color=NAVY)
        for idx, ax in enumerate(axes.flat):
            if idx < len(picks):
                cls_name, img = picks[idx]
                ax.imshow(img)
                ax.set_title(cls_name.replace("_", "\n").title(),
                             fontsize=5.5, pad=2, color=NAVY)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"  -> {path}")

    make_mosaic(food_data, 5, 10,
                "Food-101 — Sample Images (one per class)",
                os.path.join(OUT_DIR, "eda_10a_food_mosaic.png"))

    make_mosaic(paris_data, 3, 4,
                "Paris-6K — Sample Images (one per landmark class)",
                os.path.join(OUT_DIR, "eda_10b_paris_mosaic.png"))

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES 11a/b + 12a/b — Per-class feature showcase
#   Each row = one class, one sampled image.
#   Columns: Original | HOG gradient | LBP pattern | Colour Histogram | SIFT KPs
# ═══════════════════════════════════════════════════════════════════════════════

COL_LABELS = ["Original", "HOG Gradient", "LBP Pattern",
               "HSV Colour Hist", "SIFT Keypoints"]
COL_COLORS = [NAVY, STEEL, AMBER, GREEN, PLUM]


def _extract_showcase_row(img_rgb):
    """Return (original, hog_img, lbp_img, hist_data, kp_img) for one image."""
    thumb  = cv2.resize(img_rgb, (160, 160))
    gray   = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    bgr    = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)

    # HOG
    _, hog_vis = hog(gray, orientations=9,
                     pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                     block_norm="L2-Hys", visualize=True, feature_vector=True)

    # LBP
    lbp_map = local_binary_pattern(gray, P=16, R=2, method="uniform")
    lbp_norm = (lbp_map / lbp_map.max() * 255).astype(np.uint8)

    # Colour histogram (HSV, normalised to [0,1])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    h_hist = h_hist / (h_hist.max() + 1e-8)
    s_hist = s_hist / (s_hist.max() + 1e-8)
    v_hist = v_hist / (v_hist.max() + 1e-8)

    # SIFT keypoints
    sift = cv2.SIFT_create(nfeatures=120)
    kps, _ = sift.detectAndCompute(gray, None)
    kp_img = cv2.drawKeypoints(
        bgr, kps, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    kp_img = cv2.cvtColor(kp_img, cv2.COLOR_BGR2RGB)

    return thumb, hog_vis, lbp_norm, (h_hist, s_hist, v_hist), kp_img


def _make_feature_showcase(data, classes_subset, ds_label, fig_tag):
    """Build one showcase figure for the given subset of classes."""
    n = len(classes_subset)
    N_COLS = 5

    # Row height: give image rows ~1.9", header row ~0.35"
    row_h  = [0.35] + [1.9] * n
    fig = plt.figure(figsize=(N_COLS * 3.2, sum(row_h) + 0.6),
                     facecolor=BG)
    fig.suptitle(
        f"{ds_label} — Per-Class Feature Showcase ({n} classes, 1 image each)\n"
        f"Original  |  HOG Gradient  |  LBP Pattern  |  HSV Colour Hist  |  SIFT Keypoints",
        fontsize=11.5, fontweight="bold", color=NAVY, y=1.01,
    )

    gs = gridspec.GridSpec(
        n + 1, N_COLS, figure=fig,
        height_ratios=row_h,
        hspace=0.08, wspace=0.04,
        left=0.13, right=0.98, top=0.97, bottom=0.02,
    )

    # ── Column header row ────────────────────────────────────────────────────
    for col_idx, (lbl, clr) in enumerate(zip(COL_LABELS, COL_COLORS)):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.set_facecolor(clr)
        ax.text(0.5, 0.5, lbl, ha="center", va="center",
                color="white", fontsize=9.5, fontweight="bold",
                transform=ax.transAxes)
        ax.axis("off")

    # ── One row per class ─────────────────────────────────────────────────────
    random.seed(99)
    for row_idx, cls in enumerate(classes_subset):
        path = random.choice(data[cls])
        img  = load_rgb(path)
        if img is None:
            continue

        orig, hog_vis, lbp_norm, (h_h, s_h, v_h), kp_img = \
            _extract_showcase_row(img)

        r = row_idx + 1   # +1 because row 0 is header

        # Class label on the left margin
        ax_lbl = fig.add_subplot(gs[r, 0])
        ax_lbl.imshow(orig)
        ax_lbl.set_ylabel(
            cls.replace("_", "\n").title(),
            fontsize=7.8, rotation=0,
            labelpad=62, va="center", color=NAVY, fontweight="bold"
        )
        ax_lbl.set_xticks([]); ax_lbl.set_yticks([])
        for sp in ax_lbl.spines.values():
            sp.set_edgecolor(NAVY); sp.set_linewidth(1.2)

        # HOG
        ax = fig.add_subplot(gs[r, 1])
        ax.imshow(hog_vis, cmap="inferno")
        ax.axis("off")

        # LBP
        ax = fig.add_subplot(gs[r, 2])
        ax.imshow(lbp_norm, cmap="plasma")
        ax.axis("off")

        # Colour histogram — multi-line plot
        ax = fig.add_subplot(gs[r, 3])
        ax.set_facecolor(BG)
        ax.plot(h_h, color="#E74C3C", lw=1.5, label="H")
        ax.fill_between(range(len(h_h)), h_h, alpha=0.15, color="#E74C3C")
        ax.plot(np.linspace(0, len(h_h)-1, len(s_h)), s_h,
                color="#27AE60", lw=1.5, label="S")
        ax.fill_between(np.linspace(0, len(h_h)-1, len(s_h)), s_h,
                        alpha=0.15, color="#27AE60")
        ax.plot(np.linspace(0, len(h_h)-1, len(v_h)), v_h,
                color="#2980B9", lw=1.5, label="V")
        ax.fill_between(np.linspace(0, len(h_h)-1, len(v_h)), v_h,
                        alpha=0.15, color="#2980B9")
        ax.set_ylim(0, 1.15)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=6, loc="upper right", ncol=3,
                  handlelength=1, borderpad=0.3, columnspacing=0.5)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)

        # SIFT keypoints
        ax = fig.add_subplot(gs[r, 4])
        ax.imshow(kp_img)
        ax.axis("off")

    out = os.path.join(OUT_DIR, f"eda_{fig_tag}_feature_showcase.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


def fig_feature_showcase_food(food_data):
    """2 figures × 10 classes = 20 randomly sampled Food-101 classes."""
    section_header("Fig 11/12 — Food-101 feature showcase")
    random.seed(10)
    all_cls = sorted(food_data.keys())
    sampled = random.sample(all_cls, min(20, len(all_cls)))
    _make_feature_showcase(food_data, sampled[:10],
                           "Food-101 (Part 1/2)", "11a_food")
    _make_feature_showcase(food_data, sampled[10:],
                           "Food-101 (Part 2/2)", "11b_food")


def fig_feature_showcase_paris(paris_data):
    """2 figures × 6 classes — covers all 12 Paris-6K landmarks."""
    section_header("Fig 13/14 — Paris-6K feature showcase")
    random.seed(11)
    all_cls = sorted(paris_data.keys())
    _make_feature_showcase(paris_data, all_cls[:6],
                           "Paris-6K (Part 1/2)", "12a_paris")
    _make_feature_showcase(paris_data, all_cls[6:],
                           "Paris-6K (Part 2/2)", "12b_paris")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading dataset metadata …")
    food_data  = collect_dataset(config.FOOD_101_DIR)
    paris_data = collect_dataset(config.PARIS_6K_DIR)

    print(f"Food-101 : {len(food_data)} classes, "
          f"{sum(len(v) for v in food_data.values())} images")
    print(f"Paris-6K : {len(paris_data)} classes, "
          f"{sum(len(v) for v in paris_data.values())} images")

    # fig_class_distribution(food_data, paris_data)          # 1
    # fig_aspect_ratio(food_data, paris_data)                # 2
    # fig_food_hsv_per_class(food_data)                      # 3
    # fig_paris_feature_stats(paris_data)                    # 4
    # fig_intra_vs_inter_variance(food_data)                 # 5
    # fig_gradient_energy_comparison(food_data, paris_data)  # 6
    # fig_hog_visualisation(food_data, paris_data)           # 7
    # fig_paris_colour_profile(paris_data)                   # 8
    # fig_lbp_entropy_comparison(food_data, paris_data)      # 9
    # fig_sample_mosaics(food_data, paris_data)              # 10
    fig_feature_showcase_food(food_data)                   # 11a, 11b
    fig_feature_showcase_paris(paris_data)                 # 12a, 12b

    print(f"\n\u2713 All figures saved to:  {OUT_DIR}")
    print("  Add them to report.tex via \\includegraphics{figures/eda_XX_...png}")
