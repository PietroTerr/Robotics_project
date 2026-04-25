import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap

# ── Load ──────────────────────────────────────────────────────────────────────
CSV_PATH = "src/map_001_seed1.csv"
GRID_SIZE = 50

df = pd.read_csv(CSV_PATH)

def to_grid(col):
    grid = np.full((GRID_SIZE, GRID_SIZE), np.nan)
    for _, row in df.iterrows():
        x, y = int(row["x"]), int(row["y"])
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            grid[y, x] = row[col]
    return grid

traversability = to_grid("traversability")
slope          = to_grid("slope")
uphill_angle   = to_grid("uphill_angle")
stuck_event    = to_grid("stuck_event")   # 0 / 1  (False/True → 0/1 after cast)
texture        = to_grid("texture")
color_val      = to_grid("color")

# ── Figure layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Robot Navigation Map — 50 × 50 m", fontsize=16, fontweight="bold", y=0.98)
fig.patch.set_facecolor("#1a1a2e")

LABEL_KW = dict(fontsize=10, color="white", pad=8)

# ── Helper ─────────────────────────────────────────────────────────────────────
def styled_imshow(ax, data, cmap, title, vmin=None, vmax=None, cbar_label=""):
    ax.set_facecolor("#0d0d1a")
    im = ax.imshow(
        data, origin="lower", aspect="equal",
        cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest"
    )
    ax.set_title(title, **LABEL_KW)
    ax.tick_params(colors="gray", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="gray", labelsize=8)
    cb.set_label(cbar_label, color="gray", fontsize=8)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="gray")
    return im

# 1 ── Traversability  (green = easy, red = hard) ──────────────────────────────
cmap_trav = LinearSegmentedColormap.from_list(
    "trav", ["#d62728", "#ff7f0e", "#2ca02c"])
styled_imshow(axes[0, 0], traversability, cmap_trav,
              "① Traversability", 0, 1, "0 = blocked  →  1 = free")

# 2 ── Slope (°) ───────────────────────────────────────────────────────────────
styled_imshow(axes[0, 1], slope, "plasma",
              "② Slope (°)", cbar_label="degrees")

# 3 ── Uphill angle ────────────────────────────────────────────────────────────
styled_imshow(axes[0, 2], uphill_angle, "twilight",
              "③ Uphill Heading Angle (rad)", cbar_label="radians")

# 4 ── Stuck events  (binary overlay) ─────────────────────────────────────────
ax_stuck = axes[1, 0]
ax_stuck.set_facecolor("#0d0d1a")
ax_stuck.imshow(traversability, origin="lower", aspect="equal",
                cmap="Greys", alpha=0.3, interpolation="nearest")
stuck_overlay = np.where(stuck_event > 0.5, 1.0, np.nan)
ax_stuck.imshow(stuck_overlay, origin="lower", aspect="equal",
                cmap=LinearSegmentedColormap.from_list("stuck", ["#e63946", "#e63946"]),
                alpha=0.9, interpolation="nearest")
ax_stuck.set_title("④ Stuck Events", **LABEL_KW)
ax_stuck.tick_params(colors="gray", labelsize=8)
patch = mpatches.Patch(color="#e63946", label="stuck")
ax_stuck.legend(handles=[patch], loc="lower right",
                fontsize=8, facecolor="#222", labelcolor="white")
for spine in ax_stuck.spines.values():
    spine.set_edgecolor("#444")

# 5 ── Texture ─────────────────────────────────────────────────────────────────
styled_imshow(axes[1, 1], texture, "YlOrBr",
              "⑤ Texture", cbar_label="roughness")

# 6 ── Colour value ────────────────────────────────────────────────────────────
styled_imshow(axes[1, 2], color_val, "viridis",
              "⑥ Colour Value", cbar_label="normalised")

# ── Grid overlay (cell borders) ───────────────────────────────────────────────
for ax in axes.flat:
    ax.set_xticks(np.arange(-0.5, GRID_SIZE, 5), minor=False)
    ax.set_yticks(np.arange(-0.5, GRID_SIZE, 5), minor=False)
    ax.set_xticklabels(range(0, GRID_SIZE + 1, 5), fontsize=7, color="gray")
    ax.set_yticklabels(range(0, GRID_SIZE + 1, 5), fontsize=7, color="gray")
    ax.grid(which="major", color="#333", linewidth=0.4)
    ax.set_xlabel("x (m)", color="gray", fontsize=8)
    ax.set_ylabel("y (m)", color="gray", fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("map_overview.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved → map_overview.png")
plt.show()