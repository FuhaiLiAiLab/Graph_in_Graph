"""
feature_importance_heatmap.py  v8
Top-20 phenotypic biomarker feature weights — T2DS patient subgroups.

Color      : column-normalized weight (white → deep blue) within each group.
             Absent features (weight=0) shown in distinctive grey.
Annotation : raw feature weight; text auto‑colored for contrast.
Separator  : white horizontal lines between clinical domains.
Outputs    : feature_importance_heatmap.pdf  /  feature_importance_heatmap.png

Requirements: numpy pandas matplotlib seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# ── 1. Raw weight data (loaded from CSV) ───────────────────────────────────────
_df = pd.read_csv("pheno_union_weights.csv")
t2ds = dict(zip(_df["Subgroup Name"], _df["t2ds_weight"]))
pre  = dict(zip(_df["Subgroup Name"], _df["pret2ds_weight"]))
no   = dict(zip(_df["Subgroup Name"], _df["no_t2ds_weight"]))

# ── 2. Clinical domain taxonomy ────────────────────────────────────────────────
DOMAIN_ORDER = [
    "Cardiovascular", "Metabolic", "Physical Function",
    "Pulmonary", "Cognitive", "Demographic",
]
DOMAIN_MAP = {
    "_sbp2z_v1":            "Cardiovascular",
    "_dbp2z_v1":            "Cardiovascular",
    "_Pulsez_v1":           "Cardiovascular",
    "abi_bcz_invn_v1":      "Cardiovascular",
    "pp2z_v1":              "Cardiovascular",
    "ntbnpe_logz_v1":       "Cardiovascular",
    "TSadj_BP_bcz_v1":      "Cardiovascular",
    "hai_rl_v1":            "Cardiovascular",
    "_BMI_logz_v1":         "Metabolic",
    "_Waistz_v1":           "Metabolic",
    "weightz_v1":           "Metabolic",
    "albz_v1":              "Metabolic",
    "creatr_bcz_invn_v1":   "Metabolic",
    "cysc_logz_invn_v1":    "Metabolic",
    "transferrin_invnz_v1": "Metabolic",
    "hscrp_logz_v1":        "Metabolic",
    "inor_Tsadj_res_v1":    "Metabolic",
    "d2_logz_v1":           "Metabolic",
    "d3z_v1":               "Metabolic",
    "gaitspeedz_v1":        "Physical Function",
    "stand5time_invnz_v1":  "Physical Function",
    "gripz_invn_v1":        "Physical Function",
    "armspan_bcz_invn_v1":  "Physical Function",
    "fvcz_invn_v1":         "Pulmonary",
    "tmpPPFEV6z_v1":        "Pulmonary",
    "tmpPPFEV1z_v1":        "Pulmonary",
    "fev6z_v1":             "Pulmonary",
    "fev1z_lk_v1":          "Pulmonary",
    "fev1z_v1":             "Pulmonary",
    "fev1fvc_bcz_v1":       "Pulmonary",
    "digitbwdtotz_v1":      "Cognitive",
    "digitfwdtotz_v1":      "Cognitive",
    "digitsymtotz_v1":      "Cognitive",
    "mmsetot_bcz_v1":       "Cognitive",
    "logmemimtotz_v1":      "Cognitive",
    "logmemdlydtotz_v1":    "Cognitive",
    "animaltotz_v1":        "Cognitive",
    "_totscorez_invn_v1":   "Cognitive",
    "age_v1":               "Demographic",
}

DOMAIN_COLORS = {
    "Cardiovascular":    "#F5B7B1",
    "Metabolic":         "#FAD7A0",
    "Physical Function": "#ABEBC6",
    "Pulmonary":         "#AED6F1",
    "Cognitive":         "#D2B4DE",
    "Demographic":       "#A3E4D7",
}

# ── 3. Key parsing & display names ─────────────────────────────────────────────
BASE_LABEL = {
    "_sbp2z_v1":            "SBP",
    "gaitspeedz_v1":        "Gait Speed",
    "hscrp_logz_v1":        "hsCRP",
    "armspan_bcz_invn_v1":  "Arm Span",
    "digitbwdtotz_v1":      "Digit Backward",
    "inor_Tsadj_res_v1":    "Inorg. T (TSadj)",
    "albz_v1":              "Albumin",
    "_dbp2z_v1":            "DBP",
    "abi_bcz_invn_v1":      "ABI",
    "stand5time_invnz_v1":  "5× Stand Test",
    "_totscorez_invn_v1":   "Total Score",
    "creatr_bcz_invn_v1":   "Creatinine",
    "weightz_v1":           "Body Weight",
    "tmpPPFEV6z_v1":        "PP-FEV6",
    "cysc_logz_invn_v1":    "Cystatin C",
    "tmpPPFEV1z_v1":        "PP-FEV1",
    "_BMI_logz_v1":         "BMI",
    "hai_rl_v1":            "HAI",
    "ntbnpe_logz_v1":       "NT-BNP",
    "_Waistz_v1":           "Waist Circ.",
    "_Pulsez_v1":           "Pulse Rate",
    "digitfwdtotz_v1":      "Digit Forward",
    "d2_logz_v1":           "D2",
    "fvcz_invn_v1":         "FVC",
    "age_v1":               "Age",
    "fev6z_v1":             "FEV6",
    "mmsetot_bcz_v1":       "MMSE",
    "digitsymtotz_v1":      "Digit Symbol",
    "gripz_invn_v1":        "Grip Strength",
    "logmemimtotz_v1":      "Logical Mem. (Imm.)",
    "d3z_v1":               "D3",
    "logmemdlydtotz_v1":    "Logical Mem. (Del.)",
    "animaltotz_v1":        "Animal Fluency",
    "TSadj_BP_bcz_v1":      "TSadj BP",
    "fev1z_lk_v1":          "FEV1 (LK)",
    "pp2z_v1":              "Pulse Pressure",
    "fev1z_v1":             "FEV1",
    "transferrin_invnz_v1": "Transferrin",
    "fev1fvc_bcz_v1":       "FEV1/FVC",
}

def split_key(k):
    idx = k.rfind("-")
    if idx == -1:
        return k, ""
    return k[:idx], k[idx:]

def display(k):
    """
    Translates raw subgroup keys into base labels while retaining original suffixes.
    e.g., '_sbp2z_v1-1' -> 'SBP -1'
    """
    base, suf = split_key(k)
    label = BASE_LABEL.get(base, base)
    
    # 拼接基础变量名和原始后缀（如 "SBP" + " -1"）
    return f"{label} {suf}".strip() if suf else label.strip()

def get_domain(k):
    base, _ = split_key(k)
    return DOMAIN_MAP.get(base, "Other")

# ── 4. Build weight matrix ─────────────────────────────────────────────────────
all_keys = sorted(
    set(t2ds) | set(pre) | set(no),
    key=lambda k: (
        DOMAIN_ORDER.index(get_domain(k)) if get_domain(k) in DOMAIN_ORDER
        else len(DOMAIN_ORDER),
        split_key(k)[0],
        split_key(k)[1],
    ),
)

mat = pd.DataFrame(
    {
        "No T2DS":  [no.get(k, 0.0)   for k in all_keys],
        "Pre-T2DS": [pre.get(k, 0.0)  for k in all_keys],
        "T2DS":     [t2ds.get(k, 0.0) for k in all_keys],
    },
    index=[display(k) for k in all_keys],
)
domains = [get_domain(k) for k in all_keys]
n = len(all_keys)

# ── 5. Column-normalize for color: each column max → 1.0 ──────────────────────
mat_norm = mat.copy().astype(float)
for col in mat_norm.columns:
    col_max = mat_norm[col].max()
    if col_max > 0:
        mat_norm[col] = mat_norm[col] / col_max

# ── 6. No flip: DOMAIN_ORDER = [Cardiovascular … Demographic] → top to bottom
mat_norm_plot = mat_norm
mat_raw_plot  = mat
dom_flip      = domains

# ── 7. Annotation: raw weights, uniform 4 decimal places for all cells ────────
annot = mat_raw_plot.copy().astype(object)
for i in range(mat_raw_plot.shape[0]):
    for j in range(mat_raw_plot.shape[1]):
        v = mat_raw_plot.iloc[i, j]
        annot.iloc[i, j] = f"{v:.4f}"

# ── 8. Domain contiguous segments (for strip bars + separator lines) ───────────
segs = []
i = 0
while i < n:
    j = i
    while j < n and dom_flip[j] == dom_flip[i]:
        j += 1
    segs.append((dom_flip[i], i, j))
    i = j

boundaries = [s[1] for s in segs[1:]]

# ── 9. rcParams ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "pdf.fonttype":  42,
    "ps.fonttype":   42,
    "font.size":     13,
})

# ── 10. Colormap ───────────────────────────────────────────────────────────────
_x   = np.linspace(0, 1, 512)
_x_g = np.power(_x, 0.55)

_colors = [
    (1.00, 1.00, 1.00),   # 0.00  纯白
    (0.93, 0.92, 0.98),   # ~0.15 极浅冷白
    (0.82, 0.80, 0.94),   # ~0.30 浅薰衣草（偏蓝）
    (0.64, 0.61, 0.88),   # ~0.48 浅蓝紫
    (0.46, 0.40, 0.78),   # ~0.65 中靛紫
    (0.28, 0.22, 0.63),   # ~0.82 深靛紫
    (0.15, 0.10, 0.42),   # 1.00  深靛蓝紫终点
]
_base_cmap = LinearSegmentedColormap.from_list("_base", _colors, N=512)
_rgba      = _base_cmap(_x_g)

cmap = LinearSegmentedColormap.from_list("custom_seq", _rgba, N=512)
cmap.set_under("white")

import matplotlib.lines as mlines

# ══════════════════════════════════════════════════════════════════════════════
# Figure A  热图（含顶部domain色条，无colorbar，无图例）
# ══════════════════════════════════════════════════════════════════════════════
fig_w = max(15, n * 0.275)
fig   = plt.figure(figsize=(fig_w, 6), constrained_layout=True)
fig.get_layout_engine().set(rect=(0, 0, 1, 0.90))

gs      = fig.add_gridspec(2, 1, height_ratios=[0.018, 0.982], hspace=0.006)
ax_dom  = fig.add_subplot(gs[0])
ax_heat = fig.add_subplot(gs[1])

sns.heatmap(
    mat_norm_plot.T,
    ax=ax_heat,
    cmap=cmap,
    vmin=1e-9,
    vmax=1.0,
    annot=False,
    fmt="",
    annot_kws={"size": 10},
    linewidths=0.4,
    linecolor="#D6D6D6",
    cbar=False,
)

values_flat = mat_norm_plot.T.values.flatten()
for text, val in zip(ax_heat.texts, values_flat):
    text.set_color("white" if val > 0.35 else "black")


ax_heat.set_ylabel("")
ax_heat.set_xlabel("")
ax_heat.tick_params(axis="both", which="both", length=0)
ax_heat.set_yticklabels(["Non-T2DS", "Pre-T2DS", "T2DS"], rotation=0, fontsize=17)

for tick in ax_heat.get_xticklabels():
    tick.set_color("black")
    tick.set_fontsize(12)
    tick.set_rotation(90)

for dom, start, end in segs:
    mid   = (start + end) / 2.0
    width = end - start
    ax_dom.bar(mid, 1, width=width * 1.0,
               color=DOMAIN_COLORS.get(dom, "#DDDDDD"),
               edgecolor="none", linewidth=0)

ax_dom.set_ylim(0, 1)
ax_dom.set_xlim(0, n)
ax_dom.axis("off")


plt.savefig("feature_importance_heatmap.pdf", dpi=300, bbox_inches="tight")
plt.savefig("feature_importance_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Figure B  Colorbar
# ══════════════════════════════════════════════════════════════════════════════
fig_cb = plt.figure(figsize=(6.0, 10))
ax_cb  = fig_cb.add_axes([0.47, 0.38, 0.06, 0.24])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1.0))
sm.set_array([])
cb = fig_cb.colorbar(sm, cax=ax_cb, ticks=[0, 0.25, 0.5, 0.75, 1.0])
cb.ax.tick_params(labelsize=13)
cb.set_label("Relative feature weight", fontsize=14)
plt.savefig("feature_importance_colorbar.pdf", dpi=300)
plt.savefig("feature_importance_colorbar.png", dpi=300)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Figure C  Domain legend
# ══════════════════════════════════════════════════════════════════════════════
active_domains = [d for d in DOMAIN_ORDER if d in set(domains)]
dots = [
    mlines.Line2D([], [], marker="o", linestyle="None",
                  markersize=10, markerfacecolor=DOMAIN_COLORS[d],
                  markeredgecolor="none", markeredgewidth=0, label=d)
    for d in active_domains
]
fig_leg, ax_leg = plt.subplots(figsize=(3, 2.5))
ax_leg.axis("off")
ax_leg.legend(
    handles=dots,
    title="Clinical Domain",
    loc="center",
    ncol=1,
    fontsize=13,
    title_fontsize=14,
    frameon=True,
    framealpha=0.9,
    edgecolor="#CCCCCC",
    handletextpad=0.6,
    borderpad=0.8,
)
plt.savefig("feature_importance_domain_legend.pdf", dpi=300, bbox_inches="tight")
plt.savefig("feature_importance_domain_legend.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Done: {n} features × 3 groups — 3 figures saved.")