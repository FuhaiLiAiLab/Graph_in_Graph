"""
Standalone plotting script for:
  b) Kolmogorov-Smirnov test p-value dot matrix
     (T2ds vs Pre_T2ds vs No_T2ds, per clinical phenotype)

Extracted from pre_stat.ipynb (scipy.stats.ks_2samp + a matplotlib
PatchCollection dot plot). Panel a (the ridgeline plot) is generated
separately by plot_wave_distribution.R, run through real R/ggridges so it is
pixel-faithful to the original test.R output.

Inputs (produced earlier in pre_stat.ipynb):
  ./data/stat_data/label_phenodata_rvid_v1_col_name_pvalue.csv
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import TwoSlopeNorm

DATA_DIR = "./data/stat_data/"
IMAGE_DIR = "./image_storage/"

# Raw variable name (old, non-standard abbreviation) -> standardized display label.
# Full descriptions live in ./image_storage/table.tsv; the "_v1" suffix is dropped
# everywhere since every variable here comes from Visit 1.
FEATURE_LABELS = {
    "sex": "Sex",
    "age_v1": "Age",
    "hai_rl_v1": "HAI",
    "fev1z_lk_v1": "FEV1 (LK)",
    "fev1z_v1": "FEV1",
    "fev6z_v1": "FEV6",
    "fev1fvc_bcz_v1": "FEV1/FVC",
    "fvcz_invn_v1": "FVC",
    "tmpPPFEV1z_v1": "PP-FEV1",
    "tmpPPFEV6z_v1": "PP-FEV6",
    "_BMI_logz_v1": "BMI",
    "_Waistz_v1": "Waist Circ.",
    "weightz_v1": "Body Weight",
    "armspan_bcz_invn_v1": "Arm Span",
    "_Pulsez_v1": "Pulse Rate",
    "_sbp2z_v1": "SBP",
    "_dbp2z_v1": "DBP",
    "pp2z_v1": "Pulse Pressure",
    "abi_bcz_invn_v1": "ABI",
    "mmsetot_bcz_v1": "MMSE",
    "animaltotz_v1": "Animal Fluency",
    "digitfwdtotz_v1": "Digit Forward",
    "digitbwdtotz_v1": "Digit Backward",
    "digitsymtotz_v1": "Digit Symbol",
    "logmemimtotz_v1": "Logical Mem. (Imm.)",
    "logmemdlydtotz_v1": "Logical Mem. (Del.)",
    "_totscorez_invn_v1": "Total Score (SPPB)",
    "gaitspeedz_v1": "Gait Speed",
    "stand5time_invnz_v1": "5x Stand Test",
    "gripz_invn_v1": "Grip Strength",
    "albz_v1": "Albumin",
    "cholz_v1": "Cholesterol",
    "creatr_bcz_invn_v1": "Creatinine",
    "cysc_logz_invn_v1": "Cystatin C",
    "hscrp_logz_v1": "hsCRP",
    "ntbnpe_logz_v1": "NT-BNP",
    "transferrin_invnz_v1": "Transferrin",
    "inor_Tsadj_res_v1": "Inorg. T (TSadj)",
    "TSadj_BP_bcz_v1": "TSadj BP",
    "d2_logz_v1": "D2",
    "d3z_v1": "D3",
}


def standardized_label(raw_name):
    return FEATURE_LABELS.get(raw_name, raw_name)


def write_feature_label_mapping(table_tsv_path=None, out_path=None):
    """Dump full-name / standardized-label / old-abbreviation mapping to a file."""
    table_tsv_path = table_tsv_path or os.path.join(IMAGE_DIR, "table.tsv")
    out_path = out_path or os.path.join(IMAGE_DIR, "feature_label_mapping.tsv")

    descriptions = pd.read_csv(table_tsv_path, sep="\t", index_col=0)["Description"]

    rows = []
    for raw_name, label in FEATURE_LABELS.items():
        rows.append({
            "old_abbreviation": raw_name,
            "standardized_label": label,
            "full_name": descriptions.get(raw_name, ""),
        })
    mapping_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mapping_df.to_csv(out_path, sep="\t", index=False)
    return mapping_df


def plot_ks_pvalue_dotplot():
    """Panel b: KS-test p-value dot matrix across pairwise dataset comparisons."""
    pvalue_df = pd.read_csv(os.path.join(DATA_DIR, "label_phenodata_rvid_v1_col_name_pvalue.csv"))

    ylabels = [standardized_label(f) for f in pvalue_df["features"].tolist()]
    xlabels = ["T2ds vs Pre_T2ds", "T2ds vs No_T2ds", "Pre_T2ds vs No_T2ds"]
    list1 = pvalue_df["t2ds_pret2ds_pvalue"].tolist()
    list2 = pvalue_df["t2ds_no_t2ds_pvalue"].tolist()
    list3 = pvalue_df["pret2ds_no_t2ds_pvalue"].tolist()

    yn, xn = len(ylabels), len(xlabels)
    ylabels_num_list = list(range(yn)) * xn
    xlabels_num_list = [0] * yn + [1] * yn + [2] * yn
    c = np.array(list1 + list2 + list3)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_xlim(-0.5, xn - 0.5)
    ax.set_ylim(-0.5, yn - 0.5)
    ax.set(xticks=np.arange(xn), yticks=np.arange(yn),
           xticklabels=xlabels, yticklabels=ylabels)
    ax.set_xticks(np.arange(xn) - 0.5, minor=True)
    ax.set_yticks(np.arange(yn) - 0.5, minor=True)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    ax.grid(which="minor")
    ax.set_aspect("equal", "box")

    radius = 0.3
    circles = [plt.Circle((xlabels_num_list[i], ylabels_num_list[i]), radius=radius)
               for i in range(len(c))]
    norm = TwoSlopeNorm(vmin=0, vmax=1, vcenter=0.1)
    col = PatchCollection(circles, array=c, cmap="Oranges_r", norm=norm)
    ax.add_collection(col)
    fig.colorbar(col, shrink=0.2, aspect=10, pad=0.01)

    fig.tight_layout()
    os.makedirs(IMAGE_DIR, exist_ok=True)
    fig.savefig(os.path.join(IMAGE_DIR, "ks_pvalue_dotplot.png"), dpi=1200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_ks_pvalue_dotplot()
    write_feature_label_mapping()
