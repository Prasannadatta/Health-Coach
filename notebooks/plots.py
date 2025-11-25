import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
# (label, csv_path)
SYSTEM_CSVS = [
    ("baseline",      "./../eval_outputs/baseline_1b.csv"),
    ("finetuned",     "./../eval_outputs/finetuned_1b.csv"),
    ("rag_baseline",  "./../eval_outputs/rag_1b.csv"),
    ("rag_finetuned", "./../eval_outputs/rag_finetuned_1b.csv"),
]

PLOTS_DIR = "./../plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Limit rows per system (for speed / safety)
MAX_ROWS = 200  # set to None for all rows

# ---------- METRICS TO COMPARE ----------

# Numeric metrics where "smaller is better"
numeric_error_metrics = [
    "calories_abs_error",
    "protein_abs_error",
    "calories_rel_error",
    "protein_rel_error",
]

# Numeric metrics where "higher is better" (e.g. structure / similarity)
numeric_other_metrics = [
    "structure_score",    # will be skipped if not present
]

# Boolean / success metrics (True/False)
# has_extra_text_pred captures “extra junk text” behaviour.
bool_metrics = [
    "within_calories_tolerance",
    "within_protein_tolerance",
    "all_query_constraints_satisfied_pred",
    "all_dietary_constraints_satisfied_pred",
    "gluten_free_satisfied_pred",   # if present
    "has_extra_text_pred",          # structural: extra junk after recipe
    "structure_valid",              # structural: format ok (if present)
]

# Optional: focus on a subset while debugging (set to None to use full list)
FOCUS_NUMERIC_ERROR = None   # e.g. ["calories_abs_error"]
FOCUS_NUMERIC_OTHER = None   # e.g. ["structure_score"]
FOCUS_BOOL           = None  # e.g. ["structure_valid", "has_extra_text_pred"]

if FOCUS_NUMERIC_ERROR is not None:
    numeric_error_metrics = FOCUS_NUMERIC_ERROR

if FOCUS_NUMERIC_OTHER is not None:
    numeric_other_metrics = FOCUS_NUMERIC_OTHER

if FOCUS_BOOL is not None:
    bool_metrics = FOCUS_BOOL

# Bright color palette
COLOR_PALETTE = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
]


def to_bool_series(s: pd.Series) -> pd.Series:
    """Convert various string/num encodings into a proper boolean Series."""
    if s.dtype == bool:
        return s
    if s.dtype == object:
        return s.astype(str).str.lower().isin(["true", "1", "yes"])
    return s.astype(float) != 0.0


# ---------- LOAD ALL SYSTEMS (ROW-WISE) ----------

all_dfs = []
system_labels = []

print(">>> Loading CSVs and stacking row-wise...")
for label, path in SYSTEM_CSVS:
    if not os.path.exists(path):
        print(f"[WARN] File not found, skipping system '{label}': {path}")
        continue

    df = pd.read_csv(path)
    print(f"  {label}: loaded {df.shape}")

    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS)

    # Add system label column
    df["system"] = label

    all_dfs.append(df)
    system_labels.append(label)

if len(all_dfs) < 2:
    raise ValueError("Need at least 2 existing CSV files to compare systems.")

# Row-wise concat; columns = union of all columns across CSVs
combined = pd.concat(all_dfs, axis=0, ignore_index=True)
print(f">>> Combined shape (row-wise): {combined.shape}")
print(">>> Systems:", system_labels)


# ---------- NUMERIC ERROR METRICS: bar charts (mean±std) ----------

for metric in numeric_error_metrics:
    if metric not in combined.columns:
        print(f"[INFO] Metric '{metric}' not found in combined data, skipping.")
        continue

    print(f">>> Processing numeric error metric: {metric}")
    means = []
    stds = []
    labels_present = []

    for i, label in enumerate(system_labels):
        df_sys = combined[combined["system"] == label]
        if metric not in df_sys.columns:
            print(f"[INFO] Skipping {metric} for system '{label}' (column missing)")
            continue

        series = df_sys[metric].astype(float).dropna()
        if series.empty:
            continue

        means.append(series.mean())
        stds.append(series.std())
        labels_present.append(label)

    if len(means) < 2:
        print(f"[INFO] Not enough data for numeric error metric '{metric}'")
        continue

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(labels_present))
    colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in x]

    ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_present, rotation=0)  # horizontal
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} – mean ± std across systems\n(lower is better)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for i, m in enumerate(means):
        ax.text(i, m, f"{m:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"bar_mean_{metric}.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"    Saved bar (mean±std) for {metric} to {out_path}")


# ---------- OTHER NUMERIC METRICS: bar charts (mean±std) ----------

for metric in numeric_other_metrics:
    if metric not in combined.columns:
        print(f"[INFO] Metric '{metric}' not found in combined data, skipping.")
        continue

    print(f">>> Processing numeric metric: {metric}")
    means = []
    stds = []
    labels_present = []

    for i, label in enumerate(system_labels):
        df_sys = combined[combined["system"] == label]
        if metric not in df_sys.columns:
            print(f"[INFO] Skipping {metric} for system '{label}' (column missing)")
            continue

        series = df_sys[metric].astype(float).dropna()
        if series.empty:
            continue

        means.append(series.mean())
        stds.append(series.std())
        labels_present.append(label)

    if len(means) < 2:
        print(f"[INFO] Not enough data for numeric metric '{metric}'")
        continue

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(labels_present))
    colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in x]

    ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_present, rotation=0)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} – mean ± std across systems\n(higher is better)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for i, m in enumerate(means):
        ax.text(i, m, f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"bar_mean_{metric}.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"    Saved bar (mean±std) for {metric} to {out_path}")


# ---------- BOOLEAN METRICS: bar charts of success rate ----------

for metric in bool_metrics:
    if metric not in combined.columns:
        print(f"[INFO] Boolean metric '{metric}' not found in combined data, skipping.")
        continue

    print(f">>> Processing boolean metric: {metric}")
    rates = []
    labels_present = []

    for label in system_labels:
        df_sys = combined[combined["system"] == label]
        if metric not in df_sys.columns:
            print(f"[INFO] Skipping {metric} for system '{label}' (column missing)")
            continue

        bool_series = to_bool_series(df_sys[metric])
        rate = bool_series.mean() * 100.0
        rates.append(rate)
        labels_present.append(label)

    if len(rates) < 2:
        print(f"[INFO] Not enough data for boolean metric '{metric}'")
        continue

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(labels_present))
    colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in x]

    ax.bar(x, rates, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_present, rotation=0)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success rate (%)")
    ax.set_title(f"{metric} – success rate across systems")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for i, val in enumerate(rates):
        ax.text(i, val + 1, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"bar_rate_{metric}.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"    Saved success-rate bar chart for {metric} to {out_path}")

print(f"\nAll plots saved in: {PLOTS_DIR}")