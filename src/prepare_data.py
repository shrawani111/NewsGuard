import pandas as pd
import os

# ============================================================
# NewsGuard - Dataset Preparation
# ============================================================

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

COLUMNS = [
    "id",
    "label",
    "statement",
    "subjects",
    "speaker",
    "speaker_job",
    "state",
    "party",
    "barely_true_count",
    "false_count",
    "half_true_count",
    "mostly_true_count",
    "pants_fire_count",
    "context"
]


def load_dataset(file_path):
    return pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=COLUMNS
    )


def prepare_dataset(df):

    # Keep only required columns
    df = df[["statement", "label"]].copy()

    # Remove missing values
    df = df.dropna(subset=["statement", "label"])

    # Clean text
    df["statement"] = df["statement"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    # Remove empty statements
    df = df[df["statement"] != ""]

    # Remove duplicate statements
    df = df.drop_duplicates(subset=["statement"])

    # --------------------------------------------------------
    # Binary label mapping
    #
    # REAL:
    #   true
    #   mostly-true
    #
    # FAKE:
    #   false
    #   pants-fire
    #
    # UNCERTAIN:
    #   half-true
    #   barely-true
    #
    # Uncertain samples are excluded from training.
    # --------------------------------------------------------

    label_mapping = {
        "true": 1,
        "mostly-true": 1,
        "false": 0,
        "pants-fire": 0
    }

    df["target"] = df["label"].map(label_mapping)

    # Remove uncertain labels
    df = df.dropna(subset=["target"])

    # Convert target to integer
    df["target"] = df["target"].astype(int)

    # Human-readable class
    df["class"] = df["target"].map({
        0: "FAKE",
        1: "REAL"
    })

    return df


def save_dataset(df, filename):

    path = os.path.join(
        PROCESSED_DIR,
        filename
    )

    df.to_csv(
        path,
        index=False
    )

    print(f"Saved: {path}")
    print(f"Rows: {len(df)}")


# ============================================================
# MAIN
# ============================================================

print("\n========================================")
print("     NEWSGUARD DATA PREPARATION")
print("========================================\n")

# Load raw data
train_raw = load_dataset(
    os.path.join(RAW_DIR, "train.tsv")
)

valid_raw = load_dataset(
    os.path.join(RAW_DIR, "valid.tsv")
)

test_raw = load_dataset(
    os.path.join(RAW_DIR, "test.tsv")
)

print("Original dataset:")
print(f"Train      : {len(train_raw)}")
print(f"Validation : {len(valid_raw)}")
print(f"Test       : {len(test_raw)}")

# Prepare
train_df = prepare_dataset(train_raw)
valid_df = prepare_dataset(valid_raw)
test_df = prepare_dataset(test_raw)

# Save
save_dataset(
    train_df,
    "train.csv"
)

save_dataset(
    valid_df,
    "validation.csv"
)

save_dataset(
    test_df,
    "test.csv"
)

# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("        FINAL DATASET SUMMARY")
print("========================================")

print("\nTraining:")
print(train_df["class"].value_counts())

print("\nValidation:")
print(valid_df["class"].value_counts())

print("\nTesting:")
print(test_df["class"].value_counts())

total = (
    len(train_df)
    + len(valid_df)
    + len(test_df)
)

print(f"\nTotal usable samples: {total}")

print("\nExcluded uncertain labels:")
print("  half-true")
print("  barely-true")

print("\n========================================")
print("     DATA PREPARATION COMPLETE")
print("========================================")