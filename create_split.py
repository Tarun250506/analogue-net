
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_prepared"
RESULTS_DIR = ROOT / "results"

CLASSES = {
    "untreated": 0,
    "starch_treated": 1,
}

SEED = 42


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    records = []

    for class_name, label in CLASSES.items():
        class_dir = DATA_DIR / class_name

        for path in sorted(class_dir.glob("*_ambient.jpg")):
            # Example: untreated1_ambient.jpg -> untreated1
            sample_id = path.stem.rsplit("_", 1)[0]

            records.append({
                "path": str(path.relative_to(ROOT)),
                "sample_id": sample_id,
                "class_name": class_name,
                "label": label,
            })

    df = pd.DataFrame(records)

    if len(df) != 20:
        raise ValueError(
            f"Expected 20 ambient images, found {len(df)}."
        )

    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs found. Check filenames.")

    # Split separately within each class so both splits remain balanced.
    train_parts = []
    val_parts = []
    test_parts = []

    for _, class_df in df.groupby("label"):
        train_df, temp_df = train_test_split(
            class_df,
            test_size=0.4,
            random_state=SEED,
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=SEED,
        )

        train_parts.append(train_df.assign(split="train"))
        val_parts.append(val_df.assign(split="validation"))
        test_parts.append(test_df.assign(split="test"))

    manifest = pd.concat(
        train_parts + val_parts + test_parts,
        ignore_index=True,
    )

    output_path = RESULTS_DIR / "shared_split_manifest.csv"
    manifest.to_csv(output_path, index=False)

    print("Shared split created successfully.")
    print("\nCounts by split and class:")
    print(pd.crosstab(manifest["split"], manifest["class_name"]))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
