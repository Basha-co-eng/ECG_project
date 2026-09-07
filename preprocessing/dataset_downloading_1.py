import os
import wfdb
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Root folder containing all 8 dataset subfolders
DATA_ROOT = Path(r"C:\ECG_Project\training")

# The 6 hospital source folders you downloaded
SOURCE_FOLDERS = [
    "chapman_shaoxing",
    "cpsc_2018",
    "cpsc_2018_extra",
    "georgia",
    "ningbo",
    "ptb-xl"
]


def read_header_metadata(hea_path: Path, source: str) -> dict | None:
    """
    Reads a single .hea file and extracts the metadata we need,
    WITHOUT loading the actual signal (fast, low memory).
    Returns None if the file is broken/unreadable (so we can skip it later).
    """
    try:
        record_id = hea_path.stem
        header = wfdb.rdheader(str(hea_path.with_suffix("")))

        age, sex, dx_codes = None, None, None
        for comment in header.comments:
            if comment.startswith("Age:"):
                age = comment.split("Age:")[1].strip()
            elif comment.startswith("Sex:"):
                sex = comment.split("Sex:")[1].strip()
            elif comment.startswith("Dx:"):
                dx_codes = comment.split("Dx:")[1].strip()

        return {
            "record_id": record_id,
            "source_hospital": source,
            "sampling_rate": header.fs,
            "num_leads": header.n_sig,
            "num_samples": header.sig_len,
            "lead_names": ",".join(header.sig_name),
            "age": age,
            "sex": sex,
            "dx_codes": dx_codes,
            "file_path": str(hea_path.with_suffix("")),
        }
    except Exception as e:
        print(f"  [SKIPPED] {hea_path.name}: {e}")
        return None


def build_inventory():
    all_records = []
    for source in SOURCE_FOLDERS:
        folder_path = DATA_ROOT / source
        if not folder_path.exists():
            print(f"Folder not found, skipping: {folder_path}")
            continue

        hea_files = list(folder_path.rglob("*.hea"))
        print(f"{source}: found {len(hea_files)} header files")

        for hea_path in tqdm(hea_files, desc=source):
            record = read_header_metadata(hea_path, source)
            if record is not None:
                all_records.append(record)

    print(f"\nTotal records successfully read: {len(all_records)}")

    metadata_df = pd.DataFrame(all_records)
    print(metadata_df.shape)
    print(metadata_df.head())

    out_path = DATA_ROOT / "ecg_metadata_inventory.csv"
    metadata_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    return metadata_df


if __name__ == "__main__":
    build_inventory()
