# ============ START: signal_etl.py ============
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly, butter, filtfilt, iirnotch
from pathlib import Path
from tqdm import tqdm
from math import gcd

# ---------------- Config ----------------
TARGET_FS = 500
TARGET_SECONDS = 10
TARGET_SAMPLES = TARGET_FS * TARGET_SECONDS   # 5000

OUTPUT_DIR = Path(r"C:\ECG_Project\training\processed_signals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FAILED_LOG_PATH = Path(r"C:\ECG_Project\training\stage4_failed_records.csv")
STAGE4_OUTPUT_PATH = Path(r"C:\ECG_Project\training\ecg_metadata_stage4_final.csv")


# ---------------- Signal processing functions ----------------
def resample_signal(signal: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    if orig_fs == target_fs:
        return signal
    g = gcd(int(orig_fs), int(target_fs))
    up, down = int(target_fs // g), int(orig_fs // g)
    return resample_poly(signal, up, down, axis=1)


def bandpass_filter(signal: np.ndarray, fs: int, low=0.5, high=40.0, order=4) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=1)


def notch_filter(signal: np.ndarray, fs: int, freq=50.0, quality=30.0) -> np.ndarray:
    nyq = fs / 2.0
    b, a = iirnotch(freq / nyq, quality)
    return filtfilt(b, a, signal, axis=1)


def get_notch_freq(source_hospital: str) -> float:
    """US-sourced data (Georgia) uses 60Hz mains; everything else here uses 50Hz."""
    return 60.0 if source_hospital == "georgia" else 50.0


def fix_length(signal: np.ndarray, target_samples: int) -> np.ndarray:
    n_leads, n_samples = signal.shape
    if n_samples == target_samples:
        return signal
    elif n_samples < target_samples:
        pad_width = target_samples - n_samples
        return np.pad(signal, ((0, 0), (0, pad_width)), mode="constant")
    else:
        return signal[:, :target_samples]


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (signal - mean) / std


def process_one_record(file_path: str, orig_fs: int, source_hospital: str) -> np.ndarray | None:
    """Full pipeline: load -> resample -> filter -> fix length -> normalize."""
    try:
        record = wfdb.rdrecord(file_path)
        signal = record.p_signal.T  # (samples, leads) -> (leads, samples)

        signal = resample_signal(signal, orig_fs, TARGET_FS)
        signal = bandpass_filter(signal, TARGET_FS)
        signal = notch_filter(signal, TARGET_FS, freq=get_notch_freq(source_hospital))  # FIX: now actually used
        signal = fix_length(signal, TARGET_SAMPLES)
        signal = normalize_signal(signal)

        return signal.astype(np.float32)
    except Exception as e:
        print(f"  [FAILED] {file_path}: {e}")
        return None


# ---------------- Test-first sanity check (run BEFORE the full loop) ----------------
def run_test_sample(metadata_df: pd.DataFrame, n=20):
    print(f"\n--- Testing on {n} random records before full run ---")
    test_sample = metadata_df.sample(n, random_state=42)
    for idx, row in test_sample.iterrows():
        signal = process_one_record(row["file_path"], row["sampling_rate"], row["source_hospital"])
        if signal is not None:
            print(f"{row['record_id']} ({row['source_hospital']}): shape={signal.shape}, "
                  f"mean={signal.mean():.4f}, std={signal.std():.4f}")
    print("--- Test sample complete ---\n")


# ---------------- Main processing loop (resumable) ----------------
def run_full_etl(metadata_df: pd.DataFrame):
    processed_log = []
    failed_records = []

    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing signals"):
        record_id = row["record_id"]
        out_path = OUTPUT_DIR / f"{record_id}.npy"

        if out_path.exists():
            continue

        signal = process_one_record(row["file_path"], row["sampling_rate"], row["source_hospital"])

        if signal is not None:
            np.save(out_path, signal)
            processed_log.append(record_id)
        else:
            failed_records.append(record_id)

        if idx % 5000 == 0 and idx > 0:
            pd.DataFrame({"record_id": failed_records}).to_csv(FAILED_LOG_PATH, index=False)

    print(f"\nNewly processed: {len(processed_log)}")
    print(f"Failed: {len(failed_records)}")

    # FIX: named column, not an ambiguous default "0" column
    pd.DataFrame({"record_id": failed_records}).to_csv(FAILED_LOG_PATH, index=False)


# ---------------- Cleanup: the missing piece that caused your FileNotFoundError ----------------
def finalize_stage4(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters metadata_df down to ONLY records that have a real .npy file on disk.
    This checks the actual filesystem (source of truth), not just the failed-records
    log — so this is safe even across kernel restarts, re-runs, or a stale log file.
    Saves the result as ecg_metadata_stage4_final.csv, which every later stage
    (train/test split, ResNet training, fiducial extraction) should load from.
    """
    print("\n--- Verifying every record has a matching .npy file on disk ---")

    exists_mask = metadata_df["record_id"].apply(
        lambda rid: (OUTPUT_DIR / f"{rid}.npy").exists()
    )
    dropped = (~exists_mask).sum()
    print(f"Dropping {dropped} records with no .npy file on disk")

    clean_df = metadata_df[exists_mask].reset_index(drop=True)
    print(f"Final usable record count: {len(clean_df)}")

    clean_df.to_csv(STAGE4_OUTPUT_PATH, index=False)
    print(f"Saved: {STAGE4_OUTPUT_PATH}")

    return clean_df


if __name__ == "__main__":
    # FIX: load Stage 3's output (5 clients only), not Stage 2's (still has 8 sources)
    metadata_df = pd.read_csv(r"C:\ECG_Project\training\ecg_metadata_final_5clients.csv")
    print(f"Total records to process: {len(metadata_df)}")

    run_test_sample(metadata_df, n=20)   # test BEFORE committing to the full run
    run_full_etl(metadata_df)            # the actual 72k-record run (resumable)
    metadata_df = finalize_stage4(metadata_df)   # the missing cleanup step
# ============ END: signal_etl.py ============
