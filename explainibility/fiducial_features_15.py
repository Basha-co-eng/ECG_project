# ============ START: fiducial_extraction.py ============
import numpy as np
import pandas as pd
import neurokit2 as nk
from pathlib import Path
from tqdm import tqdm

PROCESSED_DIR = Path(r"C:\ECG_Project\training\processed_signals")
STAGE4_CSV_PATH = r"C:\ECG_Project\training\ecg_metadata_stage4_final.csv"
OUTPUT_CSV_PATH = r"C:\ECG_Project\training\ecg_fiducial_features.csv"
FAILED_LOG_PATH = r"C:\ECG_Project\training\stage5_failed_records.csv"

SAMPLING_RATE = 500        # matches Stage 4's TARGET_FS
LEAD_INDEX = 1              # Lead II — standard for P/QRS/T delineation
LEAD_NAME = "II"

# Checkpoint the running results every N records, in case of interruption
CHECKPOINT_INTERVAL = 5000


def extract_fiducial_features(record_id: str, processed_dir: Path,
                               sampling_rate: int = SAMPLING_RATE,
                               lead_index: int = LEAD_INDEX) -> dict | None:
    """
    Runs neurokit2's ECG pipeline on ONE lead of a preprocessed signal,
    and pulls out the fiducial features needed for the SHAP layer:
    P/QRS/T amplitudes + PR/QT/RR intervals.
    Returns None if the signal is too noisy/short for reliable delineation
    (neurokit2 will throw on these — that's expected for a small fraction).
    """
    try:
        signal = np.load(processed_dir / f"{record_id}.npy")
        lead_signal = signal[lead_index, :]   # shape (5000,) — single lead, already filtered+normalized

        # neurokit2's full pipeline: clean -> detect R-peaks -> delineate P/QRS/T
        signals_df, info = nk.ecg_process(lead_signal, sampling_rate=sampling_rate)

        # --- R-peaks -> RR intervals (in seconds) ---
        r_peaks = info["ECG_R_Peaks"]
        if len(r_peaks) < 2:
            return None   # can't compute RR interval with fewer than 2 beats detected
        rr_intervals = np.diff(r_peaks) / sampling_rate
        mean_rr = float(np.mean(rr_intervals))

        # --- Wave boundaries/peaks located by nk.ecg_delineate (already run inside ecg_process) ---
        p_peaks = np.array(info["ECG_P_Peaks"], dtype=float)
        r_onsets = np.array(info["ECG_R_Onsets"], dtype=float)
        q_peaks = np.array(info["ECG_Q_Peaks"], dtype=float)
        s_peaks = np.array(info["ECG_S_Peaks"], dtype=float)
        t_peaks = np.array(info["ECG_T_Peaks"], dtype=float)
        t_offsets = np.array(info["ECG_T_Offsets"], dtype=float)
        p_onsets = np.array(info["ECG_P_Onsets"], dtype=float)

        def safe_amplitude(peak_indices):
            """Average signal amplitude at a set of wave-peak sample indices, ignoring NaNs (undetected beats)."""
            valid = peak_indices[~np.isnan(peak_indices)].astype(int)
            valid = valid[(valid >= 0) & (valid < len(lead_signal))]
            if len(valid) == 0:
                return np.nan
            return float(np.mean(lead_signal[valid]))

        def safe_interval(start_indices, end_indices, sampling_rate):
            """Mean time (seconds) between matched start/end index pairs, ignoring incomplete beats."""
            n = min(len(start_indices), len(end_indices))
            diffs = []
            for i in range(n):
                if not (np.isnan(start_indices[i]) or np.isnan(end_indices[i])):
                    diffs.append((end_indices[i] - start_indices[i]) / sampling_rate)
            return float(np.mean(diffs)) if diffs else np.nan

        features = {
            "record_id": record_id,
            "lead_used": LEAD_NAME,
            "mean_rr_interval": mean_rr,
            "heart_rate_bpm": 60.0 / mean_rr if mean_rr > 0 else np.nan,
            "p_wave_amplitude": safe_amplitude(p_peaks),
            "qrs_amplitude": safe_amplitude(q_peaks),   # Q as QRS-complex reference point
            "t_wave_amplitude": safe_amplitude(t_peaks),
            "pr_interval": safe_interval(p_onsets, r_onsets, sampling_rate),
            "qt_interval": safe_interval(q_peaks, t_offsets, sampling_rate),
            "qrs_duration": safe_interval(q_peaks, s_peaks, sampling_rate),
            "n_beats_detected": len(r_peaks),
        }
        return features

    except Exception as e:
        print(f"  [FAILED] {record_id}: {e}")
        return None


def run_fiducial_extraction(metadata_df: pd.DataFrame):
    all_features = []
    failed_records = []

    # RESUME LOGIC: skip records already present in a previous partial run
    already_done = set()
    if Path(OUTPUT_CSV_PATH).exists():
        existing_df = pd.read_csv(OUTPUT_CSV_PATH)
        already_done = set(existing_df["record_id"])
        all_features = existing_df.to_dict("records")
        print(f"Resuming — {len(already_done)} records already processed")

    record_ids = metadata_df["record_id"].tolist()

    for idx, record_id in enumerate(tqdm(record_ids, desc="Extracting fiducial features")):
        if record_id in already_done:
            continue

        features = extract_fiducial_features(record_id, PROCESSED_DIR)
        if features is not None:
            all_features.append(features)
        else:
            failed_records.append(record_id)

        # Periodic checkpoint save
        if idx % CHECKPOINT_INTERVAL == 0 and idx > 0:
            pd.DataFrame(all_features).to_csv(OUTPUT_CSV_PATH, index=False)
            pd.DataFrame({"record_id": failed_records}).to_csv(FAILED_LOG_PATH, index=False)

    # Final save
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(OUTPUT_CSV_PATH, index=False)
    pd.DataFrame({"record_id": failed_records}).to_csv(FAILED_LOG_PATH, index=False)

    print(f"\nSuccessfully extracted: {len(features_df)}")
    print(f"Failed (too noisy for reliable delineation): {len(failed_records)}")
    print(f"Saved: {OUTPUT_CSV_PATH}")

    return features_df


if __name__ == "__main__":
    metadata_df = pd.read_csv(STAGE4_CSV_PATH)
    print(f"Total records to process: {len(metadata_df)}")

    features_df = run_fiducial_extraction(metadata_df)
    print(features_df.head())
    print(f"\nFinal shape: {features_df.shape}")
