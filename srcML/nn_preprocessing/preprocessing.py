import glob
import random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from srcML.disk_pipeline import procesiraj_podatke

FEATURE_COLUMNS = [
    "capacity_gigabytes",
    "jeSSD",
    "smart_1_raw",
    "smart_3_raw",
    "smart_4_raw",
    "smart_5_raw",
    "smart_7_raw",
    "smart_9_raw",
    "smart_12_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_191_raw",
    "smart_192_raw",
    "smart_193_raw",
    "smart_197_raw",
    "smart_198_raw",
    "any_critical_error",
    "total_error_count",
    "error_per_gb",
]

#branje csv
def read_csv_robust(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        #ce ne dela privzeto ločilo
        return pd.read_csv(path, sep=";", low_memory=False)

#iz posameznega CSV-ja definiramo koliko instanc hocemo in katerih
def sample_rows_from_csv(csv_path: str, healthy_per_file: int, failure_per_file: int, random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_csv_robust(csv_path)

    if "failure" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    healthy = df[df["failure"] == 0]
    failed = df[df["failure"] == 1]

    healthy_sample = pd.DataFrame()
    failed_sample = pd.DataFrame()

    if not healthy.empty:
        n = min(len(healthy), healthy_per_file)
        healthy_sample = healthy.sample(n=n, random_state=random_state)

    if not failed.empty and failure_per_file > 0:
        n = min(len(failed), failure_per_file)
        failed_sample = failed.sample(n=n, random_state=random_state)

    return healthy_sample, failed_sample

#dejansko pobiranje in agregiranje datotek, spet glede na arg
def build_dataset_from_many_csvs(data_dir: Path, max_files: int | None, healthy_per_file: int, failure_per_file: int, random_state: int,) -> tuple[pd.DataFrame, pd.DataFrame]:
    #iskanje .csv tudi za podmape
    pattern = str(data_dir / "**" / "*.csv")
    csv_files = glob.glob(pattern, recursive=True)

    if not csv_files:
        raise FileNotFoundError(f"Ni najdenih CSV datotek v: {data_dir}")

    random.Random(random_state).shuffle(csv_files)

    if max_files is not None:
        csv_files = csv_files[:max_files]

    healthy_parts = []
    failure_parts = []

    #for loop za cez datoteke
    for i, csv_path in enumerate(csv_files, start=1):
        try:
            healthy_sample, failed_sample = sample_rows_from_csv(
                csv_path=csv_path,
                healthy_per_file=healthy_per_file,
                failure_per_file=failure_per_file,
                random_state=random_state + i,
            )

            if not healthy_sample.empty:
                healthy_parts.append(healthy_sample)

            if not failed_sample.empty:
                failure_parts.append(failed_sample)

            #vsakih 25 predelanih datotek izpisemo stanje
            if i % 25 == 0:
                healthy_count = sum(len(x) for x in healthy_parts)
                failure_count = sum(len(x) for x in failure_parts)
                print(
                    f"[{i}/{len(csv_files)}] Healthy: {healthy_count:,} | Failure: {failure_count:,}"
                )

        except Exception as exc:
            print(f"Preskočena datoteka {csv_path}: {exc}")

    if not healthy_parts:
        raise RuntimeError("Ni bilo najdenih healthy vrstic za trening.")

    #zdruzimo vse v en dataframe
    healthy_df = pd.concat(healthy_parts, ignore_index=True)
    failure_df = (
        pd.concat(failure_parts, ignore_index=True)
        if failure_parts
        else pd.DataFrame()
    )

    return healthy_df, failure_df


#za klasifikator: two-pass — najprej zbere VSE failure vrstice, potem vzame enako zdravih
#ce podas failure_csv, preskoči Pass 1 in nalozi failure vrstice direktno iz te datoteke
def build_balanced_dataset_from_csvs(
    data_dir: Path,
    max_failure: int | None = None,
    random_state: int = 42,
    failure_csv: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pattern = str(data_dir / "**" / "*.csv")
    csv_files = glob.glob(pattern, recursive=True)

    if not csv_files:
        raise FileNotFoundError(f"Ni najdenih CSV datotek v: {data_dir}")

    random.Random(random_state).shuffle(csv_files)

    #--- PASS 1: failure vrstice iz dedicated CSV ali iz skeniranja ---
    if failure_csv is not None:
        print(f"Nalagam failure vrstice iz: {failure_csv}")
        failure_df = read_csv_robust(str(failure_csv))
        if "failure" in failure_df.columns:
            failure_df = failure_df[failure_df["failure"] == 1]
    else:
        failure_parts = []
        print(f"Pass 1: skeniranje {len(csv_files)} CSV datotek za failure vrstice...")
        for i, csv_path in enumerate(csv_files, start=1):
            try:
                df = read_csv_robust(csv_path)
                if "failure" not in df.columns:
                    continue
                failed = df[df["failure"] == 1]
                if not failed.empty:
                    failure_parts.append(failed)
                if i % 50 == 0:
                    f_count = sum(len(x) for x in failure_parts)
                    print(f"  [{i}/{len(csv_files)}] Failure vrstic zbranih: {f_count:,}")
            except Exception as exc:
                print(f"Preskočena datoteka {csv_path}: {exc}")
        if not failure_parts:
            raise RuntimeError("Ni bilo najdenih failure vrstic.")
        failure_df = pd.concat(failure_parts, ignore_index=True)

    if max_failure is not None and len(failure_df) > max_failure:
        failure_df = failure_df.sample(n=max_failure, random_state=random_state)

    n_failure = len(failure_df)
    print(f"Skupaj failure vrstic: {n_failure:,}")

    #--- PASS 2: zberi enako healthy vrstic (per-file cap da ne zasedemo RAM) ---
    healthy_per_file = max(1, (n_failure * 2) // len(csv_files))
    healthy_parts = []
    print(f"Pass 2: zbiram {n_failure:,} healthy vrstic (max {healthy_per_file}/file)...")
    for i, csv_path in enumerate(csv_files, start=1):
        try:
            df = read_csv_robust(csv_path)
            if "failure" not in df.columns:
                continue
            healthy = df[df["failure"] == 0]
            if not healthy.empty:
                n = min(len(healthy), healthy_per_file)
                healthy_parts.append(healthy.sample(n=n, random_state=random_state + i))
            if len(healthy_parts) and sum(len(x) for x in healthy_parts) >= n_failure * 2:
                break
        except Exception as exc:
            print(f"Preskočena datoteka {csv_path}: {exc}")

    if not healthy_parts:
        raise RuntimeError("Ni bilo najdenih healthy vrstic.")

    healthy_df = pd.concat(healthy_parts, ignore_index=True)
    if len(healthy_df) > n_failure:
        healthy_df = healthy_df.sample(n=n_failure, random_state=random_state)

    print(f"Healthy po izenacitvi (50:50): {len(healthy_df):,}")
    return healthy_df, failure_df


#za ciscenje in trans. podatkov
def prepare_features(df_raw: pd.DataFrame) -> pd.DataFrame:

    #najprej skozi pipeline, ki je ze definiran od prej za sklearn ml
    df_processed = procesiraj_podatke(df_raw)

    X = pd.DataFrame(index=df_processed.index)

    #cez fiksne feature columns, ce jih ni damo 0
    for col in FEATURE_COLUMNS:
        if col in df_processed.columns:
            X[col] = df_processed[col]
        else:
            X[col] = 0.0

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

    #vsi stolpci v stevilske tipe
    for col in FEATURE_COLUMNS:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    return X.astype("float32")

#rekonstrukcija napak
def reconstruction_errors(model: tf.keras.Model, X_scaled: np.ndarray) -> np.ndarray:
    reconstructed = model.predict(X_scaled, batch_size=4096, verbose=0)
    errors = np.mean(np.abs(X_scaled - reconstructed), axis=1)
    return errors

#normaliziran rezultat
def normalize_score(error: float, threshold: float, p999: float) -> float:
    if p999 <= threshold:
        return 0.0

    score = (error - threshold) / (p999 - threshold)
    return float(np.clip(score, 0.0, 1.0))