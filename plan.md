# Plan — SSD detection bug

## Problem

`procesiraj_podatke` in `srcML/sklearn/disk_pipeline.py` (line 46) detects SSDs by keyword matching against the model name:

```python
ssd_keywords = ['SSD', 'MTFD', 'SSDSC', '850 PRO', '870 EVO', '860 PRO', '5300']
```

The test drive `WDC WDS500G1R0A-68A4W0` (a WD Blue SSD) does NOT match any keyword → `jeSSD = 0` → treated as HDD.

### Consequence

Since `jeSSD == 0`, SSD-aware NaN filling (lines 58–68) is skipped. HDD medians are injected instead:

| Feature | Gets (HDD median) | Should get (SSD) |
|---|---|---|
| `smart_3_raw` (spin-up time) | 15.0 | 0 |
| `smart_4_raw` (spin-up count) | 15.0 | 0 |
| `smart_193_raw` (load cycle count) | 1200.0 | 0 |

The SSD is fed fake mechanical HDD attributes. Models reason about it as an HDD with 15 spin-up cycles and 1200 load cycles → inflated risk score (~36% instead of ~5–12%).

## Fix

Replace keyword-based detection with `rotation_rate == 0` from the smartctl JSON, which is the canonical SSD signal.

### Step 1 — `pretvori_json_v_surovi_df` (line 7)

Extract `rotation_rate` from the smartctl dict and pass it through:

```python
vsebina['rotation_rate'] = smartctl_dict.get('rotation_rate', 0)
```

### Step 2 — `procesiraj_podatke` (line 45)

Replace the keyword-based `jeSSD` logic:

```python
if 'jeSSD' not in df.columns:
    if 'rotation_rate' in df.columns:
        df['jeSSD'] = df['rotation_rate'].apply(lambda x: 1 if x == 0 else 0)
    elif 'model' in df.columns:
        ssd_keywords = ['SSD', 'MTFD', 'SSDSC', '850 PRO', '870 EVO', '860 PRO', '5300']
        df['jeSSD'] = df['model'].apply(lambda x: 1 if any(k in str(x).upper() for k in ssd_keywords) else 0)
    else:
        df['jeSSD'] = 0
```

Keyword matching kept as fallback for Backblaze CSVs (which may not have `rotation_rate`).

### Step 3 — Drop `rotation_rate` before feature extraction

`rotation_rate` is not in `FEATURE_COLUMNS`, so `prepare_features` will ignore it. No further change needed there.

## Files to edit

- `srcML/sklearn/disk_pipeline.py` — both functions

## Verification

After fix, re-run the dashboard with `DiskJson/disk_data_sda.json`:
- `jeSSD` should be 1
- `smart_3_raw`, `smart_4_raw`, `smart_193_raw` should be 0 (not HDD medians)
- HIR should drop from ~36% to single digits / low teens
