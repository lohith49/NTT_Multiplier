import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random

# -----------------------------
# Helpers
# -----------------------------
def is_power_of_2(n: int) -> int:
    return 1 if (n > 0 and (n & (n - 1)) == 0) else 0

def is_power_of_4(n: int) -> int:
    if n <= 0 or not is_power_of_2(n):
        return 0
    # power of 4 => single bit set at even position
    # equivalent: while divisible by 4, divide
    while n % 4 == 0:
        n //= 4
    return 1 if n == 1 else 0
    
def next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p

def dist_to_next_pow2(n: int) -> int:
    return max(0, next_power_of_2(n) - n)

# -----------------------------
# Load dataset (local path) and ensure required columns
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, 'fft_performance_results.csv')
OUTPUT_BALANCED = os.path.join(BASE_DIR, 'fft_balanced_dataset.csv')
EXTRA_OUTPUT = os.path.join(BASE_DIR, 'extra_data.csv')
OUTPUT_BALANCED_EQUAL = os.path.join(BASE_DIR, 'fft_balanced_equal.csv')
OUTPUT_BALANCED_80_10_10 = os.path.join(BASE_DIR, 'fft_balcnecd_ne.csv')
EXTRA_OUTPUT = os.path.join(BASE_DIR, 'extra_data.csv')

# Read CSV (expects header)
df = pd.read_csv(INPUT_CSV)

# Some datasets may have different timing column names; we only need features + target
feature_cols = ['Polynomial_Size', 'Sparsity', 'Dist_To_Next_Pow2', 'Is_Power_2', 'Is_Power_4']
target_col = 'Best_Algorithm'

missing = [c for c in feature_cols + [target_col] if c not in df.columns]
if missing:
    # Try to infer columns if names differ
    # Fallback: assume the first 10 columns are as documented in README
    # ['Polynomial_Size','Padded_Size','Sparsity','Dist_To_Next_Pow2','Is_Power_2','Is_Power_4', 'Radix_2_Time_ms','Modified_Radix_4_Time_ms','Radix_Split_Time_ms','Best_Algorithm']
    possible = ['Polynomial_Size','Padded_Size','Sparsity','Dist_To_Next_Pow2','Is_Power_2','Is_Power_4',
                'Radix_2_Time','Radix_2_Time_ms','Modified_Radix_4_Time','Modified_Radix_4_Time_ms',
                'Radix_Split_Time','Radix_Split_Time_ms','Best_Algorithm']
    keep = [c for c in possible if c in df.columns]
    df = df[keep].copy()

# Keep only the columns we need for augmentation and training
df = df[[c for c in df.columns if c in feature_cols + [target_col]]].copy()

# -----------------------------
# Split classes
# -----------------------------
df_major = df[df[target_col] == 'Modified-Radix-4'].copy()
df_radix2 = df[df[target_col] == 'Radix-2'].copy()
df_split = df[df[target_col] == 'Radix-Split'].copy()

print(f"Class counts → MR4={len(df_major)}, R2={len(df_radix2)}, RS={len(df_split)}")

# -----------------------------
# Scale only continuous features for generation: Polynomial_Size, Sparsity
# -----------------------------
cont_cols = ['Polynomial_Size', 'Sparsity']
scaler = MinMaxScaler()
scaler.fit(df[cont_cols])

X_min_r2 = scaler.transform(df_radix2[cont_cols]) if len(df_radix2) > 0 else np.empty((0,2))
X_min_rs = scaler.transform(df_split[cont_cols]) if len(df_split) > 0 else np.empty((0,2))

# -----------------------------
# GA operators in scaled space [0,1]
# -----------------------------
def crossover(parent1, parent2):
    mask = np.random.rand(parent1.shape[0]) > 0.5
    return np.where(mask, parent1, parent2)

def mutate(child, mutation_rate=0.3, mutation_strength=0.1):
    for i in range(child.shape[0]):
        if np.random.rand() < mutation_rate:
            delta = np.random.uniform(-mutation_strength, mutation_strength)
            child[i] = np.clip(child[i] + delta, 0.0, 1.0)
    return child

def generate_population(X_minority: np.ndarray, n_new: int) -> np.ndarray:
    if n_new <= 0 or X_minority.shape[0] == 0:
        return np.empty((0, X_minority.shape[1]))
    synthetic = []
    n = len(X_minority)
    for _ in range(n_new):
        p1 = X_minority[np.random.randint(0, n)]
        p2 = X_minority[np.random.randint(0, n)]
        child = crossover(p1, p2)
        child = mutate(child)
        synthetic.append(child)
    return np.vstack(synthetic)

# -----------------------------
# Target size = majority count; generate to balance
# -----------------------------
majority_size = len(df_major)
need_r2 = max(0, majority_size - len(df_radix2))
need_rs = max(0, majority_size - len(df_split))

syn_r2_scaled = generate_population(X_min_r2, need_r2)
syn_rs_scaled = generate_population(X_min_rs, need_rs)

# Inverse transform continuous features back to original scale
syn_r2_cont = scaler.inverse_transform(syn_r2_scaled) if syn_r2_scaled.size else np.empty((0,2))
syn_rs_cont = scaler.inverse_transform(syn_rs_scaled) if syn_rs_scaled.size else np.empty((0,2))

# -----------------------------
# Build full synthetic rows with integer Polynomial_Size and derived features
# -----------------------------
def build_synthetic_df(rows_cont: np.ndarray, label: str) -> pd.DataFrame:
    if rows_cont.size == 0:
        return pd.DataFrame(columns=feature_cols + [target_col])
    poly = np.rint(rows_cont[:, 0]).astype(int)  # integer polynomial sizes
    poly = np.clip(poly, 2, None)  # enforce minimum of 2
    spars = np.clip(rows_cont[:, 1], 0.0, 1.0)   # keep in [0,1]

    # Derive dependent features from integer size
    dist = np.array([dist_to_next_pow2(int(n)) for n in poly], dtype=int)
    p2 = np.array([is_power_of_2(int(n)) for n in poly], dtype=int)
    p4 = np.array([is_power_of_4(int(n)) for n in poly], dtype=int)

    out = pd.DataFrame({
        'Polynomial_Size': poly,
        'Sparsity': spars,
        'Dist_To_Next_Pow2': dist,
        'Is_Power_2': p2,
        'Is_Power_4': p4,
        'Best_Algorithm': label,
    })
    return out

syn_r2_df = build_synthetic_df(syn_r2_cont, 'Radix-2')
syn_rs_df = build_synthetic_df(syn_rs_cont, 'Radix-Split')

print(f"Generated synthetic → R2={len(syn_r2_df)}, RS={len(syn_rs_df)}")

# -----------------------------
"""
For 50/25/25, we will tag real vs synthetic pools, sample to target sizes,
write the balanced dataset, and export extras (synthetic RS/R2 actually used)
to extra_data.csv.
"""

EXTRA_OUTPUT = os.path.join(BASE_DIR, 'extra_data.csv')

# Target per-minority count so that, when combined with original data,
# MR4 ~= 50%, RS ~= 25%, R2 ~= 25%.
M = len(df_major)
minority_target_each = int(round(0.5 * M))

# Prepare tagged pools for RS and R2
rs_real = df_split[feature_cols + [target_col]].copy()
rs_real['is_synth'] = False
syn_rs_tagged_50 = syn_rs_df.copy()
if len(syn_rs_tagged_50) > 0:
    syn_rs_tagged_50['is_synth'] = True

r2_real = df_radix2[feature_cols + [target_col]].copy()
r2_real['is_synth'] = False
syn_r2_tagged_50 = syn_r2_df.copy()
if len(syn_r2_tagged_50) > 0:
    syn_r2_tagged_50['is_synth'] = True

rs_pool_tagged = pd.concat([rs_real, syn_rs_tagged_50], ignore_index=True)
r2_pool_tagged = pd.concat([r2_real, syn_r2_tagged_50], ignore_index=True)

rs_take = minority_target_each
r2_take = minority_target_each

rs_sample_tagged = rs_pool_tagged.sample(
    n=min(rs_take, len(rs_pool_tagged)),
    replace=(len(rs_pool_tagged) < rs_take),
    random_state=42
)
r2_sample_tagged = r2_pool_tagged.sample(
    n=min(r2_take, len(r2_pool_tagged)),
    replace=(len(r2_pool_tagged) < r2_take),
    random_state=42
)

# Export only the synthetic rows actually used in 50/25/25
extras_50 = pd.concat([
    rs_sample_tagged[rs_sample_tagged['is_synth'] == True],
    r2_sample_tagged[r2_sample_tagged['is_synth'] == True],
], ignore_index=True)

extras_export = extras_50[feature_cols + [target_col]].copy() if not extras_50.empty else pd.DataFrame(columns=feature_cols + [target_col])
extras_export.to_csv(EXTRA_OUTPUT, index=False)
print(f"Saved synthetic extras used for 50/25/25 to: {EXTRA_OUTPUT} ({len(extras_export)} rows)")

# -----------------------------
# 50/25/25 balanced dataset creation (DISABLED)
# -----------------------------
# The script previously built and saved a 50/25/25 balanced dataset to
# `fft_balanced_dataset.csv`. This block is intentionally disabled so the
# pipeline will not create or overwrite `fft_balanced_dataset.csv`. The
# `extra_data.csv` (synthetic rows used for the 50/25/25 selection) is
# still generated and exported above — keep that behavior for verification
# and merging.
#
# To re-enable creation of the 50/25/25 balanced dataset, restore the
# following logic: concatenate MR4 real samples with the selected RS and R2
# samples, shuffle, and save to `OUTPUT_BALANCED`.

# -----------------------------
# Equal 33/33/33 dataset generation (DISABLED)
# -----------------------------
# The code to build an equal 33/33/33 dataset (using both real and synthetic
# pools) has been intentionally disabled because the pipeline currently
# targets the 50/25/25 split and uses `extra_data.csv` for verification.
#
# To re-enable generation of the equal dataset, uncomment and restore the
# following logic which: builds pools for MR4/RS/R2 (including synthetic
# entries), computes the smallest pool size, samples each class to that
# size (with replacement only if necessary), concatenates and shuffles the
# result, then writes `fft_balanced_equal.csv`.
#
# -- disabled block start --
# (See git history or previous versions to restore if desired)
# -- disabled block end --

# -----------------------------
# 80/10/10 dataset generation (DISABLED)
# -----------------------------
# The code that generated an 80/10/10 dataset (MR4/RS/R2) was intentionally
# left here for reference but is currently disabled because the pipeline
# focuses on the 50/25/25 split and uses `extra_data.csv` for verification.
#
# To re-enable creation of the 80/10/10 dataset, uncomment the block below.
# The original implementation computed feasible sizes from the available
# pools, sampled with replacement if needed, and saved to
# `fft_balcnecd_ne.csv` (OUTPUT_BALANCED_80_10_10).

# -- disabled block start --
# (See git history or previous versions to restore if desired)
# -- disabled block end --

# No export here — extra_data.csv already holds 50/25/25 extras
