#!/usr/bin/env python3
import csv
"""Merge verified synthetic FFT rows with original performance dataset.

Algorithm summary:
1. Read original performance CSV (fft_performance_results.csv).
2. Read verifier output (verify_extra_data.csv).
3. Remove the 'Padded_Size' column from original rows (case-insensitive).
4. Keep only synthetic rows where predicted Best_Algorithm matches Actual_Best_Algorithm.
5. Drop 'Actual_Best_Algorithm' and unify columns; ensure timing columns present.
6. Write merged clean dataset to 'fft_clean_extradata.csv'.

This file performs deterministic merging; no randomness, no training.
"""
from pathlib import Path

BASE = Path(__file__).resolve().parent
perf_path = BASE / 'fft_performance_results.csv'
verify_path = BASE / 'verify_extra_data.csv'
out_path = BASE / 'fft_clean_extradata.csv'

def read_csv(path: Path):
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return [h.strip() for h in (reader.fieldnames or [])], [
            {k.strip(): v for k, v in r.items()} for r in rows
        ]

def normalize_algo(name: str) -> str:
    s = (name or '').strip().lower()
    # normalize split/radix naming variants
    if s == 'radix-split' or s == 'split-radix':
        return 'split-radix'
    return s

def main():
    # Read both CSVs
    perf_headers, perf_rows = read_csv(perf_path)
    verify_headers, verify_rows = read_csv(verify_path)

    # Drop Padded_Size from performance rows (case-insensitive)
    perf_rows = [
        {k: v for k, v in r.items() if k.lower() != 'padded_size'}
        for r in perf_rows
    ]

    # Filter verify rows: keep only where Best_Algorithm == Actual_Best_Algorithm
    filtered_verify = []
    for r in verify_rows:
        b = normalize_algo(r.get('Best_Algorithm', ''))
        a = normalize_algo(r.get('Actual_Best_Algorithm', ''))
        if b and a and b == a:
            filtered_verify.append(r)

    # Desired output columns, without Actual_Best_Algorithm
    desired = [
        'Polynomial_Size','Sparsity','Dist_To_Next_Pow2','Is_Power_2','Is_Power_4',
        'Radix_2_Time_ms','Modified_Radix_4_Time_ms','Radix_Split_Time_ms',
        'Best_Algorithm'
    ]

    # Union of headers (excluding Padded_Size and Actual_Best_Algorithm)
    union_set = set(perf_rows[0].keys()) if perf_rows else set()
    union_set |= {h for h in verify_headers if h != 'Actual_Best_Algorithm'}
    union = [h for h in desired if h in union_set] + [h for h in union_set if h not in desired]

    # Write output
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=union)
        writer.writeheader()
        # Write performance rows
        for r in perf_rows:
            # Ensure Actual_Best_Algorithm not present
            r.pop('Actual_Best_Algorithm', None)
            writer.writerow({k: r.get(k, '') for k in union})
        # Write filtered verify rows; ensure timing columns present but may be empty
        for r in filtered_verify:
            for k in ['Radix_2_Time_ms','Modified_Radix_4_Time_ms','Radix_Split_Time_ms']:
                r.setdefault(k, '')
            r.pop('Actual_Best_Algorithm', None)
            writer.writerow({k: r.get(k, '') for k in union})

    print(f"Wrote: {out_path}")
    print(f"Perf rows: {len(perf_rows)}, Verify rows (matched): {len(filtered_verify)}, Total: {len(perf_rows)+len(filtered_verify)}")

if __name__ == '__main__':
    main()
