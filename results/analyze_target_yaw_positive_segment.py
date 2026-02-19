#!/usr/bin/env python3
import argparse
import os

import numpy as np
import pandas as pd


def analyze_csv(csv_path):
    df = pd.read_csv(csv_path)
    required_cols = {"t", "yaw_deg"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{csv_path} must include columns: {sorted(required_cols)}")

    # Sync to video timeline: first frame is 0.0s.
    t = df["t"].to_numpy(dtype=float)
    yaw = df["yaw_deg"].to_numpy(dtype=float)
    t_rel = t - t[0]

    pos_idx = np.where(yaw > 0.0)[0]
    if pos_idx.size == 0:
        return {
            "csv_path": csv_path,
            "status": "no_positive_yaw",
        }

    i_first = int(pos_idx[0])

    # Among positive yaw values, find minimum positive value.
    yaw_pos = yaw[pos_idx]
    local_min = int(np.argmin(yaw_pos))
    i_min_pos = int(pos_idx[local_min])

    yaw_first = float(yaw[i_first])
    yaw_min_pos = float(yaw[i_min_pos])
    t_first = float(t_rel[i_first])
    t_min_pos = float(t_rel[i_min_pos])

    dt = t_min_pos - t_first
    dyaw = yaw_min_pos - yaw_first
    yaw_rate = dyaw / dt if abs(dt) > 1e-12 else np.nan

    return {
        "csv_path": csv_path,
        "status": "ok",
        "first_positive_idx": i_first,
        "first_positive_time_sec": t_first,
        "first_positive_yaw_deg": yaw_first,
        "min_positive_idx": i_min_pos,
        "min_positive_time_sec": t_min_pos,
        "min_positive_yaw_deg": yaw_min_pos,
        "delta_time_sec": dt,
        "delta_yaw_deg": dyaw,
        "yaw_rate_deg_per_sec": yaw_rate,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze yaw change from first positive yaw to minimum positive yaw for LK cut-in scenarios."
    )
    parser.add_argument(
        "--base-dir",
        default="/home/user/SMPC_MMPreds/results",
        help="Base results directory.",
    )
    args = parser.parse_args()

    base = os.path.abspath(args.base_dir)
    csv_paths = [
        os.path.join(
            base,
            f"scenario_lk_cut_in{i}_ego_in_scenario_smpc_var_risk_p{2000 + 2*(i-1)}_g{i-1}",
            "target_0_state.csv",
        )
        for i in range(1, 5)
    ]

    print("[info] time is synchronized to video timeline (t_rel = t - t0)")
    for csv_path in csv_paths:
        print(f"\n=== {csv_path}")
        if not os.path.exists(csv_path):
            print("status: missing_file")
            continue

        result = analyze_csv(csv_path)
        if result["status"] != "ok":
            print(f"status: {result['status']}")
            continue

        print("status: ok")
        print(
            f"first_positive: idx={result['first_positive_idx']}, "
            f"t={result['first_positive_time_sec']:.3f}s, "
            f"yaw={result['first_positive_yaw_deg']:.6f} deg"
        )
        print(
            f"min_positive:   idx={result['min_positive_idx']}, "
            f"t={result['min_positive_time_sec']:.3f}s, "
            f"yaw={result['min_positive_yaw_deg']:.6f} deg"
        )
        print(
            f"delta: dt={result['delta_time_sec']:.3f}s, "
            f"dyaw={result['delta_yaw_deg']:.6f} deg, "
            f"yaw_rate={result['yaw_rate_deg_per_sec']:.6f} deg/s"
        )


if __name__ == "__main__":
    main()
