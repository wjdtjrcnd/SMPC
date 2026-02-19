#!/usr/bin/env python3
import argparse
import os
import pickle

import numpy as np
import pandas as pd


def convert_one(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    outdir = os.path.dirname(os.path.abspath(pkl_path))

    for actor, rec in data.items():
        state = pd.DataFrame(rec["state_trajectory"], columns=["t", "x", "y", "yaw_deg", "v"])
        state["yaw_deg"] = np.degrees(state["yaw_deg"])
        state.to_csv(os.path.join(outdir, f"{actor}_state.csv"), index=False)

        pd.DataFrame(rec["input_trajectory"], columns=["a", "steer"]).to_csv(
            os.path.join(outdir, f"{actor}_input.csv"), index=False
        )
        pd.DataFrame({"feasible": rec["feasibility"]}).to_csv(
            os.path.join(outdir, f"{actor}_feasibility.csv"), index=False
        )
        pd.DataFrame({"solve_time": rec["solve_times"]}).to_csv(
            os.path.join(outdir, f"{actor}_solve_times.csv"), index=False
        )

    print(f"[OK] converted: {pkl_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert scenario_lk_cut_in1~4 scenario_result.pkl files to CSV."
    )
    parser.add_argument(
        "--base-dir",
        default="/home/user/SMPC_MMPreds/results",
        help="Base results directory (default: /home/user/SMPC_MMPreds/results)",
    )
    args = parser.parse_args()

    base = os.path.abspath(args.base_dir)
    pkl_paths = [
        os.path.join(
            base,
            f"scenario_lk_cut_in{i}_ego_in_scenario_smpc_var_risk_p{2000 + 2*(i-1)}_g{i-1}",
            "scenario_result.pkl",
        )
        for i in range(1, 5)
    ]

    missing = [p for p in pkl_paths if not os.path.exists(p)]
    if missing:
        print("[WARN] missing files:")
        for m in missing:
            print(f"  - {m}")

    found = [p for p in pkl_paths if os.path.exists(p)]
    if not found:
        raise FileNotFoundError("No scenario_result.pkl found for cut_in1~4.")

    for p in found:
        convert_one(p)

    print(f"[DONE] converted {len(found)} file(s).")


if __name__ == "__main__":
    main()
