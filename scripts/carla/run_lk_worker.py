import argparse
import json
import os
import signal
import sys

from run_all_scenarios import _safe_name, run_with_tvs


STOP_REQUESTED = False


def _handle_sigint(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[run_lk_worker] Ctrl+C detected. Stopping...", flush=True)


def run_worker(default_scenario_name, default_port, default_gpu):
    parser = argparse.ArgumentParser(description="Run one LK scenario on a fixed CARLA port/GPU.")
    parser.add_argument("--scenario", default=default_scenario_name, help="Scenario json filename under scripts/carla/scenarios/")
    parser.add_argument("--port", type=int, default=default_port, help="CARLA server port.")
    parser.add_argument("--gpu", type=int, default=default_gpu, help="GPU index for this worker (CUDA_VISIBLE_DEVICES).")
    parser.add_argument("--ip", default="localhost", help="CARLA server IP.")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    scenario_path = os.path.join(script_dir, "scenarios", args.scenario)
    if not os.path.exists(scenario_path):
        raise FileNotFoundError(f"Scenario not found: {scenario_path}")

    with open(scenario_path, "r") as f:
        scenario_dict = json.load(f)

    # Force LK execution settings for this worker.
    scenario_dict.setdefault("carla_params", {})
    scenario_dict["carla_params"]["port"] = args.port
    scenario_dict["carla_params"]["ip_addr"] = args.ip

    # Resolve relative csv path relative to scenario json directory (if used).
    csv_loc = scenario_dict.get("carla_params", {}).get("intersection_csv_loc", "")
    if csv_loc and not os.path.isabs(csv_loc):
        csv_candidate = os.path.join(os.path.dirname(scenario_path), csv_loc)
        if os.path.exists(csv_candidate):
            scenario_dict["carla_params"]["intersection_csv_loc"] = csv_candidate

    results_folder = os.path.join(os.path.abspath(__file__).split("scripts")[0], "results")
    os.makedirs(results_folder, exist_ok=True)

    scenario_name = _safe_name(os.path.splitext(os.path.basename(scenario_path))[0])
    ego_init_name = "ego_in_scenario"
    ego_policy_config = "smpc_var_risk"
    savedir = os.path.join(
        results_folder,
        f"{scenario_name}_{ego_init_name}_{ego_policy_config}_p{args.port}_g{args.gpu}",
    )

    print(
        "****************\n"
        f"{scenario_name}_{ego_init_name}_{ego_policy_config} (port={args.port}, gpu={args.gpu})\n"
        "****************\n",
        flush=True,
    )

    if not STOP_REQUESTED:
        run_with_tvs("highway", scenario_dict, {}, ego_policy_config, savedir)


if __name__ == "__main__":
    run_worker("scenario_lk_cut_in.json", 2000, 0)
