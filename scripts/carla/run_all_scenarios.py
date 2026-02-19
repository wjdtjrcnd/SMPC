import os
import glob
import json
import pdb
import signal
import sys

# Reduce TensorFlow C++ backend logs unless the user overrides it explicitly.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Use a sensible default CARLA install path when CARLA_ROOT is not exported.
os.environ.setdefault("CARLA_ROOT", os.path.expanduser("~/carla/0.9.10"))


STOP_REQUESTED = False


def _handle_sigint(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[run_all_scenarios] Ctrl+C detected. Stopping after current step...", flush=True)


def _safe_name(name):
    text = str(name)
    for ch in ["[", "]", "'", "\"", " ", ","]:
        text = text.replace(ch, "")
    return text


def _strip_meta_keys(d):
    return {k: v for k, v in d.items() if not str(k).startswith("_")}




def run_without_tvs(scene, scenario_dict, ego_init_dict, savedir, get_cl=False):
    if scene =="intersection":
        from scenarios.run_intersection_scenario import CarlaParams, DroneVizParams, VehicleParams, PredictionParams, RunIntersectionScenario
    else:
        from scenarios.run_lk_scenario import CarlaParams, DroneVizParams, VehicleParams, PredictionParams, RunLKScenario


    carla_params     = CarlaParams(**_strip_meta_keys(scenario_dict["carla_params"]))
    drone_viz_params = DroneVizParams(**_strip_meta_keys(scenario_dict["drone_viz_params"]))
    pred_params      = PredictionParams()

    vehicles_params_list = []

    for vp_dict in scenario_dict["vehicle_params"]:
        vp_dict = _strip_meta_keys(vp_dict)
        if vp_dict["role"] == "static":
            continue
            # vehicles_params_list.append( VehicleParams(**vp_dict) )
        elif "target" in vp_dict["role"]:
            pass
        elif vp_dict["role"] == "ego":
            if get_cl:
                vp_dict['goal_left_offset']=0.0
            vp_dict.update(ego_init_dict)
            vp_dict["policy_type"] = "blsmpc"
            vp_dict["smpc_config"] = ""
            vehicles_params_list.append( VehicleParams(**vp_dict) )
        else:

            raise ValueError(f"Invalid vehicle role: {vp_dict['role']}")

    if scene =="intersection":
        runner = RunIntersectionScenario(carla_params,
                                        drone_viz_params,
                                        vehicles_params_list,
                                        pred_params,
                                        savedir)
    else:
        runner = RunLKScenario(carla_params,
                                        drone_viz_params,
                                        vehicles_params_list,
                                        pred_params,
                                        savedir)
    
    runner.run_scenario()

def run_with_tvs(scene, scenario_dict, ego_init_dict, ego_policy_config, savedir):
    if scene =="intersection":
        from scenarios.run_intersection_scenario import CarlaParams, DroneVizParams, VehicleParams, PredictionParams, RunIntersectionScenario
    else:
        from scenarios.run_lk_scenario import CarlaParams, DroneVizParams, VehicleParams, PredictionParams, RunLKScenario
    
    
    carla_params     = CarlaParams(**_strip_meta_keys(scenario_dict["carla_params"]))
    drone_viz_params = DroneVizParams(**_strip_meta_keys(scenario_dict["drone_viz_params"]))
    pred_params      = PredictionParams()

    vehicles_params_list = []

    if ego_policy_config == "blsmpc":
        policy_type   = "blsmpc"
        policy_config = ""
    elif ego_policy_config.startswith("smpc"):
        policy_type = "smpc"
        policy_config = ego_policy_config.split("smpc_")[-1]
    elif ego_policy_config == "mpc":
        policy_type = "mpc"
        policy_config = ""
    else:
        raise ValueError(f"Invalid ego policy config: {ego_policy_config}")

    for vp_dict in scenario_dict["vehicle_params"]:
        vp_dict = _strip_meta_keys(vp_dict)
        if vp_dict["role"] == "static":
            # Ignore static vehicles in intersection scenarios.
            if scene == "intersection":
                continue
            vehicles_params_list.append( VehicleParams(**vp_dict) )
        elif "target" in vp_dict["role"]:
            vehicles_params_list.append( VehicleParams(**vp_dict) )
        elif vp_dict["role"] == "ego":
         
            vp_dict.update(ego_init_dict)
            vp_dict["policy_type"] = policy_type
            vp_dict["smpc_config"] = policy_config
            vehicles_params_list.append( VehicleParams(**vp_dict) )
        else:

            raise ValueError(f"Invalid vehicle role: {vp_dict['role']}")

    if scene == "intersection":
        runner = RunIntersectionScenario(carla_params,
                                        drone_viz_params,
                                        vehicles_params_list,
                                        pred_params,
                                        savedir)
    else:
        runner = RunLKScenario(carla_params,
                                     drone_viz_params,
                                     vehicles_params_list,
                                     pred_params,
                                     savedir)
    runner.run_scenario()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, _handle_sigint)

    scenario_folder = os.path.join( os.path.dirname( os.path.abspath(__file__)  ), "scenarios/" )
    scenarios_list = sorted(glob.glob(os.path.join(scenario_folder, "scenario_lk_cut_in.json")))
    results_folder = os.path.join( os.path.abspath(__file__).split("scripts")[0], "results" )

    try:
        for scenario in scenarios_list:
            if STOP_REQUESTED:
                break

            # Load the scenario and generate parameters.
            with open(scenario, "r") as f:
                scenario_dict = json.load(f)
            scenario_dir = os.path.dirname(os.path.abspath(scenario))
            scenario_name = _safe_name(os.path.splitext(os.path.basename(scenario))[0])
            if "lk" in scenario_name:
                scene = "highway"
            else:
                scene = "intersection"

            # For LK scenarios, resolve CSV relative to each scenario json file.
            if scene == "highway":
                csv_loc = scenario_dict.get("carla_params", {}).get("intersection_csv_loc", "")
                if csv_loc and not os.path.isabs(csv_loc):
                    csv_candidate = os.path.join(scenario_dir, csv_loc)
                    if os.path.exists(csv_candidate):
                        scenario_dict["carla_params"]["intersection_csv_loc"] = csv_candidate

            ego_init_runs = []
            if scene == "highway":
                # LK scenarios use ego params directly from scenario json.
                ego_init_runs = [("ego_in_scenario", {})]
            else:
                inits_folder = os.path.join( os.path.dirname( os.path.abspath(__file__)  ), "scenarios/inits/" )
                ego_init_names = scenario_dict.get("ego_init_jsons", [])
                ego_init_list = []

                # Prefer scenario-defined init files for non-LK scenarios.
                for init_name in ego_init_names:
                    init_candidates = [
                        os.path.join(scenario_dir, init_name),
                        os.path.join(inits_folder, init_name),
                    ]
                    for cand in init_candidates:
                        if os.path.exists(cand):
                            ego_init_list.append(cand)
                            break

                # Backward-compatible fallback for intersection scenarios.
                if not ego_init_list:
                    ego_init_list = sorted(glob.glob(os.path.join(inits_folder, "ego_init_01.json")))

                for ego_init in ego_init_list:
                    with open(ego_init, "r") as f:
                        ego_init_dict = json.load(f)
                    ego_init_name = _safe_name(os.path.splitext(os.path.basename(ego_init))[0])
                    ego_init_runs.append((ego_init_name, ego_init_dict))

            for ego_init_name, ego_init_dict in ego_init_runs:
                if STOP_REQUESTED:
                    break

                # Run only var-risk SMPC for now.
                for ego_policy_config in ["smpc_var_risk"]:
                    if STOP_REQUESTED:
                        break

                    savedir = os.path.join( results_folder,
                                            f"{scenario_name}_{ego_init_name}_{ego_policy_config}")
                    print("****************\n"
                        f"{scenario_name}_{ego_init_name}_{ego_policy_config}\n"
                        "****************\n")
                    run_with_tvs(scene, scenario_dict, ego_init_dict, ego_policy_config, savedir)
    except KeyboardInterrupt:
        print("\n[run_all_scenarios] Interrupted by user.", flush=True)
        sys.exit(130)
