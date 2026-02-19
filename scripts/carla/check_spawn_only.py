import argparse
import glob
import json
import os
import queue
import sys
import time
from types import SimpleNamespace


def _bootstrap_carla_pythonapi(default_root="~/carla/0.9.10"):
    carla_root = os.environ.get("CARLA_ROOT", os.path.expanduser(default_root))
    os.environ.setdefault("CARLA_ROOT", carla_root)

    py_paths = [
        os.path.join(carla_root, "PythonAPI"),
        os.path.join(carla_root, "PythonAPI", "carla"),
        os.path.join(carla_root, "PythonAPI", "carla", "agents"),
    ]
    for p in py_paths:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)

    egg_glob = os.path.join(carla_root, "PythonAPI", "carla", "dist", "carla-*.egg")
    eggs = sorted(glob.glob(egg_glob))
    if eggs:
        egg = eggs[-1]
        if egg not in sys.path:
            sys.path.append(egg)


_bootstrap_carla_pythonapi()

import carla
import cv2
import numpy as np

DEFAULT_SCENARIO_JSON = "scripts/carla/scenarios/scenario_lk_cut_in3.json"


def _prefer_preds_path(path):
    """If a *_origin path is resolved, prefer the sibling SMPC_MMPreds path when it exists."""
    p = os.path.abspath(path)
    alt = p.replace("/SMPC_MMPreds_origin/", "/SMPC_MMPreds/")
    if alt != p and os.path.exists(alt):
        return alt
    return p


def _load_json(path):
    with open(_prefer_preds_path(path), "r") as f:
        return json.load(f)


def _get_scenario_utils(scenario_path, scenario_dict):
    scenario_name = os.path.basename(scenario_path).lower()
    is_lk = ("lk" in scenario_name)
    if is_lk:
        from scenarios.run_lk_scenario import get_intersection_transform
    else:
        from scenarios.run_intersection_scenario import get_intersection_transform
    return get_intersection_transform


def _load_intersection_from_points(intersection_points):
    intersection = []
    for row in intersection_points:
        if len(row) != 3:
            raise ValueError("Each ref_positions row must have 3 values: x, y, yaw")
        start_pose = [float(row[0]), float(row[1]), int(row[2])]
        goal_pose = [float(row[0]), float(row[1]), int(row[2])]
        intersection.append([start_pose, goal_pose])
    return intersection


def _setup_drone(world, scenario_dict):
    viz = scenario_dict.get("drone_viz_params", {})
    bp_library = world.get_blueprint_library()
    bp_drone = bp_library.find("sensor.camera.rgb")
    img_w = int(viz.get("img_width", 1920))
    img_h = int(viz.get("img_height", 1080))
    fov = float(viz.get("fov", 90))
    bp_drone.set_attribute("image_size_x", str(img_w))
    bp_drone.set_attribute("image_size_y", str(img_h))
    bp_drone.set_attribute("fov", str(fov))
    bp_drone.set_attribute("role_name", "spawn_check_drone")

    cam_tf = carla.Transform(
        carla.Location(
            x=float(viz.get("x", -30.0)),
            y=float(viz.get("y", 0.0)),
            z=float(viz.get("z", 50.0)),
        ),
        carla.Rotation(
            roll=float(viz.get("roll", 0.0)),
            pitch=float(viz.get("pitch", -90.0)),
            yaw=float(viz.get("yaw", 0.0)),
        ),
    )
    drone = world.spawn_actor(bp_drone, cam_tf)
    return drone, img_w, img_h


def _image_to_bgr(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    return arr[:, :, :3]


def main():
    parser = argparse.ArgumentParser(description="Spawn-only checker for CARLA scenarios.")
    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO_JSON,
        help=f"Path to scenario json file. Default: {DEFAULT_SCENARIO_JSON}",
    )
    parser.add_argument("--hold-sec", type=float, default=3.0, help="Seconds to keep actors spawned.")
    parser.add_argument("--focus-ego", action="store_true", help="Move spectator camera above ego after spawn.")
    parser.add_argument(
        "--show-goal",
        action="store_true",
        help="Spawn start first, then goal positions (each for --hold-sec).",
    )
    parser.add_argument(
        "--drive-to-goal",
        action="store_true",
        help="Spawn at start and move actors toward goal while optionally recording drone video.",
    )
    parser.add_argument("--drive-speed", type=float, default=-1.0, help="Override speed (m/s). If negative, use nominal_speed/init_speed from scenario.")
    parser.add_argument("--max-drive-sec", type=float, default=30.0, help="Max simulation time for --drive-to-goal.")
    parser.add_argument("--goal-tol", type=float, default=2.0, help="Distance tolerance (m) for goal reached in --drive-to-goal.")
    parser.add_argument("--save-avi", action="store_true", help="Save drone-view video (useful with --drive-to-goal).")
    parser.add_argument("--avi-path", default="", help="Output AVI path. Default: /home/user/SMPC_MMPreds/results/spawn_check_<scenario>.avi")
    args = parser.parse_args()

    scenario_path = _prefer_preds_path(args.scenario)
    scenario_dict = _load_json(scenario_path)
    ego_init_dict = {}
    get_intersection_transform = _get_scenario_utils(scenario_path, scenario_dict)

    carla_params = scenario_dict["carla_params"]
    ref_positions = carla_params.get("ref_positions")
    if ref_positions is None:
        ref_positions = carla_params.get("ref_posisions")
    if ref_positions is None and carla_params.get("intersection_points"):
        ref_positions = [row[:3] for row in carla_params["intersection_points"]]
    if not ref_positions:
        raise ValueError("Scenario JSON must include carla_params.ref_positions.")
    intersection = _load_intersection_from_points(ref_positions)

    client = carla.Client(carla_params.get("ip_addr", "localhost"), carla_params.get("port", 2004))
    client.set_timeout(carla_params.get("timeout_period", 2.0))
    world = client.load_world(carla_params["map_str"])
    world.set_weather(getattr(carla.WeatherParameters, carla_params["weather_str"]))

    bp_library = world.get_blueprint_library()
    print(f"[spawn-check] scenario={scenario_path}")
    if args.drive_to_goal:
        phases = ["start"]
    else:
        phases = ["start", "goal"] if args.show_goal else ["start"]
    for endpoint in phases:
        spawned = []
        goal_by_actor_id = {}
        speed_by_actor_id = {}
        drone = None
        writer = None
        img_queue = None
        original_settings = None
        print(f"[spawn-check] phase={endpoint}")
        for idx, vp in enumerate(scenario_dict["vehicle_params"]):
            vp_local = dict(vp)
            if vp_local.get("role") == "ego":
                vp_local.update(ego_init_dict)

            vp_ns = SimpleNamespace(**vp_local)
            veh_bp = bp_library.find(vp_ns.vehicle_type)
            veh_bp.set_attribute("color", vp_ns.vehicle_color)
            veh_bp.set_attribute("role_name", vp_ns.role)
            spawn_tf = get_intersection_transform(intersection, vp_ns, endpoint)
            start_tf = get_intersection_transform(intersection, vp_ns, "start")
            goal_tf = get_intersection_transform(intersection, vp_ns, "goal")
            wp = world.get_map().get_waypoint(
                spawn_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            road_z = None
            base_z = float(spawn_tf.location.z)
            z_candidates = [base_z]
            if wp is not None:
                road_z = float(wp.transform.location.z)
                z_candidates = [road_z + 0.5, road_z + 1.5, base_z, base_z + 0.5, base_z + 2.0]

            actor = None
            used_z = None
            for z_try in z_candidates:
                spawn_tf.location.z = z_try
                actor = world.try_spawn_actor(veh_bp, spawn_tf)
                if actor is not None:
                    used_z = z_try
                    break

            if actor is None:
                print(
                    f"[FAILED] idx={idx} role={vp_ns.role} type={vp_ns.vehicle_type} "
                    f"at ({spawn_tf.location.x:.2f}, {spawn_tf.location.y:.2f}, z_try={z_candidates}, yaw={spawn_tf.rotation.yaw:.1f})"
                )
                continue

            spawned.append(actor)
            goal_by_actor_id[actor.id] = goal_tf
            desired_speed = float(args.drive_speed) if args.drive_speed >= 0.0 else float(
                getattr(vp_ns, "nominal_speed", getattr(vp_ns, "init_speed", 4.0))
            )
            speed_by_actor_id[actor.id] = max(0.0, desired_speed)
            actor_tf = actor.get_transform()
            print(
                f"[OK] idx={idx} role={vp_ns.role} actor_id={actor.id} "
                f"spawn_{endpoint}=({spawn_tf.location.x:.2f}, {spawn_tf.location.y:.2f}, z={used_z:.2f}, yaw={spawn_tf.rotation.yaw:.1f}) "
                f"goal=({goal_tf.location.x:.2f}, {goal_tf.location.y:.2f}, yaw={goal_tf.rotation.yaw:.1f}) "
                f"goal_longitudinal_offset={float(getattr(vp_ns, 'goal_longitudinal_offset', 0.0)):.2f} "
                f"actor_z={actor_tf.location.z:.2f} road_z={road_z if road_z is not None else float('nan'):.2f}"
            )

        if args.focus_ego:
            ego_actor = None
            for actor in spawned:
                if actor.attributes.get("role_name") == "ego":
                    ego_actor = actor
                    break
            if ego_actor is not None:
                print("[spawn-check] spectator will track ego during hold.")
            else:
                print("[spawn-check] no ego actor found; will track first spawned actor.")

        print(f"[spawn-check] spawned={len(spawned)}")
        try:
            if args.drive_to_goal and spawned:
                drone, img_w, img_h = _setup_drone(world, scenario_dict)
                img_queue = queue.Queue()
                drone.listen(img_queue.put)

                if args.save_avi:
                    if args.avi_path:
                        avi_path = _prefer_preds_path(args.avi_path)
                    else:
                        scenario_base = os.path.splitext(os.path.basename(scenario_path))[0]
                        avi_path = os.path.join(
                            os.path.abspath(__file__).split("scripts")[0],
                            "results",
                            f"spawn_check_{scenario_base}.avi",
                        )
                    os.makedirs(os.path.dirname(avi_path), exist_ok=True)
                    writer = cv2.VideoWriter(
                        avi_path, cv2.VideoWriter_fourcc(*"MJPG"), float(carla_params.get("fps", 20)), (img_w, img_h)
                    )
                    print(f"[spawn-check] recording drone video to: {avi_path}")

                original_settings = world.get_settings()
                new_settings = world.get_settings()
                new_settings.synchronous_mode = True
                new_settings.fixed_delta_seconds = 1.0 / float(carla_params.get("fps", 20))
                world.apply_settings(new_settings)

                max_ticks = int(max(1.0, args.max_drive_sec) * float(carla_params.get("fps", 20)))
                for _ in range(max_ticks):
                    all_done = True
                    for actor in spawned:
                        goal_tf = goal_by_actor_id[actor.id]
                        loc = actor.get_location()
                        dx = goal_tf.location.x - loc.x
                        dy = goal_tf.location.y - loc.y
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist > args.goal_tol:
                            all_done = False
                            spd = speed_by_actor_id[actor.id]
                            vx = spd * dx / max(dist, 1e-6)
                            vy = spd * dy / max(dist, 1e-6)
                            actor.set_target_velocity(carla.Vector3D(x=vx, y=vy, z=0.0))
                        else:
                            actor.set_target_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))

                    world.tick()

                    if args.focus_ego:
                        track_actor = ego_actor if ego_actor is not None else spawned[0]
                        ego_tf = track_actor.get_transform()
                        world.get_spectator().set_transform(
                            carla.Transform(
                                carla.Location(x=ego_tf.location.x, y=ego_tf.location.y, z=ego_tf.location.z + 40.0),
                                carla.Rotation(pitch=-90.0, yaw=180.0),
                            )
                        )

                    if writer is not None and img_queue is not None:
                        try:
                            image = img_queue.get(timeout=1.0)
                            writer.write(_image_to_bgr(image))
                        except queue.Empty:
                            pass

                    if all_done:
                        print("[spawn-check] all actors reached goals within tolerance.")
                        break
            else:
                print(f"[spawn-check] hold_sec={args.hold_sec}")
                end_t = time.time() + max(0.0, args.hold_sec)
                while time.time() < end_t:
                    if args.focus_ego:
                        track_actor = ego_actor if ego_actor is not None else (spawned[0] if spawned else None)
                        if track_actor is None:
                            time.sleep(0.1)
                            continue
                        ego_tf = track_actor.get_transform()
                        spec_tf = carla.Transform(
                            carla.Location(x=ego_tf.location.x, y=ego_tf.location.y, z=ego_tf.location.z + 40.0),
                            carla.Rotation(pitch=-90.0, yaw=180.0),
                        )
                        world.get_spectator().set_transform(spec_tf)
                    time.sleep(0.1)
        finally:
            if drone is not None:
                drone.stop()
            if writer is not None:
                writer.release()
            if drone is not None:
                drone.destroy()
            if original_settings is not None:
                world.apply_settings(original_settings)
            for actor in spawned:
                actor.destroy()
            print(f"[spawn-check] cleaned up actors for phase={endpoint}.")


if __name__ == "__main__":
    main()
