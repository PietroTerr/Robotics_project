import utils as u


def main():
    # Load the map once — shared by all robots
    map_api = u.load_map(csv_path="map_001_seed42.csv", rng_seed=42)


    start  = (3, 3)
    # The map indices are 0 to 49 for a 50x50 map, so target must be at most (49, 49)
    # otherwise the robots will get permanently stuck in the corner!
    target = (49, 49)

    # ------------------------------------------------------------------
    # 1) DRONE — fly repeatedly until the target cell is explored
    # ------------------------------------------------------------------
    all_paths, all_obs, all_battery, n_flights = u.explore_until_target_found(
        map_api,
        start=start,
        target=target,
        speed=2.0,
    )
    print(f"Drone finished in {n_flights} flight(s).\n")

    # ------------------------------------------------------------------
    # 2) ROVER — now drive to the explored target
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  PHASE 2: Rover driving to target")
    print("=" * 60)

    final_pos, elapsed, stuck = u.move_rover_to_target(
        map_api,
        robot_id="rover_1",
        start=start,
        target=target,
        speed=0.1,
        dt=0.01,
    )
    print(
        f"\n[rover] Done. Final pos: ({final_pos[0]:.2f}, {final_pos[1]:.2f}), "
        f"Time: {elapsed:.1f}s, Stuck: {stuck}"
    )


if __name__ == "__main__":
    main()
