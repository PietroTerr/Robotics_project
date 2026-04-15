import utilis_v2 as u2

def main_v2():
    map_api = u2.load_map("map_001_seed42.csv", rng_seed=42)

    start = (3, 3)
    target = (49, 49)

    path, obs, memory, stuck_cells = u2.scout_coverage_exploration(
        map_api,
        robot_id="scout_1",
        start=start,
        target=target,
    )

    # Find where Phase 1 ended (scout reached target)
    # by counting steps until scout was near the target
    phase1_steps = 0
    tx, ty = float(target[0]), float(target[1])
    for i, (px, py) in enumerate(path):
        if (tx - px)**2 + (ty - py)**2 < 0.5**2:
            phase1_steps = i
            break

    u2.plot_scout_results(
        path, memory,
        target=target,
        start=start,
        phase1_steps=phase1_steps,
        stuck_cells=stuck_cells,
    )

if __name__ == "__main__":
    main_v2()
