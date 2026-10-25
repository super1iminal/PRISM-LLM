import random
import csv
import argparse
from collections import deque


# ---------- 1. BFS path finder for sequential goals (S -> 1 -> 2 -> 3) ----------

def is_solvable_sequence_with_path(n, start, goals, static_obstacles):
    """
    Returns (True, full_path) where full_path visits goals in order
    AND never steps on future goals early.

    Future goals are treated as blocked in earlier segments.
    """
    blocked_static = set(static_obstacles)
    goal_keys = sorted(goals.keys())
    goal_list = [goals[k] for k in goal_keys]

    # sanity
    if start in blocked_static:
        return (False, None)
    for g in goal_list:
        if g in blocked_static:
            return (False, None)

    def neighbors(r, c):
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                yield nr, nc

    full_path = []
    current_start = start

    # BFS each segment, blocking future goals
    for seg_idx, goal in enumerate(goal_list):
        # future goals are blocked for this segment
        future_goals = set(goal_list[seg_idx+1:])
        blocked = blocked_static | future_goals

        q = deque([current_start])
        visited = {current_start}
        parent = {current_start: None}
        found = False

        while q:
            r, c = q.popleft()
            if (r, c) == goal:
                found = True
                break

            for nr, nc in neighbors(r, c):
                nxt = (nr, nc)
                if nxt in blocked or nxt in visited:
                    continue
                visited.add(nxt)
                parent[nxt] = (r, c)
                q.append(nxt)

        if not found:
            return (False, None)

        # reconstruct this segment path
        seg_path = []
        node = goal
        while node is not None:
            seg_path.append(node)
            node = parent[node]
        seg_path.reverse()

        # avoid duplicating junction node
        if full_path:
            seg_path = seg_path[1:]

        full_path.extend(seg_path)
        current_start = goal

    return (True, full_path)


# ---------- 2. Helper: plausibility check around S and final goal ----------

def count_blocked_neighbors(n, cell, blocked):
    r, c = cell
    neighbors = [
        (r - 1, c), (r + 1, c),
        (r, c - 1), (r, c + 1),
    ]
    count = 0
    for nr, nc in neighbors:
        if 0 <= nr < n and 0 <= nc < n and (nr, nc) in blocked:
            count += 1
    return count


# ---------- 3. Generate a moving obstacle path ----------

def generate_moving_obstacle(n, start, goals, static_obstacles, max_tries=50):
    """
    Generate a linear or L-shaped moving obstacle path of 2-4 waypoints.

    Waypoints must not overlap start, goals, or static obstacles.
    The moving obstacle must not block solvability: BFS is re-run with all
    trajectory cells added to the blocked set.

    Returns a list of waypoints, or [] if generation fails.
    """
    occupied = {start} | set(goals.values()) | set(static_obstacles)
    free_cells = [(r, c) for r in range(n) for c in range(n) if (r, c) not in occupied]

    if len(free_cells) < 2:
        return []

    def grid_neighbors(r, c):
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                yield nr, nc

    for _ in range(max_tries):
        num_waypoints = random.randint(2, min(4, len(free_cells)))
        # Pick a random starting waypoint
        wp_start = random.choice(free_cells)
        waypoints = [wp_start]

        # Grow path by walking to adjacent free cells
        for _ in range(num_waypoints - 1):
            r, c = waypoints[-1]
            candidates = [
                (nr, nc) for nr, nc in grid_neighbors(r, c)
                if (nr, nc) not in occupied and (nr, nc) not in waypoints
            ]
            if not candidates:
                break
            waypoints.append(random.choice(candidates))

        if len(waypoints) < 2:
            continue

        # Expand the path to get all trajectory cells (back-and-forth)
        if waypoints[-1] == waypoints[0]:
            trajectory = set(waypoints[:-1])
        else:
            trajectory = set(waypoints) | set(waypoints[-2:0:-1])

        # Check that the grid is still solvable with trajectory cells blocked
        all_blocked = list(static_obstacles) + list(trajectory)
        ok, _ = is_solvable_sequence_with_path(n, start, goals, all_blocked)
        if ok:
            return waypoints

    return []


# ---------- 4. Generate ONE solvable grid with sequential goals + BFS path ----------

def generate_grid_with_sequential_goals(n, num_static, num_goals=3,
                                         moving_obstacle_prob=0.5,
                                         max_tries=200):
    """
    Generates an n*n grid with:
      - start S = (0,0)
      - num_goals ordered goals labeled 1..num_goals
      - num_static static obstacles X
      - optional moving obstacle (for n >= 4)

    Ensures a solvable ordered path exists:
      S -> goal1 -> goal2 -> ... -> goalk

    Returns:
      (grid_str, goals_dict, static_obstacles, visit_order_path, moving_waypoints)
    """
    start = (0, 0)

    if n == 1:
        grid_str = "  0\n0 S"
        goals = {1: start}
        return grid_str, goals, [], [start], []

    for attempt in range(max_tries):
        # all cells excluding start
        all_cells = [(i, j) for i in range(n) for j in range(n)]
        all_cells.remove(start)

        if len(all_cells) < num_goals:
            raise ValueError("Grid too small for requested number of goals.")

        # pick distinct goal cells
        goal_cells = random.sample(all_cells, num_goals)
        goals = {k+1: goal_cells[k] for k in range(num_goals)}

        # remaining cells for obstacles can't include goals
        remaining_cells = [c for c in all_cells if c not in goal_cells]
        if num_static > len(remaining_cells):
            continue

        random.shuffle(remaining_cells)
        static_obstacles = remaining_cells[:num_static]

        # ordered solvability + get BFS path
        ok, visit_order = is_solvable_sequence_with_path(
            n, start, goals, static_obstacles
        )
        if not ok:
            continue

        # local plausibility: don't box in S or final goal too much
        blocked = set(static_obstacles)
        if count_blocked_neighbors(n, start, blocked) > 2:
            continue
        final_goal = goals[num_goals]
        if count_blocked_neighbors(n, final_goal, blocked) > 2:
            continue

        # Optionally generate moving obstacle (only for n >= 4)
        moving_waypoints = []
        if n >= 4 and random.random() < moving_obstacle_prob:
            moving_waypoints = generate_moving_obstacle(
                n, start, goals, static_obstacles
            )

        # Build reverse lookup for goals
        goal_positions = {pos: num for num, pos in goals.items()}

        # Collect moving obstacle trajectory cells for display
        moving_cells = set(moving_waypoints)

        # build ASCII grid
        rows = []
        rows.append("  " + " ".join(str(c) for c in range(n)))

        for i in range(n):
            row = f"{i} "
            for j in range(n):
                cell = (i, j)
                if cell == start:
                    row += "S "
                elif cell in static_obstacles:
                    row += "X "
                elif cell in goal_positions:
                    row += f"{goal_positions[cell]} "
                elif cell in moving_cells:
                    row += "M "
                else:
                    row += ". "
            rows.append(row.strip())

        grid_str = "\n".join(rows)
        return grid_str, goals, static_obstacles, visit_order, moving_waypoints

    raise RuntimeError(
        f"Could not generate solvable sequential-goal grid (n={n}) after {max_tries} tries."
    )


# ---------- 5. Choose static obstacle counts ----------

def choose_static_count(n, num_goals=3):
    """
    Pick a plausible number of static obstacles, accounting for S and goals.
    """
    free_cells = max(n*n - 1 - num_goals, 0)  # exclude start + goals

    if free_cells <= 0:
        return 0

    # size-dependent density range
    if n <= 3:
        d_min, d_max = 0.0, 0.10
    elif n <= 6:
        d_min, d_max = 0.05, 0.20
    else:
        d_min, d_max = 0.10, 0.30

    density = random.uniform(d_min, d_max)
    num_static = round(density * free_cells)
    num_static = max(0, min(num_static, free_cells))
    return num_static


# ---------- 6. Main entry point ----------

def main():
    parser = argparse.ArgumentParser(description="Generate grid-world datasets")
    parser.add_argument("--seed", type=int, default=124, help="Random seed")
    parser.add_argument("--total", type=int, default=100, help="Total samples to generate")
    parser.add_argument("--num-goals", type=int, default=3, help="Number of sequential goals")
    parser.add_argument("--min-size", type=int, default=2, help="Minimum grid size")
    parser.add_argument("--max-size", type=int, default=10, help="Maximum grid size")
    parser.add_argument("--moving-prob", type=float, default=0.5,
                        help="Probability of adding a moving obstacle (for n>=4)")
    parser.add_argument("--output", type=str, default="grid_100_samples.csv",
                        help="Output CSV filename")
    parser.add_argument("--max-tries", type=int, default=300,
                        help="Max attempts per grid generation")
    args = parser.parse_args()

    random.seed(args.seed)

    samples = []
    seen_grids = set()

    sizes = list(range(args.min_size, args.max_size + 1))

    # ----- 1. Base equal distribution -----
    base = args.total // len(sizes)
    remainder = args.total % len(sizes)

    target_counts = {n: base for n in sizes}
    extra_sizes = random.sample(sizes, remainder)
    for n in extra_sizes:
        target_counts[n] += 1

    # ----- 2. Cap n=2 and redistribute extras -----
    MAX_UNIQUE_FOR_2 = 4  # heuristic upper bound on distinct solvable 2x2 grids

    if 2 in target_counts and target_counts[2] > MAX_UNIQUE_FOR_2:
        diff = target_counts[2] - MAX_UNIQUE_FOR_2
        target_counts[2] = MAX_UNIQUE_FOR_2

        larger_ns = [n for n in sizes if n >= 3]
        idx = 0
        for _ in range(diff):
            target_counts[larger_ns[idx % len(larger_ns)]] += 1
            idx += 1

    print("Target samples per size:", target_counts)
    print("Total samples:", sum(target_counts.values()))

    # ----- 3. Sampling loop -----
    j = 0
    for n in sizes:
        count_target = target_counts[n]
        count_collected = 0

        while count_collected < count_target:
            num_static = choose_static_count(n, num_goals=args.num_goals)

            grid_str, goals, xs, visit_order, moving = generate_grid_with_sequential_goals(
                n=n,
                num_static=num_static,
                num_goals=args.num_goals,
                moving_obstacle_prob=args.moving_prob,
                max_tries=args.max_tries,
            )

            # == ENSURE UNIQUENESS (GLOBAL) ==
            grid_key = grid_str
            if grid_key in seen_grids:
                continue

            seen_grids.add(grid_key)
            count_collected += 1
            j += 1

            print(f"\nSample {j} (n={n}):")
            print(grid_str)
            print("Goals:", goals)
            if moving:
                print("Moving:", moving)
            print("Visit order:", visit_order)
            print("BFS Steps:", len(visit_order) - 1)

            samples.append({
                "n": n,
                "grid": grid_str,
                "goals": goals,
                "static": xs,
                "moving": moving,
                "visit_order": visit_order,
                "min_steps": len(visit_order) - 1,
            })

    print(f"\nTotal unique samples: {len(samples)}")

    # Save to CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "grid", "goals", "static", "moving", "BFS_steps"])

        for s in samples:
            writer.writerow([
                s["n"],
                s["grid"].replace("\n", "\\n"),
                s["goals"],
                s["static"],
                s["moving"],
                s["min_steps"],
            ])

    print(f"Saved {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
