from collections import deque
from math import inf
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the space-separated parts of a PDDL fact string without surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'floortile' domain. The heuristic estimates the
    number of actions required to satisfy all "painted tile color" goal facts that are
    not yet achieved in the current state. It reasons about robot travel cost (moves),
    possible color changes, and the paint action itself. When multiple robots exist the
    heuristic uses a greedy assignment strategy to allocate tiles to robots and updates
    the robot's (estimated) position and color as tiles are assigned. The heuristic is
    not guaranteed to be admissible but aims to be informative and cheaply computable.

    Assumptions
    -----------
    - The cost of a single atomic action (move, change_color, paint) is 1.
    - To paint a tile T with color C:
        * A robot must be on a tile X such that a static relation (up T X) or
          (down T X) or (left T X) or (right T X) exists (these are the legal
          "painting vantage points").
        * The tile T must be clear at the moment of painting.
        * The robot must have color C (if not, it can change color with one action).
        * Painting itself costs 1 action.
    - Movement cost is computed on the static directed adjacency graph encoded
      by the static move relations. Movement preconditions that depend on dynamic
      'clear' facts are ignored for pathfinding (this yields an efficiently
      computable estimate).
    - If a tile is currently not clear because some robot occupies it, the heuristic
      charges an extra cost of 1 to clear it (i.e., move the occupant away).
    - The heuristic treats painting goals independently but uses a greedy assignment
      to avoid extreme underestimation when one robot must paint multiple tiles in sequence.

    Heuristic Initialization
    ------------------------
    In the constructor we:
    - Store the set of goal facts (task.goals).
    - Parse static facts (task.static) to build:
        * adjacency: a directed graph mapping a tile -> list of tiles reachable by a single move.
                    (Edge x -> y exists if a static fact like "(up y x)" exists because
                     the move operator moves from x to y.)
        * paint_vantage_points: for each tile T, the list of tiles X from which a robot
                                can execute a paint_* action to paint T (collected from
                                static up/down/left/right facts).
        * available colors (set) from static facts, for completeness (not strictly required).
    The static information is extracted once to enable efficient repeated heuristic calls.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state), the heuristic follows these steps:

    1. Goal check:
       - If all goal facts are already satisfied in the state, return 0.

    2. Extract dynamic information from the state:
       - Which tiles are currently painted (map tile -> set(colors)).
       - Which tiles are clear.
       - Where each robot is located and which color it currently carries.

    3. Build the list of painting tasks:
       - For each goal fact "(painted TILE COLOR)", if TILE is not already painted with COLOR
         in the current state, it becomes a painting task to achieve.

    4. Compute costs for accomplishing painting tasks:
       - For a robot r at position P with current color Cr and a task (T, Cgoal):
         * Determine all vantage points X for painting T (from static paint_vantage_points).
         * Compute shortest move distance dist(P, X) on the static directed graph (BFS).
         * The cost to paint T from P via X is:
               dist(P, X) + (1 if Cr != Cgoal else 0) + 1
           where +1 is the painting action itself.
         * If T is currently not clear and some robot occupies T (robot-at T), add +1
           to the cost to account for moving that occupant away.
       - If no vantage points exist for T or no robot can reach any vantage point, treat the
         task as (very) expensive by assigning a large finite penalty (to keep values finite).

    5. Greedy assignment of tasks to robots:
       - Repeatedly select the (robot, task, vantage point) triple with minimal incremental cost,
         add that cost to the total heuristic, update the robot's estimated position (to the
         vantage point used) and its color (to Cgoal), and remove the task.
       - If clearing an occupant was charged, we update the occupant robot's estimated position
         by moving it to one of its static neighbors if possible (a cheap approximation).
       - Repeat until all tasks are assigned. The sum of chosen costs is returned.

    Rationale and complexity
    ------------------------
    - The heuristic captures the main structure of the problem: travel + possible color change + paint.
    - Using BFS on the static adjacency graph produces fast shortest-path estimates that ignore
      temporary dynamic blockages (clear). This keeps computation efficient and usually informative.
    - The greedy assignment is an inexpensive way to approximate the cost of multiple tasks for
      the same robot and reduces the degree to which we severely underestimate multi-step sequences.
    """

    def __init__(self, task):
        # Goals and static facts
        self.goals = task.goals
        static_facts = task.static

        # Directed adjacency graph for movement: from_tile -> [to_tiles]
        # Movement edges are created when a static relation like "(up Y X)" exists -> edge X -> Y
        self.adjacency = {}
        # Paint vantage points: tile -> list of tiles from where it can be painted
        self.paint_vantage_points = {}
        # Available colors
        self.available_colors = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred in ("up", "down", "left", "right"):
                # parts: [pred, tile_target, tile_source] -> can move from tile_source -> tile_target
                if len(parts) >= 3:
                    target = parts[1]
                    source = parts[2]
                    # adjacency: source -> target (move from source to target)
                    self.adjacency.setdefault(source, []).append(target)
                    # painting vantage point: to paint 'target' one can be at 'source' (because
                    # paint_up uses (up target source) and robot-at source)
                    self.paint_vantage_points.setdefault(target, []).append(source)
            elif pred == "available-color" and len(parts) >= 2:
                self.available_colors.add(parts[1])

        # Ensure every known tile appears in adjacency and paint_vantage_points structures
        # (avoid KeyErrors later)
        # adjacency keys might be missing for tiles that have no outgoing moves; add empty lists lazily
        # paint_vantage_points: if a tile is never paintable from any neighbor (isolated), it will not exist.

        # Precompute the set of tiles (keys of adjacency and paint vantage points)
        self.tiles = set(self.adjacency.keys()) | set(self.paint_vantage_points.keys())
        for t in list(self.tiles):
            self.adjacency.setdefault(t, [])
            self.paint_vantage_points.setdefault(t, [])

    def _bfs_distances(self, start):
        """
        Compute shortest distances (number of moves) from start to all reachable tiles
        using the static directed adjacency graph. Returns a dict tile -> distance.
        """
        d = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            for neighbor in self.adjacency.get(cur, []):
                if neighbor not in d:
                    d[neighbor] = d[cur] + 1
                    q.append(neighbor)
        return d

    def __call__(self, node):
        state = node.state

        # If already in a goal state, heuristic is 0
        if self.goals <= state:
            return 0

        # Parse painted tiles: tile -> set(colors)
        painted = {}
        clear_tiles = set()
        robot_positions = {}  # robot -> tile
        robot_colors = {}  # robot -> color

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "painted" and len(parts) >= 3:
                tile = parts[1]
                color = parts[2]
                painted.setdefault(tile, set()).add(color)
            elif pred == "clear" and len(parts) >= 2:
                clear_tiles.add(parts[1])
            elif pred == "robot-at" and len(parts) >= 3:
                robot = parts[1]
                tile = parts[2]
                robot_positions[robot] = tile
            elif pred == "robot-has" and len(parts) >= 3:
                robot = parts[1]
                color = parts[2]
                robot_colors[robot] = color

        # Collect painting tasks (tile, color) that are required by goals but not yet satisfied
        tasks = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) >= 3:
                tile = parts[1]
                color = parts[2]
                if tile not in painted or color not in painted[tile]:
                    tasks.append((tile, color))

        # If no tasks remain, goal satisfied (should have returned earlier), but safe guard
        if not tasks:
            return 0

        # If there are no robots, return a large cost (unsolvable)
        if not robot_positions:
            return 10000

        # Prepare mutable copies for greedy assignment: robot -> current estimated position & color
        robots = list(robot_positions.keys())
        robot_pos = {r: robot_positions.get(r) for r in robots}
        robot_col = {r: robot_colors.get(r) for r in robots}

        total_cost = 0
        remaining_tasks = list(tasks)

        # A reasonably large finite penalty for unreachable tasks (keeps heuristic finite)
        LARGE_PENALTY = 10000

        # Greedy assignment loop
        while remaining_tasks:
            best = None  # (cost, robot, task_index, chosen_vantage, clear_penalty, occupant_robot)
            # For each robot compute BFS distances once
            bfs_cache = {}
            for r in robots:
                pos = robot_pos.get(r)
                if pos is None:
                    continue
                bfs_cache[r] = self._bfs_distances(pos)

            for t_index, (tile, needed_color) in enumerate(remaining_tasks):
                vantage_points = self.paint_vantage_points.get(tile, [])
                if not vantage_points:
                    # No vantage points -> task effectively unreachable
                    # Try to mark as extremely expensive
                    for r in robots:
                        if r not in bfs_cache:
                            continue
                        cost = LARGE_PENALTY
                        candidate = (cost, r, t_index, None, 0, None)
                        if best is None or candidate[0] < best[0]:
                            best = candidate
                    continue

                # detect whether tile is currently clear; if not clear, check for occupant robot
                clear_penalty = 0
                occupant_robot = None
                if tile not in clear_tiles:
                    # If the tile is not clear and not yet painted, it is likely occupied by a robot
                    for r in robots:
                        if robot_pos.get(r) == tile:
                            clear_penalty = 1
                            occupant_robot = r
                            break
                    # If tile not clear and no occupant, it is probably already painted in some colour
                    # (in which case it might be impossible to paint). Charge LARGE_PENALTY then.
                    if clear_penalty == 0 and tile in painted:
                        # already painted with other color(s) but not with needed_color: unreachable
                        for r in robots:
                            if r not in bfs_cache:
                                continue
                            cost = LARGE_PENALTY
                            candidate = (cost, r, t_index, None, 0, None)
                            if best is None or candidate[0] < best[0]:
                                best = candidate
                        continue

                # For each robot, compute minimal cost to reach any vantage point and paint
                for r in robots:
                    if r not in bfs_cache:
                        continue
                    dists = bfs_cache[r]
                    min_dist = inf
                    chosen_vp = None
                    for vp in vantage_points:
                        # distance from robot pos to vantage point (0 allowed if already there)
                        dist = dists.get(vp, inf)
                        if dist < min_dist:
                            min_dist = dist
                            chosen_vp = vp
                    if min_dist is inf:
                        # robot cannot reach any vantage point (in static graph)
                        continue
                    # color change cost
                    col_cost = 0 if robot_col.get(r) == needed_color else 1
                    # total cost: move distance + color change + paint action + clear_penalty
                    cost = min_dist + col_cost + 1 + clear_penalty
                    candidate = (cost, r, t_index, chosen_vp, clear_penalty, occupant_robot)
                    if best is None or candidate[0] < best[0]:
                        best = candidate

            # If best is still None, no robot can reach any remaining task -> extremely expensive
            if best is None:
                total_cost += LARGE_PENALTY * len(remaining_tasks)
                break

            # Unpack best assignment and apply it greedily
            cost, robot_chosen, task_idx, vp, clear_penalty, occupant = best
            # Add the chosen cost
            total_cost += cost

            # Update the robot's estimated state: position becomes the vantage point used; color becomes needed color
            tile_chosen, color_chosen = remaining_tasks[task_idx]
            if vp is not None:
                robot_pos[robot_chosen] = vp
            robot_col[robot_chosen] = color_chosen

            # If we charged a clear penalty for an occupant robot, move that occupant somewhere (cheap approx.)
            if clear_penalty and occupant is not None:
                occ_pos = robot_pos.get(occupant)
                # Try to move occupant to any outgoing neighbor (static adjacency) if possible
                moved = False
                for neigh in self.adjacency.get(occ_pos, []):
                    # move occupant to the first neighbor found
                    robot_pos[occupant] = neigh
                    moved = True
                    break
                if not moved:
                    # if occupant has no outgoing neighbors, try incoming neighbors by checking adjacency keys
                    for from_tile, outs in self.adjacency.items():
                        if occ_pos in outs:
                            robot_pos[occupant] = from_tile
                            moved = True
                            break
                    # if still not moved, leave position unchanged (best effort)

                # update clear tiles set conservatively (we moved occupant away)
                clear_tiles.add(tile_chosen)  # painting hasn't yet happened in our optimistic sim
                # The tile being painted will become not clear after the paint action, but since we
                # remove the task below, tracking the clear set further is not necessary for correctness.

            # Remove the assigned task from remaining_tasks
            # Note: tasks list may shift; we pop by index
            remaining_tasks.pop(task_idx)

        # Return the summed estimated cost (ensure it's an int)
        return int(total_cost)