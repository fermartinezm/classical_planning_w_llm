from collections import deque
from math import inf
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the tokens of a PDDL fact string, without surrounding parentheses."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """Return True if fact matches the given pattern (supports '*' wildcards)."""
    parts = get_parts(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the 'floortile' domain.
    The heuristic estimates the number of actions required to satisfy all
    (painted tile color) goals. It models the main costs:
      - moving a robot from its current tile to a tile adjacent to the target tile
        (one move per adjacency step),
      - optionally changing the robot's held color (one change_color action),
      - painting the tile (one paint action).

    Assumptions
    -----------
    - Painting a tile requires the robot to be positioned on a neighboring tile
      (up/down/left/right relation in the static facts).
    - A single paint action paints the tile with the robot's currently held color.
      Changing the robot's color takes one change_color action (assume an
      available-color exists for the needed color since the domain provides
      available-color facts).
    - We ignore temporary blocking effects of clear/unclear tiles when computing
      movement distances (this keeps the heuristic efficiently computable and
      optimistic). Once a tile has been assigned an intended paint action we do
      mark it as painted for future assignments (so the same tile is not counted twice).
    - Multiple robots can be used. We assign tasks greedily to robots and update
      each robot's assumed position and color after an assignment. This gives a
      more informed (and typically stronger) heuristic than treating every goal
      independently while still being efficient.

    Heuristic Initialization
    ------------------------
    In the constructor we extract static information:
      - adjacency graph between tiles derived from the up/down/left/right static facts;
        edges are treated undirected for movement distance computation.
      - for every tile, the set of tiles from which that tile can be painted (its
        "painting neighbor" tiles). A painting neighbor X for tile Y is a tile X
        that appears in a static fact like (up Y X) or (left Y X), etc.
      - precompute BFS distances between all tiles in the adjacency graph.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given planner node/state:
    1. If all goal painted facts are already satisfied in the state, return 0.
    2. Parse the state to obtain:
       - current robot positions (robot-at facts),
       - current robot colors (robot-has facts).
       - set of already painted facts (we treat goals already satisfied as done).
    3. Build a list of remaining painting goals: (tile, desired_color) where the
       corresponding predicate "(painted tile color)" is not in the state.
    4. Greedy assignment:
       - Maintain an up-to-date simulated position and color for each robot.
       - For each remaining goal, pick the robot and neighboring painting position
         that yields the minimal estimated cost:
           cost = distance(robot_pos -> neighbor_pos)
                + (1 if robot_color != desired_color else 0)  # change_color
                + 1  # paint action
         Choose the minimal cost neighbor and robot pair.
       - After assigning a goal to a robot, update that robot's simulated position
         to be the neighbor tile (the place from which it painted) and update its
         color (to desired_color if a change was required). Also mark the goal tile
         as painted so we don't assign it again.
       - Repeat until all goals are assigned.
    5. Sum all assignment costs and return that sum as the heuristic value.

    Notes
    -----
    - The heuristic is intentionally optimistic with respect to movement (ignores
      dynamic blocking by non-clear tiles), which keeps it admissible in many
      cases but more importantly keeps computation fast. Because the heuristic
      is used with greedy best-first search (and admissibility is not required),
      the greedy assignment strategy improves guidance by accounting for robot
      repositioning and color changes between successive paints.
    - The heuristic always returns a finite integer (if a tile or neighbor is
      unreachable in the static adjacency graph, a large finite penalty is used).
    """

    def __init__(self, task):
        """
        Extract static information from the task:
         - build adjacency graph between tiles,
         - compute painting-neighbor sets,
         - compute all-pairs shortest-path distances (BFS per tile).
        """
        self.goals = task.goals  # set of goal fact strings like "(painted tile_1_1 white)"
        static_facts = task.static

        # Collect all tiles mentioned in static facts (safe superset of domain tiles).
        tiles = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ("up", "down", "left", "right"):
                tiles.add(parts[1])
                tiles.add(parts[2])

        # Build undirected adjacency graph between tiles using the directional static facts.
        adjacency = {t: set() for t in tiles}
        # For painting neighbors: for a tile Y, which X satisfy (up Y X) or (down Y X) etc.
        painting_neighbors = {t: set() for t in tiles}

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in ("up", "down", "left", "right"):
                a, b = parts[1], parts[2]
                # Add undirected edge between a and b for movement purposes.
                adjacency.setdefault(a, set()).add(b)
                adjacency.setdefault(b, set()).add(a)
                # For painting, facts of form (up Y X) mean X is a valid robot position
                # to paint Y. So record that relationship:
                # parts[1] is Y, parts[2] is X for predicates used in painting preconditions.
                painting_neighbors.setdefault(parts[1], set()).add(parts[2])

        self.adjacency = adjacency
        self.painting_neighbors = painting_neighbors

        # Precompute BFS distances between all tiles (ignoring dynamic clears).
        # If a tile is isolated (not in adjacency), it will get no entries and
        # distances will later be treated as unreachable.
        self.distances = {}
        for t in adjacency:
            self.distances[t] = self._bfs_distances(t)

        # Small finite large penalty for unreachable situations (keeps heuristic finite).
        self.UNREACHABLE_PENALTY = 1000

    def _bfs_distances(self, start):
        """Return a dict mapping tile -> distance (int) from start using adjacency."""
        dist = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in self.adjacency.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def __call__(self, node):
        """Compute the heuristic estimate (integer) for the given search node."""
        state = node.state

        # Quick check: if all goals already satisfied in state, return 0.
        if self.goals <= state:
            return 0

        # Parse robot locations and colors from the state.
        robot_positions = {}  # robot -> tile
        robot_colors = {}  # robot -> color string

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_positions[robot] = tile
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        # Ensure there is at least one robot (defensive).
        robots = list(robot_positions.keys())
        if not robots:
            # No robot available -> cannot paint; return large penalty.
            return len([g for g in self.goals if g not in state]) * self.UNREACHABLE_PENALTY

        # Construct list of remaining painting goals as (tile, color).
        remaining_goals = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                if goal not in state:
                    remaining_goals.append((tile, color))

        # If no remaining goals, we're done.
        if not remaining_goals:
            return 0

        # We'll greedily assign goals to robots. Make local mutable copies of robot info.
        simulated_pos = dict(robot_positions)  # robot -> tile
        simulated_color = dict(robot_colors)  # robot -> color

        total_cost = 0

        # For each goal, greedily pick the robot+neighbor that yields minimal incremental cost.
        # We iterate until all goals are assigned.
        goals_to_assign = remaining_goals.copy()

        while goals_to_assign:
            best_choice = None  # (cost, robot, goal_index, chosen_neighbor)
            # Evaluate all combinations (robot x goal) and pick minimal.
            for gi, (goal_tile, goal_color) in enumerate(goals_to_assign):
                neighbors = self.painting_neighbors.get(goal_tile, set())
                if not neighbors:
                    # No valid painting neighbor -> highly penalize
                    best_local_cost = self.UNREACHABLE_PENALTY
                    # Assign with dummy neighbor None; will still allow algorithm to finish.
                    for robot in robots:
                        cost = best_local_cost
                        if best_choice is None or cost < best_choice[0]:
                            best_choice = (cost, robot, gi, None, goal_tile, goal_color)
                    continue

                for robot in robots:
                    rpos = simulated_pos.get(robot)
                    rcol = simulated_color.get(robot)
                    if rpos is None:
                        # Robot position unknown; penalize
                        rpos = None

                    # For each neighbor position from which goal_tile can be painted, compute cost.
                    best_neighbor_cost = inf
                    best_neighbor = None
                    for nbr in neighbors:
                        # distance from rpos to nbr
                        if rpos is None or rpos not in self.distances:
                            d = inf
                        else:
                            d = self.distances[rpos].get(nbr, inf)
                        change_cost = 0 if (rcol == goal_color) else 1
                        paint_cost = 1
                        # Total cost to move there and paint.
                        totalc = d + change_cost + paint_cost
                        if totalc < best_neighbor_cost:
                            best_neighbor_cost = totalc
                            best_neighbor = nbr

                    # If no neighbor reachable, use unreachable penalty
                    if best_neighbor_cost == inf:
                        cost = self.UNREACHABLE_PENALTY
                        choice = (cost, robot, gi, None, goal_tile, goal_color)
                    else:
                        cost = int(best_neighbor_cost)
                        choice = (cost, robot, gi, best_neighbor, goal_tile, goal_color)

                    if best_choice is None or choice[0] < best_choice[0]:
                        best_choice = choice

            # Apply best_choice
            if best_choice is None:
                # No viable assignment found (shouldn't happen), break to avoid infinite loop.
                break

            cost, robot, gi, chosen_neighbor, goal_tile, goal_color = best_choice
            total_cost += cost

            # Update simulated robot position and color after performing the assigned action.
            if chosen_neighbor is not None:
                simulated_pos[robot] = chosen_neighbor
                # If change was needed, update color to goal_color (even if change_cost was penalized).
                if simulated_color.get(robot) != goal_color:
                    simulated_color[robot] = goal_color
            else:
                # chosen_neighbor None implies unreachable; still pop the goal to terminate,
                # and do not update robot state.
                pass

            # Remove the assigned goal from the list.
            # Use gi index to pop; careful as gi refers to current goals_to_assign ordering.
            # We need to remove the goal matching (goal_tile, goal_color).
            # Safer: remove the first matching element.
            removed = False
            for idx, (t, c) in enumerate(goals_to_assign):
                if t == goal_tile and c == goal_color:
                    goals_to_assign.pop(idx)
                    removed = True
                    break
            if not removed:
                # Fallback: remove by gi if in range
                if 0 <= gi < len(goals_to_assign):
                    goals_to_assign.pop(gi)
                else:
                    # As a last resort, clear remaining goals to avoid infinite loop.
                    goals_to_assign.clear()

        # Ensure returned heuristic is an integer and finite.
        if total_cost >= inf:
            return int(self.UNREACHABLE_PENALTY * len(remaining_goals))
        return int(total_cost)