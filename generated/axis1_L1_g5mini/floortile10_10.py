from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact string into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (w133ormt) painting domain.

    The domain models one or more robots that can:
      - move on a directed grid (four different relation types encode directed edges),
      - change their "paint color" (two colors are available in the provided instances),
      - paint tiles from an adjacent tile (two painting action schemas exist, each
        requiring the robot to be on a specific predecessor tile relative to the
        tile to be painted).
    Goals are of the form (yh6q5j8r tile color) — i.e. "tile must have color".

    This heuristic estimates the number of actions remaining by greedily assigning
    remaining painting tasks (goal tiles not yet painted with the desired color)
    to robots. For each robot-task pair the estimated cost is:
      movement_cost + color_change_cost + paint_action_cost
    where:
      - movement_cost is the shortest-path distance (in movement actions) from the
        robot's current tile to one of the valid painting positions for the target tile,
      - color_change_cost is 0 if the robot already has the desired color, otherwise 1
        (one application of z7pcmkeu can change the robot color to any available color),
      - paint_action_cost is 1 (the painting operator c1qh1p5g / cadwr4wd).

    The assignment is built greedily: repeatedly pick the robot-task pair with the
    smallest estimated cost, add that cost to the total, update the robot's position
    to the painting position used and its color to the goal color, and remove the
    completed task. This produces a fast, informative estimate that accounts for:
      - robot mobility (via shortest-path distances on the movement graph),
      - color change operations,
      - the fact that multiple robots can share the painting load.

    Assumptions
    -----------
    - Movement relations in the static facts encode directed edges. A static fact
      "(rel target source)" allows an action that moves the robot from `source` to `target`.
      We therefore build movement edges source -> target for all movement relation facts.
    - Painting is always performed from a predecessor tile as encoded by two
      painting relations (xr2t1c1d and oh0smogc in the domain). For a target tile T,
      any tile P for which a static fact "(xr2t1c1d T P)" or "(oh0smogc T P)" holds
      is a valid painting position for T.
    - We ignore dynamic obstacles (e.g., tiles that become non-traversable when painted).
      Distances are computed on the static movement graph (i.e., optimistic connectivity).
    - Color changes have uniform cost 1 and are possible to any known color (as per z7pcmkeu).
    - If a goal tile has no valid painting positions (no predecessor in the two painting relations),
      the tile is treated as unreachable and contributes a large finite penalty to the heuristic.

    Heuristic Initialization
    ------------------------
    The constructor extracts and stores:
      - The set of goal (tile, color) pairs from task.goals.
      - The movement adjacency graph (source -> set(targets)) from the static facts
        for the four relation types used by movement operators:
        {"xr2t1c1d", "oh0smogc", "w224no0l", "g7ya35is"}.
      - For each tile, the set of valid painting positions (predecessors) according
        to the painting relations {"xr2t1c1d", "oh0smogc"}.
      - The set of known colors (facts of the form (cdb9op3o color)) which is used
        for consistency of the color-change assumption.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state):

    1. Parse the state's facts to find:
       - Current robot positions: facts of the form (gk9u3vg3 robot tile).
         There may be multiple robots. For each robot we track its current tile.
       - Current robot colors: facts of the form (i4u94rl9 robot color).
       - Currently painted tiles: facts of the form (yh6q5j8r tile color).

    2. Identify remaining goals:
       - For each goal (yh6q5j8r tile color) that is not present in the state,
         add (tile, color) to the list of remaining tasks.

    3. Precompute (or cache) shortest-path distances on the movement graph:
       - For each robot's current tile we run BFS to obtain distances to all tiles.
       - Distances are counted in movement actions (each directed graph edge has cost 1).

    4. Greedy task assignment:
       - While tasks remain:
         a) For every robot and every remaining task (tile,color), compute the minimal
            estimated cost to perform that task:
               min_{p in painting_positions[tile]} ( dist(robot_tile, p) )
               + (0 if robot_color == color else 1)
               + 1     # painting action
            If a tile has no painting positions or is unreachable, a large finite penalty
            is used for that assignment to keep the heuristic finite.
         b) Select the robot-task pair with the smallest estimated cost.
         c) Add that cost to the cumulative heuristic estimate.
         d) Update the assigned robot: its tile becomes the painting position chosen,
            its color becomes the goal color.
         e) Remove the task from the remaining tasks.

    5. Return the cumulative cost as the heuristic value. If no tasks remain, return 0.

    Notes on admissibility and performance:
    - This heuristic is not guaranteed admissible (it may underestimate or overestimate),
      but is designed to be informative and cheaply computable to reduce node expansions
      in greedy best-first search.
    """

    def __init__(self, task):
        # Store goals (as strings) for potential use; also extract tile->color goals.
        self.goals = task.goals

        # Movement relations used by movement operators (all create directed edges).
        movement_predicates = {"xr2t1c1d", "oh0smogc", "w224no0l", "g7ya35is"}

        # Painting relations (from which we infer valid painting positions for each tile).
        painting_predicates = {"xr2t1c1d", "oh0smogc"}

        # Known colors (facts like "(cdb9op3o white)").
        self.colors = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "cdb9op3o" and len(parts) >= 2:
                self.colors.add(parts[1])

        # Build movement adjacency: source -> set(targets)
        # For a static fact "(rel target source)" we add edge source -> target.
        self.adjacency = {}
        # Also build painting positions mapping: target -> set(sources) for painting relations.
        self.paint_positions = {}

        # Collect a set of all tiles (nodes) that appear in static facts to have consistent BFS scope.
        self.all_tiles = set()

        for fact in task.static:
            parts = get_parts(fact)
            if len(parts) >= 3:
                pred, a, b = parts[0], parts[1], parts[2]
                # Track tiles generically (these args are typed bffr59sc in domain)
                self.all_tiles.add(a)
                self.all_tiles.add(b)
                if pred in movement_predicates:
                    # edge from b -> a
                    self.adjacency.setdefault(b, set()).add(a)
                    # ensure nodes exist in adjacency map
                    self.adjacency.setdefault(a, set())
                if pred in painting_predicates:
                    # painting position for tile a is b (robot must be at 'b' to paint 'a')
                    self.paint_positions.setdefault(a, set()).add(b)

        # Ensure adjacency lists for any tile referenced only in paint_positions or goals
        for t in list(self.paint_positions.keys()):
            self.adjacency.setdefault(t, set())
            for p in self.paint_positions[t]:
                self.adjacency.setdefault(p, set())

        # Precompute a list of goal (tile, color) pairs extracted from task.goals
        self.goal_tile_color = []
        for fact in self.goals:
            parts = get_parts(fact)
            if parts and parts[0] == "yh6q5j8r" and len(parts) >= 3:
                tile, color = parts[1], parts[2]
                self.goal_tile_color.append((tile, color))
                self.all_tiles.add(tile)

        # Large finite penalty for unreachable tasks (keeps heuristic finite).
        self.large_cost = 100000

    def _bfs_distances(self, start):
        """
        Compute BFS shortest-path distances in the directed movement graph from start.
        If start is None or unknown, return empty dict.
        """
        if start is None or start not in self.adjacency:
            return {}
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
        """
        Compute the heuristic estimate for the given search node.

        Steps:
        - Find robots, their positions and colors.
        - Determine which goal tile-color pairs are still unsatisfied.
        - Greedily assign tasks to robots using BFS distances on the static movement graph.
        - Return the cumulative estimated cost.
        """
        state = node.state

        # Extract current painted tiles (yh6q5j8r tile color)
        painted = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "yh6q5j8r" and len(parts) >= 3:
                painted.add((parts[1], parts[2]))

        # Remaining goal tasks: those goals not yet achieved in the state.
        remaining_tasks = [
            (tile, color) for (tile, color) in self.goal_tile_color if (tile, color) not in painted
        ]

        # If no remaining tasks, heuristic is 0.
        if not remaining_tasks:
            return 0

        # Extract robot positions and colors.
        # robots: dict robot_name -> {'pos': tile_name or None, 'color': color or None}
        robots = {}
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "gk9u3vg3" and len(parts) >= 3:
                    robot, tile = parts[1], parts[2]
                    robots.setdefault(robot, {})["pos"] = tile
                    self.all_tiles.add(tile)
                elif parts[0] == "i4u94rl9" and len(parts) >= 3:
                    robot, color = parts[1], parts[2]
                    robots.setdefault(robot, {})["color"] = color

        # If no robot facts found, we create a dummy robot to keep heuristic finite.
        if not robots:
            robots["__dummy_robot__"] = {"pos": None, "color": None}

        # Initialize robot list for assignment: list of dicts with 'pos' and 'color'
        robot_list = []
        for rname, info in robots.items():
            pos = info.get("pos", None)
            color = info.get("color", None)
            robot_list.append({"name": rname, "pos": pos, "color": color})

        total_cost = 0

        # Cache BFS distances per robot position to avoid recomputing.
        bfs_cache = {}

        # Greedy assignment loop.
        tasks = remaining_tasks.copy()
        while tasks:
            best = None  # (cost, robot_index, task_index, chosen_paint_pos)
            # For each robot and each task compute minimal cost
            for ri, robot in enumerate(robot_list):
                pos = robot["pos"]
                color = robot["color"]
                if pos not in bfs_cache:
                    bfs_cache[pos] = self._bfs_distances(pos)
                distances = bfs_cache[pos]
                for ti, (tile, goal_color) in enumerate(tasks):
                    # Candidate painting positions for this tile
                    painting_positions = self.paint_positions.get(tile, set())
                    if not painting_positions:
                        # No painting positions known; make this assignment very expensive
                        est_cost = self.large_cost
                        chosen_p = None
                    else:
                        # Find minimal distance to any painting position
                        min_dist = None
                        chosen_p = None
                        for p in painting_positions:
                            d = distances.get(p, None)
                            if d is None:
                                # If unreachable from this robot pos (in BFS), treat as large
                                continue
                            if min_dist is None or d < min_dist:
                                min_dist = d
                                chosen_p = p
                        if min_dist is None:
                            # All painting positions unreachable from this robot pos (in BFS)
                            est_cost = self.large_cost
                        else:
                            move_cost = min_dist
                            color_cost = 0 if color == goal_color else 1
                            paint_cost = 1
                            est_cost = move_cost + color_cost + paint_cost
                    # Update best
                    if best is None or est_cost < best[0]:
                        best = (est_cost, ri, ti, chosen_p, tile, goal_color)

            # If best cost is extremely large, no valid assignments exist via static connectivity.
            if best is None or best[0] >= self.large_cost:
                # Return a large finite value to indicate difficulty/unreachability.
                return self.large_cost

            est_cost, ri, ti, chosen_p, tile_assigned, color_assigned = best
            total_cost += est_cost

            # Update robot state: its new position is the chosen painting position (robot does not move onto painted tile),
            # and its color becomes the painting color (since it either changed color or already had it).
            robot_list[ri]["pos"] = chosen_p
            robot_list[ri]["color"] = color_assigned

            # Remove the assigned task
            tasks.pop(ti)

            # If total_cost grows beyond large_cost, bail out early and return large_cost to keep heuristic finite.
            if total_cost >= self.large_cost:
                return self.large_cost

        return int(total_cost)