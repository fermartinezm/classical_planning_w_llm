from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the tokens of a PDDL fact string by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (w133ormt) that estimates the
    number of actions required to achieve all target "painted tile" goals (yh6q5j8r).
    The domain models robots that move on a grid of tiles, hold a color, and can paint
    neighboring tiles with their held color. Moving, changing color, and painting are
    atomic actions in the domain; movement is constrained by a tile being free
    (d7ekk7vb) and by adjacency relations in the static facts.

    Assumptions
    -----------
    - Goals are of the form (yh6q5j8r <tile> <color>) and each such goal requires exactly
      one paint action that produces that fact.
    - Any robot can change its held color to any color in a single color-change action.
    - Painting a tile requires:
        * the tile to be free (d7ekk7vb) at the time of painting,
        * some robot to be located on a specific neighboring tile (as defined by the
          static predicates xr2t1c1d and oh0smogc), and
        * the robot to hold the color required by the goal.
    - Robots can move along edges defined by the static movement relations
      (xr2t1c1d, oh0smogc, g7ya35is, w224no0l). Movement cost is 1 per step.
    - For heuristic estimation we allow traversal through tiles that are currently free
      or currently occupied by robots (optimistic assumption to keep the estimate
      informative and cheap to compute).
    - The heuristic is not required to be admissible (it may underestimate or overestimate),
      but it should be finite for solvable states.

    Heuristic Initialization
    ------------------------
    In the constructor we extract and store:
    - self.goals: the set of goal fact strings (task.goals).
    - movement_graph: a mapping tile -> list of neighbor tiles reachable in one move.
      This is built from the static facts for the four movement predicates:
      xr2t1c1d, oh0smogc, g7ya35is, w224no0l. For a static fact
      "(pred A B)" the movement edge is B -> A because movement operators in
      the domain use that ordering (robot at B can move to A if (pred A B) holds).
    - paint_adjacency: a mapping from target_tile -> set of robot_tiles from which
      that target can be painted; this is derived from the static facts of predicates
      xr2t1c1d and oh0smogc (the two predicates used by paint actions).
    - available_colors: set of colors present as cdb9op3o in static.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a planner node/state, the heuristic computes an estimate as follows:

    1. Quick goal check:
       - If all goal facts are already true in the state, return 0.

    2. State extraction:
       - Parse the state facts to obtain:
         * robot_positions: mapping robot -> its current tile (from gk9u3vg3 facts),
         * robot_colors: mapping robot -> held color (from i4u94rl9 facts),
         * free_tiles: set of tiles that currently have d7ekk7vb,
         * painted: mapping tile -> color for tiles already painted (yh6q5j8r facts).
       - Parse goal facts into (tile -> required_color) pairs.

    3. For every goal (tile T, color C) not yet satisfied in the state:
       - Compute a per-goal minimal estimated cost across all robots as:
            cost = moves_needed + color_change_cost + paint_action_cost + free_penalty
         where:
           * moves_needed: shortest number of movement steps for a robot to reach
             one of the robot-positions from which painting T is permitted (using
             the static paint_adjacency mapping). Movement distances are computed
             with BFS on the movement_graph. The BFS considers a node (tile)
             traversable if it is currently free (in free_tiles) or is a tile that
             currently contains a robot (optimistic assumption).
           * color_change_cost: 0 if that robot already holds color C, else 1
             (one z7pcmkeu action to switch to color C).
           * paint_action_cost: 1 (for the painting action).
           * free_penalty: 0 if target tile T is currently free, else 1. Painting
             requires T to be free; if it's not free we add a small optimistic
             penalty assuming it can be freed in roughly one action.
       - If a goal tile has no known painting-adjacent tiles (no static facts),
         or if every robot is unable to reach any painting position under the
         optimistic traversability assumptions, we assign a finite large cost for
         that goal (so the heuristic remains finite).

    4. The final heuristic value is the sum of the minimal per-goal costs. This
       is a simple, efficiently computed approximation: it treats each goal
       independently (so it may overcount shared movement or color-change savings),
       but it provides informative guidance by considering movement distance and
       required color changes.

    Notes on Design Choices
    -----------------------
    - We choose an optimistic traversability model (treat other robots' tiles as
      passable) to keep computations cheap and to avoid marking many states as
      unreachable. This keeps the heuristic finite and reasonably informative.
    - The heuristic counts one paint action per unsatisfied goal (exact).
    - Color switches are counted minimally (0 or 1) because the domain allows a
      robot to change color in a single action.
    - The heuristic is cheap: BFS runs on the static movement graph from each robot
      only for the relevant target adjacency set; complexity is linear in the
      number of tiles reachable under the optimistic model.
    """

    def __init__(self, task):
        # Goals (set of fact strings)
        self.goals = set(task.goals)

        # Static facts (frozenset of strings)
        static_facts = set(task.static)

        # Build movement graph and paint adjacency from static facts.
        # Movement predicates (edges from second arg -> first arg)
        movement_preds = {"xr2t1c1d", "oh0smogc", "g7ya35is", "w224no0l"}
        # Paint predicates (target tile, robot tile): xr2t1c1d and oh0smogc
        paint_preds = {"xr2t1c1d", "oh0smogc"}

        self.movement_graph = {}  # tile -> set(neighbor_tiles)
        self.paint_adjacency = {}  # target_tile -> set(robot_tiles)
        self.available_colors = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred in movement_preds and len(parts) >= 3:
                a, b = parts[1], parts[2]
                # edge b -> a (robot at b can move to a)
                self.movement_graph.setdefault(b, set()).add(a)
                # ensure a and b exist as keys for robustness
                self.movement_graph.setdefault(a, set())
            if pred in paint_preds and len(parts) >= 3:
                target, robot_tile = parts[1], parts[2]
                self.paint_adjacency.setdefault(target, set()).add(robot_tile)
            if pred == "cdb9op3o" and len(parts) >= 2:
                # available color
                self.available_colors.add(parts[1])

        # make sure keys exist for any isolated tiles referenced in paint adjacency
        for target, neighs in list(self.paint_adjacency.items()):
            for n in neighs:
                self.movement_graph.setdefault(n, set())
            self.movement_graph.setdefault(target, set())

        # A large but finite fallback cost for unreachable goals
        self.UNREACHABLE_COST = max(100, 10 * len(self.goals))

    def __call__(self, node):
        state = set(node.state)

        # If all goals already satisfied, return 0 (check 2 from checklist)
        if self.goals <= state:
            return 0

        # Extract state information
        robot_positions = {}  # robot -> tile
        robot_colors = {}  # robot -> color
        free_tiles = set()  # tiles with d7ekk7vb
        painted = {}  # tile -> color where yh6q5j8r holds

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "gk9u3vg3" and len(parts) >= 3:
                # (gk9u3vg3 robot tile)
                robot, tile = parts[1], parts[2]
                robot_positions[robot] = tile
            elif pred == "i4u94rl9" and len(parts) >= 3:
                # (i4u94rl9 robot color)
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif pred == "d7ekk7vb" and len(parts) >= 2:
                # (d7ekk7vb tile) -> tile is free
                free_tiles.add(parts[1])
            elif pred == "yh6q5j8r" and len(parts) >= 3:
                # (yh6q5j8r tile color) -> painted
                tile, color = parts[1], parts[2]
                painted[tile] = color

        # Parse required goals into mapping tile->color
        goal_map = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts:
                continue
            pred = parts[0]
            if pred == "yh6q5j8r" and len(parts) >= 3:
                tile, color = parts[1], parts[2]
                goal_map[tile] = color

        # Build set of unsatisfied goal tiles
        unsatisfied = []
        for tile, color in goal_map.items():
            if tile not in painted or painted[tile] != color:
                unsatisfied.append((tile, color))

        # Helper: BFS to find minimal moves from start_tile to any tile in targets.
        def bfs_min_distance(start_tile, targets, traversable_set):
            if start_tile in targets:
                return 0
            visited = {start_tile}
            q = deque([(start_tile, 0)])
            while q:
                cur, dist = q.popleft()
                for neigh in self.movement_graph.get(cur, ()):
                    if neigh in visited:
                        continue
                    # Consider neigh traversable under optimistic assumption:
                    if neigh in traversable_set:
                        if neigh in targets:
                            return dist + 1
                        visited.add(neigh)
                        q.append((neigh, dist + 1))
                # Also consider nodes that may have no outgoing edges; loop continues.
            return None  # unreachable under current optimistic assumptions

        total_est = 0

        # Precompute the set of tiles we treat as traversable: free tiles + any robot-occupied tiles
        # (optimistic assumption)
        traversable_tiles = set(free_tiles) | set(robot_positions.values())

        # For each unsatisfied goal tile compute minimal cost to paint
        for target_tile, req_color in unsatisfied:
            # If there is no known painting-adjacent tile, this target is hard to paint
            adj_positions = self.paint_adjacency.get(target_tile, set())
            if not adj_positions:
                total_est += self.UNREACHABLE_COST
                continue

            best_cost_for_target = None

            # If target is currently not free, optimistic small penalty (we add below)
            free_penalty = 0 if target_tile in free_tiles else 1

            # Try each robot as a candidate painter
            for robot, start_tile in robot_positions.items():
                # Distance from robot start to any tile in adj_positions
                dist = bfs_min_distance(start_tile, adj_positions, traversable_tiles)
                if dist is None:
                    # unreachable by this robot under optimistic model
                    continue
                color_change = 0 if robot_colors.get(robot) == req_color else 1
                # 1 for the paint action itself
                cost = dist + color_change + 1 + free_penalty
                if best_cost_for_target is None or cost < best_cost_for_target:
                    best_cost_for_target = cost

            if best_cost_for_target is None:
                # No robot could reach the painting-adjacent tiles under optimistic model
                total_est += self.UNREACHABLE_COST
            else:
                total_est += best_cost_for_target

        # Return a finite integer heuristic (checklist item 3 and 4)
        return int(total_est)