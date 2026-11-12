from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the "color" (secret_domain) PDDL domain.
    The domain's goal is to produce (painted <tile-2>) facts. Painting a tile-2
    requires a sequence of actions that (a) align a robot's available-color
    with the tile's available-color at the same tile position, (b) ensure the
    robot is 'at' a color that is currently free (free-color), and (c) execute
    paint_up. Robots can move their available-color forward along the static
    up graph using change_color; move_down sets the robot-at relation when the
    robot's available-color and a color's available-color coincide at the same
    tile position.

    This heuristic estimates the number of actions needed to reach the goal by
    planning, in a fast (greedy) simulated fashion, how to bring robots and
    free colors together with each target tile to perform paint_up. It uses
    the static "up" graph to compute minimal numbers of change_color steps
    required and accounts for move_down and paint_up actions. The heuristic
    simulates the consumption of free-color resources (a free color is removed
    after a simulated paint) and updates robot available-color positions after
    simulated change_color moves to produce informed estimates for subsequent
    tiles.

    Assumptions
    -----------
    - The only relevant way to modify positions for robots' available colors is
      by change_color (which moves a robot's available-color forward along an
      up edge).
    - Colors that appear as (free-color <c>) are the free color resources that
      must exist at paint time. After a paint_up, the used free-color becomes
      unavailable (paint_up's effect removes the free-color). The heuristic
      simulates this consumption.
    - Color and tile available-color facts for non-robot entities do not change
      through actions considered here (we do not simulate changing their
      available-color positions).
    - If no sequence of change_color steps can align a robot with a tile's
      available-color position and a free color at the same position, the tile
      is considered difficult/unreachable and a large finite penalty is used.
    - The heuristic is not required to be admissible and is intentionally
      optimistic/greedy to guide a best-first search effectively.

    Heuristic Initialization
    ------------------------
    The constructor extracts:
    - self.goals: the set of goal facts (task.goals) so that the heuristic
      returns zero whenever the node's state satisfies the goals.
    - The static up graph: a directed adjacency mapping (from tile to successor
      tile) derived from static facts of the form '(up X Y)'. This graph is
      used to compute shortest forward paths (number of change_color steps)
      from a robot's current available-color position to a target tile
      position.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For each call on a node/state:
    1. If the state's facts already satisfy all goals, return 0.
    2. Parse the state to collect:
       - available_positions_by_entity: mapping entity -> set(positions) from
         '(available-color ENTITY POS)' facts.
       - robot_at_map: mapping robot -> set(colors) from '(robot-at ROBOT COLOR)'.
       - free_colors: set of colors currently free from '(free-color COLOR)'.
       - painted_set and down_set to know which tiles are already painted/down.
    3. Identify which goal tiles (painted <tile-2>) are still unpainted.
    4. For each unpainted tile, compute the minimal local plan cost to paint it:
       - Find all positions P where the tile's available-color is located.
       - For each free color c (from free_colors), require that c's available-color
         is also at some position p in P (colors' available positions are not
         changed by actions and thus must already be present at p).
       - For each robot r, find robot's current available positions q and compute
         the shortest forward path along the 'up' graph from q to p (BFS).
         The number of change_color actions equals the path length (number of edges).
       - If the robot is already robot-at r c, we do not need move_down; otherwise
         one move_down is required (provided the robot's available-color is at p
         and the color's available-color is at p).
       - One paint_up action is always required to produce (painted tile).
       - So cost = (#change_color steps) + (0 or 1 for move_down) + 1 for paint_up.
       - Among all choices (robot r, free color c, positions p, robot start q)
         choose the minimal cost local plan.
    5. To avoid unrealistically reusing a single free color for many paints, the
       heuristic greedily selects the currently cheapest tile to paint first,
       applies the chosen local plan in a short simulation (updates the robot's
       available-position to the plan's target p, marks the color as consumed,
       marks the tile as painted), adds the plan cost to the total estimate, and
       repeats the computation for remaining tiles. This simulated execution is
       cheap and yields a more informed aggregate estimate than a simple sum of
       independent costs.
    6. If for a tile no feasible plan is found given the current simulated
       resources (robots, free colors and positions), a large finite penalty is
       used to represent difficulty while keeping the heuristic finite.
    """

    def __init__(self, task):
        # store goals (set of fact strings)
        self.goals = task.goals
        # static facts: extract up edges
        self.up_graph = defaultdict(list)  # node -> list of successors (forward along up)
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "up" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.up_graph[src].append(dst)

    def _bfs_path(self, start, target):
        """Return the forward path (list of nodes) from start to target along up_graph, or None."""
        if start == target:
            return [start]
        visited = {start}
        queue = deque([[start]])
        while queue:
            path = queue.popleft()
            node = path[-1]
            for succ in self.up_graph.get(node, []):
                if succ in visited:
                    continue
                visited.add(succ)
                new_path = path + [succ]
                if succ == target:
                    return new_path
                queue.append(new_path)
        return None

    def __call__(self, node):
        state = node.state

        # If goals already satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Parse state facts
        available_positions_by_entity = defaultdict(set)  # entity -> set(positions)
        robot_at_map = defaultdict(set)  # robot -> set(colors)
        free_colors = set()
        painted = set()
        down_tiles = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "available-color" and len(parts) >= 3:
                ent, pos = parts[1], parts[2]
                available_positions_by_entity[ent].add(pos)
            elif pred == "robot-at" and len(parts) >= 3:
                robot, color = parts[1], parts[2]
                robot_at_map[robot].add(color)
            elif pred == "free-color" and len(parts) >= 2:
                free_colors.add(parts[1])
            elif pred == "painted" and len(parts) >= 2:
                painted.add(parts[1])
            elif pred == "down" and len(parts) >= 2:
                down_tiles.add(parts[1])

        # Determine which tile-2 goals remain unpainted
        remaining_tiles = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) >= 2:
                tile = parts[1]
                if tile not in painted:
                    remaining_tiles.append(tile)

        # If no remaining goals, return 0 (should have been caught earlier)
        if not remaining_tiles:
            return 0

        # Identify robots: any entity that appears as first arg in a robot-at predicate
        robots = list(robot_at_map.keys())

        # Precompute tile available positions (for those tiles to paint)
        tile_positions = {t: available_positions_by_entity.get(t, set()) for t in remaining_tiles}

        # Large finite penalty for unreachable cases
        LARGE = 1000

        # Make mutable simulation copies of resources
        sim_available = {k: set(v) for k, v in available_positions_by_entity.items()}
        sim_robot_at = {k: set(v) for k, v in robot_at_map.items()}
        sim_free_colors = set(free_colors)
        sim_remaining = set(remaining_tiles)

        total_estimate = 0

        # Greedily simulate painting the easiest remaining tile until none left
        while sim_remaining:
            # For each tile compute the best plan under current simulated resources
            best_plan = None  # tuple (cost, tile, robot, color, start_pos, target_pos, path)
            for tile in sim_remaining:
                # If tile is not down in the current real state, painting is likely impossible.
                # The domain has no action to re-establish down, so set an unreachable cost.
                if tile not in down_tiles:
                    # give it a large penalty but still finite
                    tile_best_cost = LARGE
                    plan = (LARGE, tile, None, None, None, None, None)
                    if best_plan is None or tile_best_cost < best_plan[0]:
                        best_plan = plan
                    continue

                positions_for_tile = tile_positions.get(tile, set())
                if not positions_for_tile:
                    # No available-color positions for the tile -> unreachable in this model
                    tile_best_cost = LARGE
                    plan = (LARGE, tile, None, None, None, None, None)
                    if best_plan is None or tile_best_cost < best_plan[0]:
                        best_plan = plan
                    continue

                # Try all combinations robot, free color, tile positions and robot start positions
                tile_best_cost = None
                tile_best_plan = None

                # If no free colors available at all, painting any further tile is impossible
                if not sim_free_colors:
                    tile_best_cost = LARGE
                    tile_best_plan = (LARGE, tile, None, None, None, None, None)
                    if best_plan is None or tile_best_cost < best_plan[0]:
                        best_plan = tile_best_plan
                    continue

                for c in sim_free_colors:
                    # color c must have available-position at the same tile position p to be usable for move_down/paint
                    color_positions = sim_available.get(c, set())
                    # consider only p that are both in positions_for_tile and color_positions
                    candidate_positions = positions_for_tile & color_positions
                    if not candidate_positions:
                        continue
                    for p in candidate_positions:
                        # For each robot, consider moving robot's available-color to p
                        for r in robots:
                            robot_positions = sim_available.get(r, set())
                            if not robot_positions:
                                continue
                            # for each starting pos q compute path q->p
                            for q in robot_positions:
                                path = self._bfs_path(q, p)
                                if path is None:
                                    # not reachable along up graph
                                    continue
                                change_steps = len(path) - 1  # number of change_color actions
                                already_robot_at = (c in sim_robot_at.get(r, set()))
                                move_down_needed = 0 if already_robot_at else 1
                                # paint_up always needed
                                cost = change_steps + move_down_needed + 1
                                if tile_best_cost is None or cost < tile_best_cost:
                                    tile_best_cost = cost
                                    tile_best_plan = (cost, tile, r, c, q, p, path)

                if tile_best_cost is None:
                    # no feasible plan found for this tile under current simulation state
                    tile_best_cost = LARGE
                    tile_best_plan = (LARGE, tile, None, None, None, None, None)

                if best_plan is None or tile_best_cost < best_plan[0]:
                    best_plan = tile_best_plan

            # After evaluating all tiles, apply the best plan found
            if best_plan is None:
                # Should not happen, but guard to avoid infinite loop
                total_estimate += LARGE
                break

            cost, tile, robot, color, start_pos, target_pos, path = best_plan

            if cost >= LARGE:
                # Unreachable tile under current assumptions; add penalty and stop
                total_estimate += cost
                break

            # Apply the simulated plan effects:
            # 1) move robot's available-color along the path: remove start_pos, add target_pos
            #    (we only move one starting position; robots may have multiple available positions,
            #     but change_color acts on a specific available-color instance)
            if robot is not None and start_pos is not None and target_pos is not None:
                # remove the start_pos occurrence for that robot (if present) and add target_pos
                if start_pos in sim_available.get(robot, set()):
                    sim_available[robot].discard(start_pos)
                    sim_available[robot].add(target_pos)

            # 2) if move_down was needed, update sim_robot_at
            if color is not None and robot is not None and (color not in sim_robot_at.get(robot, set())):
                sim_robot_at.setdefault(robot, set()).add(color)

            # 3) consume the free color used
            if color in sim_free_colors:
                sim_free_colors.remove(color)

            # 4) mark tile as painted (remove from remaining)
            if tile in sim_remaining:
                sim_remaining.remove(tile)

            # 5) add the cost to total estimate
            total_estimate += cost

        return total_estimate