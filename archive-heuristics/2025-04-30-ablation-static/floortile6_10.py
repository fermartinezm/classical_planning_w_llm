from heuristics.heuristic_base import Heuristic

import re
from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Handle cases where fact has fewer parts than args (e.g., "(clear tile_1_1)" vs pattern "painted", "*", "*")
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_name(tile_name):
    """Parses 'tile_R_C' into (R, C) integer coordinates."""
    match = re.match(r"tile_(\d+)_(\d+)", tile_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    # Handle non-grid tiles if necessary, though problem description implies grid
    # For robustness, return None or raise error
    return None

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles given their names."""
    coord1 = parse_tile_name(tile1_name)
    coord2 = parse_tile_name(tile2_name)
    if coord1 is None or coord2 is None:
        # This should not happen with valid tile names in this domain
        # Returning a large value indicates an issue or unreachable state
        return float('inf')
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing up
    estimated costs for each unsatisfied goal predicate `(painted tile color)`.
    For each tile that needs to be painted with a specific color but isn't,
    the heuristic estimates the minimum cost for any robot to paint that tile.
    This minimum cost is estimated as the sum of:
    1. Cost to get the correct color (1 if the robot doesn't have it, 0 otherwise).
    2. Cost to move the robot to a tile adjacent to the target tile (Manhattan distance).
    3. Cost of the paint action itself (1).
    Tiles painted with the wrong color are considered unreachable goals from that state,
    and contribute a large penalty.

    # Assumptions:
    - Tiles are named in the format `tile_R_C` allowing coordinate parsing for Manhattan distance.
    - The grid structure defined by `up`, `down`, `left`, `right` facts corresponds to these coordinates.
    - All colors required by the goal are available (`available-color`) at least initially.
    - A robot always holds some color (no `free-color` state to pick up first).
    - Tiles painted with the wrong color cannot be repainted or cleared according to the domain definition.

    # Heuristic Initialization
    - Extract goal conditions `(painted tile color)`.
    - Build an adjacency map for tiles based on `up`, `down`, `left`, `right` static facts.
    - Identify available colors from static facts and initial state.
    - Identify robots from initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Identify all goal predicates of the form `(painted T C)`.
    3. For each such goal predicate `(painted T C)`:
        a. Check the current state of tile `T`:
           - If `(painted T C)` is already in the state, this goal is satisfied; continue to the next goal.
           - If `(painted T C')` where `C' != C` is in the state, the tile is painted the wrong color. This goal is likely unreachable. Add a large penalty (e.g., 1000) to the total cost and continue to the next goal.
           - If `(clear T)` is in the state, the tile needs painting. Proceed to step 3b.
           - If the tile `T` is in any other state (e.g., not clear, not painted - which shouldn't happen in a valid state based on domain predicates), skip or penalize.
        b. If `T` is `clear` and needs color `C`:
           - Find the minimum cost among all robots to paint tile `T` with color `C`.
           - Initialize `min_cost_for_this_goal` to infinity.
           - For each robot `R`:
             - Determine robot `R`'s current location `LocR` and color `ColorR` from the state.
             - Calculate `color_cost`: 1 if `ColorR` is not `C`, and `C` is in `available_colors`; 0 if `ColorR` is `C`. If `C` is not available, this robot cannot paint it, cost is infinity for this robot.
             - Calculate `movement_cost_adjacent`: Find all tiles `X` adjacent to `T` using the pre-computed adjacency map. Calculate `ManhattanDistance(LocR, X)` for each adjacent tile `X`. The `movement_cost_adjacent` is the minimum of these distances. If `LocR` is itself adjacent to `T`, the cost is 0. If `T` has no adjacent tiles, movement cost is infinity.
             - If `color_cost` is not infinity and `movement_cost_adjacent` is not infinity:
               - Cost for robot `R` to paint `T` = `color_cost + movement_cost_adjacent + 1` (for the paint action).
               - Update `min_cost_for_this_goal = min(min_cost_for_this_goal, cost_r)`.
           - If `min_cost_for_this_goal` is still infinity (no robot could paint this tile), add a large penalty (e.g., 1000) to the total cost.
           - Otherwise, add `min_cost_for_this_goal` to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and building data structures."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Extract goal painted conditions: {(tile, color)}
        self.goal_painted = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                self.goal_painted.add((parts[1], parts[2]))

        # Build adjacency map: tile -> set(neighbors)
        self.adjacency_map = {}
        # Combine static and initial state facts to build the map, as adjacency facts are static
        for fact in static_facts | initial_state:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                # PDDL fact is (relation tile1 tile2) where tile1 is relative to tile2
                # e.g., (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1
                # This implies tile_0_1 is down from tile_1_1
                # Let's build bidirectional map
                rel, tile_a, tile_b = parts
                self.adjacency_map.setdefault(tile_a, set()).add(tile_b)
                self.adjacency_map.setdefault(tile_b, set()).add(tile_a) # Connections are bidirectional

        # Identify available colors
        self.available_colors = set()
        # Available colors might be in static facts or initial state
        for fact in static_facts | initial_state:
             parts = get_parts(fact)
             if parts[0] == "available-color":
                 self.available_colors.add(parts[1])

        # Identify robots (assuming they are in initial state with robot-at or robot-has)
        self.robots = set()
        for fact in initial_state:
            parts = get_parts(fact)
            if parts[0] in ["robot-at", "robot-has"]:
                self.robots.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Track current robot locations and colors
        robot_locations = {} # robot -> tile
        robot_colors = {}    # robot -> color
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot_locations[parts[1]] = parts[2]
            elif parts[0] == "robot-has":
                robot_colors[parts[1]] = parts[2]

        # Track current tile status (clear or painted)
        tile_status = {} # tile -> 'clear' or ('painted', color)
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "clear":
                tile_status[parts[1]] = 'clear'
            elif parts[0] == "painted":
                 tile_status[parts[1]] = ('painted', parts[2])


        total_cost = 0
        PENALTY = 1000 # Large penalty for unreachable goals

        # Iterate through all goal painted conditions
        for (target_tile, target_color) in self.goal_painted:

            # Check if goal is already satisfied
            if ('painted', target_color) == tile_status.get(target_tile):
                 continue # Goal already met for this tile

            # Check the current status of the target tile
            current_status = tile_status.get(target_tile)

            # Case 1: Tile is painted the wrong color
            if isinstance(current_status, tuple) and current_status[0] == 'painted' and current_status[1] != target_color:
                 # This goal is likely unreachable from this state
                 total_cost += PENALTY
                 continue

            # Case 2: Tile is clear and needs painting
            if current_status == 'clear':
                min_cost_for_this_goal = float('inf')

                # Calculate minimum cost over all robots
                for robot in self.robots:
                    current_robot_location = robot_locations.get(robot)
                    current_robot_color = robot_colors.get(robot)

                    # Should always have location and color in a valid state for active robots
                    if current_robot_location is None or current_robot_color is None:
                         continue # Skip this robot if its state is incomplete

                    # Calculate color change cost
                    color_cost = 0
                    if current_robot_color != target_color:
                        if target_color in self.available_colors:
                             color_cost = 1
                        else:
                             # Target color is not available, this robot cannot change to it.
                             color_cost = float('inf') # This robot cannot get the color

                    # Calculate movement cost to an adjacent tile
                    movement_cost_adjacent = float('inf')
                    adjacent_tiles = self.adjacency_map.get(target_tile, set())

                    if not adjacent_tiles:
                         # Target tile has no neighbors, cannot be painted.
                         # This goal is likely impossible.
                         movement_cost_adjacent = float('inf') # Mark as impossible for movement
                    else:
                        for adj_tile in adjacent_tiles:
                            dist = manhattan_distance(current_robot_location, adj_tile)
                            if dist != float('inf'): # Ensure valid tile names were parsed
                                movement_cost_adjacent = min(movement_cost_adjacent, dist)
                        # If movement_cost_adjacent is still inf, it means all adjacent tiles had invalid names, unlikely.

                    # Paint action cost
                    paint_cost = 1

                    # Total cost for this robot to paint this tile
                    # Only consider if both color and movement are possible
                    if color_cost != float('inf') and movement_cost_adjacent != float('inf'):
                        cost_r = color_cost + movement_cost_adjacent + paint_cost
                        min_cost_for_this_goal = min(min_cost_for_this_goal, cost_r)
                    # else: cost_r is effectively infinity for this robot

                # Add the minimum cost found for this goal to the total
                if min_cost_for_this_goal != float('inf'):
                    total_cost += min_cost_for_this_goal
                else:
                    # If no robot could paint this tile (e.g., no robot, no adjacent tiles, color unavailable to all)
                    total_cost += PENALTY # Large penalty

            # Case 3: Tile is not clear, not painted correctly, and not painted wrong color.
            # This implies the tile is not mentioned in clear or painted facts.
            # In a well-formed state, a tile is either clear or painted.
            # If a goal tile is not in tile_status, it's an anomaly. Treat as unreachable.
            elif target_tile not in tile_status:
                 total_cost += PENALTY


        return total_cost
